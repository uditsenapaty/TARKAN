"""Training (paper Algorithm 1, Eq. 25).

Loads cached teacher labels (no LLM at train time), runs the student forward, computes
L = L_tag + λ1 L_rel + λ2 L_kg + λ3 L_asc, early-stops on dev joint-F1.

CPU-runnable for smoke tests; the real runs go on the T4 server (set CONFIG.device='cuda').
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import CONFIG
from data import TarkanDataset, collate_fn, load_split
from evaluate import evaluate_all
from losses import compute_losses
from models import TarkanStudent
from seeding import seed_everything, worker_init_fn
from teacher import TeacherCache, build_targets
from utils import get_logger, save_checkpoint

log = get_logger("train")


def build_kg():
    """Load the built KG index if present; else return None (KG stream becomes inert)."""
    from kg import KnowledgeGraph

    sqlite = CONFIG.paths.kg_index / "kg.sqlite"
    if sqlite.exists():
        return KnowledgeGraph(sqlite_path=str(sqlite))
    return None


def make_loader(dataset: str, split: str, cfg, shuffle: bool, captions=None) -> DataLoader:
    data_dir = cfg.paths.data / dataset
    images = cfg.paths.data / "images" / dataset
    insts = load_split(data_dir, split)
    ds = TarkanDataset(insts, cfg, captions=captions, images_dir=images)
    # P2 (OBEYING, result-neutral): overlap data prep with the GPU step via worker
    # processes (seeded by worker_init_fn, so deterministic). num_workers=0 on CPU.
    nw = getattr(cfg, "num_workers", 0) if cfg.device == "cuda" else 0
    kwargs = dict(batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    if nw > 0:
        kwargs.update(num_workers=nw, pin_memory=True, persistent_workers=True)
    return DataLoader(ds, **kwargs)


def train(cfg=CONFIG, dataset: str = "twitter2015", max_epochs: Optional[int] = None) -> dict:
    seed_everything(cfg.seed)
    device = cfg.device
    max_epochs = max_epochs or cfg.max_epochs

    kg = build_kg()
    entity_embedder = None
    nb = cfg.paths.conceptnet / "numberbatch-en.txt"
    if nb.exists():
        from kg_retrieval import EntityEmbedder

        entity_embedder = EntityEmbedder.from_txt(str(nb))

    model = TarkanStudent(cfg, kg=kg, entity_embedder=entity_embedder,
                          pool_mode=getattr(cfg, "pool_mode", "mean")).to(device)
    cache = TeacherCache.load(dataset)

    train_loader = make_loader(dataset, "train", cfg, shuffle=True)
    dev_loader = make_loader(dataset, "dev", cfg, shuffle=False)

    # A1 (DISOBEYING, opt-in): inverse-frequency BIO class weights over the FULL train split
    # (computed once so it's stable, not batch-noisy). Inert unless cfg.tag_class_weight.
    if getattr(cfg, "tag_class_weight", False):
        from config import NUM_BIO_TAGS, BIO_TAGS
        counts = torch.zeros(NUM_BIO_TAGS)
        for b in train_loader:
            lab = b["bio_labels"].reshape(-1)
            counts += torch.bincount(lab[lab != -100], minlength=NUM_BIO_TAGS).float()
        # A1 (O-preserving): keep O at 1.0 so extraction is unaffected; inverse-frequency
        # balance ONLY the 6 aspect tags (up-weight rare NEG) normalized to mean 1 among
        # themselves. Full 7-class inverse-freq crushes O (~0.02) -> over-tagging -> MATE
        # collapse; the real bottleneck is POS/NEU/NEG balance *within* aspects.
        w = torch.ones(NUM_BIO_TAGS)
        asp = [i for i, t in enumerate(BIO_TAGS) if t != "O"]
        ac = counts[asp]
        iw = ac.sum() / (ac + 1e-6)
        iw = iw / iw.mean()
        for j, i in enumerate(asp):
            w[i] = iw[j]
        cfg._tag_weight_vec = w.to(device)
        log.info(f"A1 class weights (O-preserving, polarity-balanced): {[round(x,2) for x in cfg._tag_weight_vec.tolist()]}")

    # A3 (DISOBEYING, opt-in): discriminative LR — encoders at learning_rate, fresh modules higher.
    if getattr(cfg, "layerwise_lr", None):
        enc_ids = set()
        enc_params, new_params = [], []
        for mod in (model.text_encoder, model.visual_encoder):
            if mod is not None:
                for p in mod.parameters():
                    enc_ids.add(id(p)); enc_params.append(p)
        for p in model.parameters():
            if id(p) not in enc_ids:
                new_params.append(p)
        optim = torch.optim.AdamW(
            [{"params": enc_params, "lr": cfg.learning_rate},
             {"params": new_params, "lr": cfg.layerwise_lr}],
            weight_decay=cfg.weight_decay,
        )
        log.info(f"A3 layerwise LR: encoders={cfg.learning_rate}, fresh modules={cfg.layerwise_lr}")
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    accum = max(1, int(getattr(cfg, "grad_accum", 1)))
    steps_per_epoch = max(1, (len(train_loader) + accum - 1) // accum)
    total_steps = max_epochs * steps_per_epoch
    warmup = int(cfg.warmup_ratio * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, s / max(1, warmup)) if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    )

    best_f1, best_state, patience = -1.0, None, 0
    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        optim.zero_grad()
        pending = False
        for i, batch in enumerate(train_loader):
            batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(batch_dev)
            targets = build_targets(batch_dev, outputs, cache, cfg)
            losses = compute_losses(outputs, targets, cfg, model=model)
            (losses["total"] / accum).backward()
            pending = True
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad()
                pending = False
            running += float(losses["total"].item())
        if pending:  # flush trailing partial accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            optim.zero_grad()

        metrics = evaluate_all(model, dev_loader, device)
        dev_f1 = metrics["joint"]["F1"]
        log.info(f"epoch {epoch}: train_loss={running/max(1,len(train_loader)):.4f} dev_joint_F1={dev_f1:.2f} {metrics}")

        if dev_f1 > best_f1:
            best_f1, patience = dev_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(model, cfg.paths.checkpoints / f"{dataset}_best.pt", {"dev_f1": dev_f1, "epoch": epoch})
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                log.info(f"early stop at epoch {epoch} (best dev F1 {best_f1:.2f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_dev_f1": best_f1, "model": model}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--device", default=CONFIG.device)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lambda1", type=float, default=CONFIG.lambda1)
    ap.add_argument("--lambda2", type=float, default=CONFIG.lambda2)
    ap.add_argument("--fusion", default=CONFIG.fusion)
    args = ap.parse_args()
    cfg = replace(CONFIG, device=args.device, lambda1=args.lambda1, lambda2=args.lambda2, fusion=args.fusion)
    res = train(cfg, dataset=args.dataset, max_epochs=args.epochs)
    log.info(f"done. best dev joint-F1 = {res['best_dev_f1']:.2f}")
