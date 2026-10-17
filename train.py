"""Train the six-loss student protocol; select checkpoints on dev joint F1."""
from __future__ import annotations
import argparse
import json
import platform
from dataclasses import replace
from importlib.metadata import version, PackageNotFoundError
import torch
from torch.utils.data import DataLoader
from config import CONFIG, load_config
from data import TarkanDataset, collate_fn, load_split
from evaluate import evaluate_all
from losses import compute_losses
from models import TarkanStudent
from seeding import seed_everything, worker_init_fn
from teacher import TeacherCache, build_targets
from utils import get_logger, save_checkpoint, save_json

log = get_logger("train")
ABLATIONS = {
    "full": {},
    "no_teacher": {"use_teacher": False},
    "no_aceq": {"lambda_cf": 0.0},
    "no_relevance": {"use_relevance": False},
    "no_kg_filter": {"use_kg_filter": False},
    "no_kg": {"use_kg_stream": False},
    "no_visual": {"use_visual_stream": False},
    "mlp_fusion": {"fusion": "concat_mlp"},
    "no_kan_tag": {"use_kan_tag_representation": False},
}

def check_device(cfg):
    if not str(cfg.device).startswith("cuda"):
        if cfg.require_a100:
            raise RuntimeError("The manuscript profile requires an NVIDIA A100 CUDA device.")
        return "CPU (smoke test only)"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the manuscript profile on an NVIDIA A100.")
    name = torch.cuda.get_device_name(torch.device(cfg.device))
    if cfg.require_a100 and "A100" not in name.upper():
        raise RuntimeError(f"Expected NVIDIA A100, found {name}.")
    return name

def build_kg(cfg=CONFIG):
    if not cfg.use_kg_stream:
        return None
    from kg import KnowledgeGraph
    path = cfg.paths.kg_index / "kg.sqlite"
    if not path.exists():
        raise FileNotFoundError(f"Missing KG index: {path}; see docs/reproducibility.md")
    return KnowledgeGraph(sqlite_path=str(path))

def load_visual_concepts(dataset, cfg=CONFIG):
    if not cfg.use_kg_stream or not cfg.use_visual_stream:
        return {}
    path = cfg.paths.data / "visual_concepts" / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing CLIP concept cache: {path}")
    with path.open(encoding="utf-8") as stream:
        concepts = json.load(stream)
    if not isinstance(concepts, dict) or any(not isinstance(v, list) for v in concepts.values()):
        raise ValueError("Visual concepts must map image IDs to ranked concept lists, not BLIP captions.")
    return concepts

def make_loader(dataset, split, cfg, shuffle=False, captions=None):
    instances = load_split(cfg.paths.data / dataset, split)
    if not instances:
        raise ValueError(f"Empty {dataset}/{split}")
    concepts = load_visual_concepts(dataset, cfg) if captions is None else captions
    if cfg.use_kg_stream and cfg.use_visual_stream:
        missing = {inst.image_id for inst in instances} - set(concepts)
        if missing:
            raise ValueError(f"CLIP concept cache is missing {len(missing)} images; regenerate it.")
    ds = TarkanDataset(instances, cfg, captions=concepts,
                       images_dir=cfg.paths.data / "images" / dataset,
                       with_images=cfg.use_visual_stream)
    workers = cfg.num_workers if str(cfg.device).startswith("cuda") else 0
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate_fn,
                      num_workers=workers, worker_init_fn=worker_init_fn,
                      pin_memory=bool(workers), persistent_workers=bool(workers))

def _fp32_outputs(outputs):
    for key in ("tag_logits", "anchor_logits", "asc_paper_logits", "relevance", "cf_relevance"):
        outputs[key] = outputs[key].float()
    for key in ("kg_scores", "cf_kg_scores"):
        outputs[key] = [s.float() for s in outputs[key]]
    return outputs

def train(cfg=CONFIG, dataset="twitter2015", max_epochs=None, run_name="full"):
    if max_epochs is not None:
        cfg = replace(cfg, max_epochs=max_epochs)
    gpu = check_device(cfg)
    seed_everything(cfg.seed)
    kg = build_kg(cfg)
    cache = TeacherCache.load(dataset, cfg) if cfg.use_teacher else None
    if cache is not None:
        if cfg.use_visual_stream and cfg.use_relevance and not cache.rel:
            raise FileNotFoundError("Missing teacher relevance labels; run prepare.py teacher first.")
        if cfg.use_kg_stream and cfg.use_kg_filter and not cache.kg:
            raise FileNotFoundError("Missing teacher KG labels; run prepare.py teacher first.")
    train_loader = make_loader(dataset, "train", cfg, True)
    dev_loader = make_loader(dataset, "dev", cfg)
    model = TarkanStudent(cfg, kg=kg).to(cfg.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps = cfg.max_epochs * len(train_loader)
    warmup = int(cfg.warmup_ratio * steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / max(1, warmup) if s < warmup else
        max(0.0, (steps-s) / max(1, steps-warmup)))
    amp = str(cfg.device).startswith("cuda") and cfg.use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run_dir = cfg.paths.results / dataset / run_name / f"seed_{cfg.seed}"
    checkpoint = run_dir / "best.pt"
    packages = {}
    for name in ("torch", "transformers", "numpy", "efficient-kan", "bitsandbytes"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    save_json(dict(config=cfg.to_dict(), python=platform.python_version(), gpu=gpu,
                   packages=packages, dataset=dataset, ablation=run_name), run_dir / "manifest.json")
    best, patience, history = -1.0, 0, []
    for epoch in range(cfg.max_epochs):
        model.train()
        running = {}
        for batch in train_loader:
            batch = {k: v.to(cfg.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                outputs = model(batch)
            outputs = _fp32_outputs(outputs)
            targets = build_targets(batch, outputs, cache, cfg)
            losses = compute_losses(outputs, targets, cfg)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + value.item()
        metrics = evaluate_all(model, dev_loader, cfg.device)
        history.append(dict(epoch=epoch+1, losses={k: v/len(train_loader) for k,v in running.items()},
                            dev=metrics))
        save_json(history, run_dir / "history.json")
        score = metrics["joint"]["F1"]
        log.info(f"epoch {epoch+1}: dev joint F1={score:.4f}")
        if score > best:
            best, patience = score, 0
            save_checkpoint(model, checkpoint, dict(config=cfg.to_dict(), epoch=epoch+1,
                                                     dev_f1=best, dataset=dataset))
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break
    payload = torch.load(checkpoint, map_location=cfg.device, weights_only=True)
    model.load_state_dict(payload["model_state"])
    return dict(best_dev_f1=best, model=model, checkpoint=str(checkpoint))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/information_fusion.yaml")
    ap.add_argument("--dataset", choices=("twitter2015", "twitter2017"), default="twitter2015")
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--ablation", choices=ABLATIONS, default="full")
    args = ap.parse_args()
    cfg = load_config(args.config, device=args.device, seed=args.seed, max_epochs=args.epochs)
    cfg = replace(cfg, **ABLATIONS[args.ablation])
    result = train(cfg, args.dataset, run_name=args.ablation)
    log.info(f"Saved {result['checkpoint']}; best dev joint F1={result['best_dev_f1']:.4f}")

