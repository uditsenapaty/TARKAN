"""One-time offline teacher-labeling pass (paper Algorithm 1 lines 7-11, Table 4).

Runs the captioner + LLM teacher over a split to produce r^T_k and s^T_kq, cached to
data/teacher_labels/. RUN THIS ON THE T4 SERVER (heavy). Training then reads the cache
and never loads the LLM. Resumable: already-labeled (instance, aspect[, triple]) skipped.

P1 (OBEYING, result-neutral): prompts are collected then scored in GPU batches via the
teacher's left-padded batched generate() (greedy -> identical {0,1} labels). Triples are
scored independently (no joint-prompt collapse) so the labeling semantics are unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG  # noqa: E402
from captioner import Captioner  # noqa: E402
from data import build_queries, load_split  # noqa: E402
from kg import KnowledgeGraph  # noqa: E402
from kg_retrieval import _triple_key as triple_key, retrieve_triples  # noqa: E402
from teacher import LLMTeacher, TeacherCache  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("teacher_labeling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--device", default=CONFIG.device)
    ap.add_argument("--limit", type=int, default=None, help="cap instances (debug)")
    ap.add_argument("--batch-size", type=int, default=16, help="P1 LLM generate() batch size")
    ap.add_argument("--save-every", type=int, default=4000, help="checkpoint cache every N prompts")
    args = ap.parse_args()

    CONFIG.device = args.device
    images_dir = CONFIG.paths.data / "images" / args.dataset
    sqlite = CONFIG.paths.kg_index / "kg.sqlite"
    kg = KnowledgeGraph(sqlite_path=str(sqlite)) if sqlite.exists() else None
    if kg is None:
        log.warning("kg.sqlite missing -> KG usefulness labels skipped (L_kg will be 0 in training)")
    captioner = Captioner(device=args.device)
    teacher = LLMTeacher(device=args.device)

    prev = TeacherCache.load(args.dataset)

    # ---- Phase A: caption (cache per image_id) + collect un-cached prompts ----
    caption_cache: dict = {}
    rel_items, rel_keys = [], []          # (tweet, term, caption), (instance_id, aspect_idx)
    kg_items, kg_keys = [], []            # (tweet, term, triple),  (instance_id, aspect_idx, triple_key)

    for split in args.splits:
        insts = load_split(CONFIG.paths.data / args.dataset, split)
        if args.limit:
            insts = insts[: args.limit]
        log.info(f"{split}: {len(insts)} instances — captioning + collecting prompts")
        for n, inst in enumerate(insts):
            tweet = " ".join(inst.tokens)
            if inst.image_id not in caption_cache:
                try:
                    caption_cache[inst.image_id] = captioner.caption_image(images_dir / inst.image_id)
                except Exception:
                    caption_cache[inst.image_id] = ""
            caption = caption_cache[inst.image_id]
            queries = build_queries(inst, {inst.image_id: caption})
            for k, ((s, e, pol), term) in enumerate(zip(inst.aspects, inst.aspect_terms)):
                if (inst.id, k) not in prev.rel:
                    rel_items.append((tweet, term, caption))
                    rel_keys.append((inst.id, k))
                if kg is not None:
                    for tr in retrieve_triples(queries[k], kg, CONFIG.top_m_triples):
                        tk = triple_key(tr)
                        if (inst.id, k, tk) in prev.kg:
                            continue
                        kg_items.append((tweet, term, tr))
                        kg_keys.append((inst.id, k, tk))
            if (n + 1) % 500 == 0:
                log.info(f"  collected {n+1}/{len(insts)} (rel={len(rel_items)} kg={len(kg_items)})")

    # ---- Phase B: batched labeling with incremental checkpointing ----
    # A multi-hour run must survive a hiccup: flush the merged cache every `save_every`
    # prompts so re-running resumes (the (instance,aspect[,triple]) skip-set above
    # already de-dups against whatever was saved). Saving rewrites the parquet from the
    # full accumulated set, so partial progress is never lost.
    log.info(f"labeling {len(rel_items)} relevance + {len(kg_items)} kg prompts "
             f"(batch={args.batch_size}, save_every={args.save_every})")

    base_rel = [{"instance_id": i, "aspect_idx": k, "label": v} for (i, k), v in prev.rel.items()]
    base_kg = [{"instance_id": i, "aspect_idx": k, "triple_key": t, "label": v} for (i, k, t), v in prev.kg.items()]
    rel_rows, kg_rows = [], []

    def _flush():
        TeacherCache.save(args.dataset, base_rel + rel_rows, base_kg + kg_rows)

    done = 0
    for i in range(0, len(rel_items), args.batch_size):
        labs = teacher.relevance_label_batch(rel_items[i: i + args.batch_size], batch_size=args.batch_size)
        for (iid, k), v in zip(rel_keys[i: i + args.batch_size], labs):
            rel_rows.append({"instance_id": iid, "aspect_idx": k, "label": v})
        done += len(labs)
        if done % args.save_every < args.batch_size:
            _flush(); log.info(f"  relevance {len(rel_rows)}/{len(rel_items)} labeled (checkpointed)")
    _flush()

    done = 0
    for i in range(0, len(kg_items), args.batch_size):
        labs = teacher.kg_label_batch(kg_items[i: i + args.batch_size], batch_size=args.batch_size)
        for (iid, k, t), v in zip(kg_keys[i: i + args.batch_size], labs):
            kg_rows.append({"instance_id": iid, "aspect_idx": k, "triple_key": t, "label": v})
        done += len(labs)
        if done % args.save_every < args.batch_size:
            _flush(); log.info(f"  kg {len(kg_rows)}/{len(kg_items)} labeled (checkpointed)")
    _flush()
    log.info(f"saved {len(base_rel)+len(rel_rows)} relevance + {len(base_kg)+len(kg_rows)} kg labels for {args.dataset}")


if __name__ == "__main__":
    main()
