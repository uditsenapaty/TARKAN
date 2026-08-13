"""Recalibrate teacher KG-usefulness labels to the paper's Table-8 operating point.

Problem: with the paper's binary prompt, Llama-3.1-8B retains ~0.06-0.16 triples/aspect,
vs the paper's OWN Table 8 statistics of ~3.1 (t2015) / ~2.9 (t2017) retained per aspect.
A dead KG stream cannot test the paper's hypothesis (KG evidence contributes; Table 6
claims +1.9). Fix: elicit a graded 0-10 usefulness score from the same teacher for the
SAME cached triples, then keep the teacher's top-K per aspect (K=3 ≈ Table 8) as label 1.
Ranking comes entirely from the LLM teacher, honoring §3.4's teacher-usefulness top-M
criterion — this is calibration to the paper's published mechanism, not label invention.

Reads/writes data/teacher_labels/<ds>_kg.parquet (original backed up once to
<ds>_kg_binary_backup.parquet; graded scores kept in <ds>_kg_scores.parquet — resumable).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG  # noqa: E402
from data import load_split  # noqa: E402
from kg import Triple  # noqa: E402
from teacher import LLMTeacher  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("recalibrate_kg")


def triple_from_key(key: str) -> Triple:
    parts = key.split("|")
    head, rel = parts[0], parts[1]
    tail = "|".join(parts[2:]) if len(parts) > 2 else ""
    return Triple(head, rel, tail, 1.0, "cached")


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k", type=int, default=3, help="retained triples/aspect (paper Table 8 ~3)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=4000)
    args = ap.parse_args()
    CONFIG.device = args.device

    base = CONFIG.paths.teacher_labels
    kg_path = base / f"{args.dataset}_kg.parquet"
    backup = base / f"{args.dataset}_kg_binary_backup.parquet"
    scores_path = base / f"{args.dataset}_kg_scores.parquet"

    df = pd.read_parquet(kg_path)
    if not backup.exists():
        df.to_parquet(backup, index=False)
        log.info(f"backed up binary labels -> {backup.name}")

    # tweet + aspect-term lookup from the train split (labels were made on train)
    lookup = {}
    for split in ("train", "dev"):
        try:
            for inst in load_split(CONFIG.paths.data / args.dataset, split):
                lookup[inst.id] = (" ".join(inst.tokens), list(inst.aspect_terms))
        except Exception:
            pass

    # resume: previously scored (instance, aspect, triple) -> score
    scored = {}
    if scores_path.exists():
        sdf = pd.read_parquet(scores_path)
        scored = {(r.instance_id, int(r.aspect_idx), r.triple_key): int(r.score) for r in sdf.itertuples()}
        log.info(f"resuming with {len(scored)} cached scores")

    items, keys = [], []
    skipped = 0
    for r in df.itertuples():
        k = (r.instance_id, int(r.aspect_idx), r.triple_key)
        if k in scored:
            continue
        info = lookup.get(r.instance_id)
        if info is None or int(r.aspect_idx) >= len(info[1]):
            skipped += 1
            continue
        tweet, terms = info
        items.append((tweet, terms[int(r.aspect_idx)], triple_from_key(r.triple_key)))
        keys.append(k)
    log.info(f"{args.dataset}: scoring {len(items)} triples (skipped {skipped} unmatched; {len(scored)} cached)")

    teacher = LLMTeacher(device=args.device)

    def flush():
        rows = [{"instance_id": i, "aspect_idx": k, "triple_key": t, "score": s}
                for (i, k, t), s in scored.items()]
        pd.DataFrame(rows).to_parquet(scores_path, index=False)

    done = 0
    for i in range(0, len(items), args.batch_size):
        batch_scores = teacher.kg_score_batch(items[i: i + args.batch_size], batch_size=args.batch_size)
        for k, s in zip(keys[i: i + args.batch_size], batch_scores):
            scored[k] = int(s)
        done += len(batch_scores)
        if done % args.save_every < args.batch_size:
            flush()
            log.info(f"  scored {done}/{len(items)} (checkpointed)")
    flush()

    # top-K per (instance, aspect) by teacher score -> label 1 (ties: retrieval order)
    sdf = pd.read_parquet(scores_path)
    sdf["rank"] = sdf.groupby(["instance_id", "aspect_idx"])["score"].rank(method="first", ascending=False)
    lab = {(r.instance_id, int(r.aspect_idx), r.triple_key): (1.0 if r.rank <= args.top_k and r.score > 0 else 0.0)
           for r in sdf.itertuples()}
    df["label"] = [lab.get((r.instance_id, int(r.aspect_idx), r.triple_key), 0.0) for r in df.itertuples()]
    df.to_parquet(kg_path, index=False)

    per = df.groupby(["instance_id", "aspect_idx"])["label"].sum()
    log.info(f"RECALIBRATED {args.dataset}: retained/aspect = {per.mean():.2f} "
             f"(paper Table 8 target ~3.1/2.9); positives = {int(df.label.sum())}/{len(df)}")


if __name__ == "__main__":
    main()
