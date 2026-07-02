"""Table 10 — fusion-strategy ablation. Only the final fusion module changes; text/
visual encoders, KG retrieval, and teacher filtering are identical across variants.
Strategies: concat_linear, concat_mlp, gated, cross_modal_attention, bilinear, tensor, kan.
"""
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG  # noqa: E402
from kan_fusion import FUSION_REGISTRY  # noqa: E402
from train import train, make_loader  # noqa: E402
from evaluate import evaluate_all  # noqa: E402


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["twitter2015", "twitter2017"])
    ap.add_argument("--device", default=CONFIG.device)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    # Incremental + resumable (mirrors run_ablations.py): one row per (fusion, dataset),
    # appended immediately; completed cells skipped on restart.
    out = ROOT / "results" / "tables" / "ablation_fusion.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for r in csv.DictReader(open(out)):
            done.add((r["fusion"], r["dataset"]))
    write_header = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fusion", "dataset", "joint_P", "joint_R", "joint_F1"])
        if write_header:
            w.writeheader()
        for fusion in FUSION_REGISTRY.keys():
            for ds in args.datasets:
                if (fusion, ds) in done:
                    print(f"skip (done): {fusion} / {ds}")
                    continue
                cfg = replace(CONFIG, device=args.device, fusion=fusion)
                res = train(cfg, dataset=ds, max_epochs=args.epochs)
                m = evaluate_all(res["model"], make_loader(ds, "test", cfg, shuffle=False), cfg.device)
                row = {"fusion": fusion, "dataset": ds, "joint_P": round(m["joint"]["P"], 2),
                       "joint_R": round(m["joint"]["R"], 2), "joint_F1": round(m["joint"]["F1"], 2)}
                w.writerow(row)
                f.flush()
                print(row)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
