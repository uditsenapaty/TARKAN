"""Single tuning run with EXPLICIT patch overrides (keeps config.py = paper default).

Mutates the global CONFIG in place from CLI flags (so both cfg-readers and the few
global-CONFIG readers like KANFusion see the same values), trains one dataset, evaluates
on TEST, prints metrics, and appends a row to results/tables/iterations.csv with the exact
patch settings used. Every patch applied in the obeying->disobeying loop is logged here.

Example:
  python scripts/tune_run.py --dataset twitter2015 --device cuda --tag R1_obeying \
      --evidence-dropout 0.2 --kan-hidden 768 --patience 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="override CONFIG.seed (ensembling)")
    # ---- patch override knobs ----
    ap.add_argument("--evidence-dropout", type=float, default=None)
    ap.add_argument("--kan-hidden", default=None, help="comma list, e.g. 768 or 1024,512")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--lambda1", type=float, default=None)
    ap.add_argument("--lambda2", type=float, default=None)
    ap.add_argument("--pool-mode", default=None, help="mean|max|first|attn (aspect-span pooling)")
    # ---- disobeying patch toggles (applied only when explicitly requested) ----
    ap.add_argument("--class-weight", action="store_true", help="A1: inverse-freq weighted L_tag")
    ap.add_argument("--label-smoothing", type=float, default=None, help="A5: label smoothing on L_tag")
    ap.add_argument("--layerwise-lr", type=float, default=None, help="A3: LR for fresh (non-encoder) modules")
    ap.add_argument("--aux-asc-head", action="store_true", help="A7: dedicated ASC polarity head as polarity source")
    ap.add_argument("--lambda-asc", type=float, default=None, help="weight of L_asc when A7 on")
    ap.add_argument("--crf", action="store_true", help="A4: word-level linear-chain CRF for L_tag + Viterbi decode")
    ap.add_argument("--conf-append", action="store_true", help="A9: append [r, mean(s), max(s)] to KAN input")
    ap.add_argument("--feat-gate", action="store_true", help="A10: learnable (1+γ)⊙v, (1+δ)⊙g evidence gates")
    ap.add_argument("--reliability", action="store_true", help="A11: evidence reliability learning (softmax MoE over [t,v,g])")
    ap.add_argument("--text-model", default=None, help="A8: text encoder HF id (e.g. vinai/bertweet-large)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None, help="A8: grad accumulation steps")
    args = ap.parse_args()

    # ---- apply overrides to the global CONFIG singleton ----
    CONFIG.device = args.device
    if args.seed is not None:
        CONFIG.seed = args.seed
    if args.evidence_dropout is not None:
        CONFIG.evidence_dropout = args.evidence_dropout
    if args.kan_hidden is not None:
        CONFIG.kan_hidden = tuple(int(x) for x in args.kan_hidden.split(","))
    if args.patience is not None:
        CONFIG.early_stop_patience = args.patience
    if args.lr is not None:
        CONFIG.learning_rate = args.lr
    if args.dropout is not None:
        CONFIG.dropout = args.dropout
    if args.lambda1 is not None:
        CONFIG.lambda1 = args.lambda1
    if args.lambda2 is not None:
        CONFIG.lambda2 = args.lambda2
    # disobeying knobs are read by losses.py / train.py via these config attrs
    CONFIG.tag_class_weight = bool(args.class_weight)          # A1
    CONFIG.tag_label_smoothing = args.label_smoothing or 0.0   # A5
    CONFIG.layerwise_lr = args.layerwise_lr                    # A3 (None = off)
    CONFIG.pool_mode = args.pool_mode or "mean"                # O5
    CONFIG.aux_asc_head = bool(args.aux_asc_head)              # A7
    if args.lambda_asc is not None:
        CONFIG.lambda_asc = args.lambda_asc
    CONFIG.use_crf = bool(args.crf)                            # A4
    CONFIG.fusion_conf_append = bool(args.conf_append)         # A9
    CONFIG.fusion_feat_gate = bool(args.feat_gate)             # A10
    CONFIG.fusion_reliability = bool(args.reliability)         # A11
    if args.text_model:
        CONFIG.text_model_id = args.text_model                 # A8
    if args.batch_size:
        CONFIG.batch_size = args.batch_size
    if args.grad_accum:
        CONFIG.grad_accum = args.grad_accum

    from train import train, make_loader          # noqa: E402  (import after config mutation)
    from evaluate import evaluate_all             # noqa: E402

    res = train(CONFIG, dataset=args.dataset, max_epochs=args.epochs)
    model = res["model"]
    test = make_loader(args.dataset, "test", CONFIG, shuffle=False)
    m = evaluate_all(model, test, CONFIG.device)
    print(args.dataset, m)

    row = {
        "tag": args.tag, "dataset": args.dataset, "best_dev_F1": round(res["best_dev_f1"], 2),
        "joint_P": round(m["joint"]["P"], 2), "joint_R": round(m["joint"]["R"], 2),
        "joint_F1": round(m["joint"]["F1"], 2), "mate_F1": round(m["mate"]["F1"], 2),
        "masc_Acc": round(m["masc"]["Acc"], 2), "masc_F1": round(m["masc"]["F1"], 2),
        "evidence_dropout": CONFIG.evidence_dropout, "kan_hidden": str(CONFIG.kan_hidden),
        "patience": CONFIG.early_stop_patience, "pool_mode": CONFIG.pool_mode,
        "class_weight": CONFIG.tag_class_weight, "label_smoothing": CONFIG.tag_label_smoothing,
        "layerwise_lr": CONFIG.layerwise_lr, "aux_asc_head": CONFIG.aux_asc_head,
        "use_crf": CONFIG.use_crf, "text_model": CONFIG.text_model_id,
        "batch_size": CONFIG.batch_size, "grad_accum": CONFIG.grad_accum,
        "conf_append": CONFIG.fusion_conf_append, "feat_gate": CONFIG.fusion_feat_gate,
    }
    out = ROOT / "results" / "tables" / "iterations.csv"
    exists = out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"logged -> {out}")


if __name__ == "__main__":
    main()
