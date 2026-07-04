"""Heterogeneous span-level ensemble (stage-1 'minimal solution').

Members: any mix of
  - AoM-family dumps (dump_*.jsonl with BPE-pointer pairs) -> projected to word spans
    via the calibrated offset machinery (ns_offline)
  - student-family dumps (JSON: list per sample of [ws, we_incl, POL_STR] word spans)

Aggregation: weighted span voting with a dev-tuned acceptance threshold.
  score(span) = sum of member_weight over members proposing (ws, we);
  accept if score >= theta; polarity = weight-majority among proposers.
Weights default to each member's dev F1 (softmax-tempered). Grid over theta and
temperature on DEV; report TEST at the best cells-cleared/F1 operating point.

All member predictions must be aligned to the SAME sample order (AoM json order).
Student dumps are keyed by image_id+sentence match -> aligned externally.

Usage:
  python3 graft/ensemble_stack.py --config graft/ensemble_t15.json
Config: {"dataset_dir": ".../twitter2015", "tag": "t15",
         "members": [{"name": "official", "kind": "aom", "dev": "...", "test": "..."},
                     {"name": "student", "kind": "word", "dev": "...", "test": "..."}]}
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

GRAFT = Path(__file__).resolve().parent
POL_FROM_ID = {3: "POS", 4: "NEU", 5: "NEG"}


def load_aom_member(dump_path, data, tok, offset):
    """AoM dump -> per-sample list of (ws, we_incl, POL)."""
    import sys
    sys.path.insert(0, str(GRAFT))
    from ns_offline import bpe_starts, pair_to_wordspan

    samples = [json.loads(l) for l in open(dump_path)]
    out = [[] for _ in range(len(data))]
    for s in samples:
        raw = data[s["idx"]]
        starts = bpe_starts(raw["words"], tok)
        spans = []
        for p in s["pred"]:
            if p[2] not in (3, 4, 5):
                continue
            w = pair_to_wordspan(p, starts, offset)
            if w is not None:
                spans.append((w[0], w[1], POL_FROM_ID[p[2]]))
        out[s["idx"]] = spans
    return out


def gold_word_spans(data):
    return [[(a["from"], a["to"] - 1, a["polarity"]) for a in rec.get("aspects", [])] for rec in data]


def prf(preds, golds):
    tp = fp = fn = 0
    for p_list, g_list in zip(preds, golds):
        pc, gc = Counter(p_list), Counter(g_list)
        inter = sum((pc & gc).values())
        tp += inter
        fp += sum(pc.values()) - inter
        fn += sum(gc.values()) - inter
    pre = tp / (tp + fp + 1e-13)
    rec = tp / (tp + fn + 1e-13)
    f = 2 * pre * rec / (pre + rec + 1e-13)
    return round(100 * pre, 2), round(100 * rec, 2), round(100 * f, 2)


def vote(member_preds, weights, theta):
    n = len(member_preds[0])
    out = []
    for i in range(n):
        score = defaultdict(float)
        pol_w = defaultdict(lambda: defaultdict(float))
        for m, preds in enumerate(member_preds):
            for (ws, we, pol) in preds[i]:
                score[(ws, we)] += weights[m]
                pol_w[(ws, we)][pol] += weights[m]
        spans = []
        for k, sc in score.items():
            if sc >= theta:
                pol = max(pol_w[k].items(), key=lambda x: x[1])[0]
                spans.append((k[0], k[1], pol))
        out.append(sorted(spans))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    ddir = Path(cfg["dataset_dir"])

    from transformers import BartTokenizer
    tok = BartTokenizer.from_pretrained(str(GRAFT / "bart-base"))

    data = {s: json.load(open(ddir / f"{s}.json")) for s in ("dev", "test")}
    golds = {s: gold_word_spans(data[s]) for s in ("dev", "test")}

    # offset calibration once per split from any AoM dump
    import sys
    sys.path.insert(0, str(GRAFT))
    from ns_offline import calibrate_offset
    first_aom = next(m for m in cfg["members"] if m["kind"] == "aom")
    offs = {}
    for s in ("dev", "test"):
        samples = [json.loads(l) for l in open(first_aom[s])]
        offs[s] = calibrate_offset(samples, data[s], tok)

    members = {}
    dev_f1 = []
    for m in cfg["members"]:
        preds = {}
        for s in ("dev", "test"):
            if m["kind"] == "aom":
                preds[s] = load_aom_member(m[s], data[s], tok, offs[s])
            else:  # word-span json: [[ [ws,we,POL], ...] per sample]
                preds[s] = [[tuple(x) for x in inst] for inst in json.load(open(m[s]))]
        members[m["name"]] = preds
        p, r, f = prf(preds["dev"], golds["dev"])
        tp, tr, tf = prf(preds["test"], golds["test"])
        dev_f1.append(f)
        print(f"member {m['name']:12} dev F {f:6.2f} | test P {tp:.2f} R {tr:.2f} F {tf:.2f}")

    names = [m["name"] for m in cfg["members"]]
    mp = {s: [members[n][s] for n in names] for s in ("dev", "test")}

    best = None
    for temp in (0.0, 0.5, 1.0, 2.0):  # 0 = uniform weights
        if temp == 0.0:
            w = [1.0] * len(names)
        else:
            mx = max(dev_f1)
            w = [pow(2.718, (f - mx) / temp) for f in dev_f1]
        for theta_frac in (0.34, 0.4, 0.5, 0.6, 0.67):
            theta = theta_frac * sum(w)
            p, r, f = prf(vote(mp["dev"], w, theta), golds["dev"])
            if best is None or f > best[0]:
                best = (f, temp, theta_frac, w)
            print(f"  temp={temp} theta={theta_frac:.2f} -> dev P {p:.2f} R {r:.2f} F {f:.2f}")
    f, temp, tf_, w = best
    print(f"\nBEST dev: temp={temp} theta_frac={tf_} (dev F {f})")
    theta = tf_ * sum(w)
    p, r, fb = prf(vote(mp["test"], w, theta), golds["test"])
    print(f"ENSEMBLE TEST: P {p} R {r} F {fb}")


if __name__ == "__main__":
    main()
