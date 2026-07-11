"""Direct 2-family probe: Qwen(MLLM) vs AoM(BART). Measure combination rules on dev+test.
Tune rule on DEV, report TEST. No test peeking for selection."""
import json, sys
from pathlib import Path
from collections import Counter
GRAFT = Path("/teamspace/studios/this_studio/graft")
sys.path.insert(0, str(GRAFT))
from ns_offline import calibrate_offset
from ensemble_stack import load_aom_member, gold_word_spans, prf
from transformers import BartTokenizer

tok = BartTokenizer.from_pretrained(str(GRAFT / "bart-base"))
ddir = GRAFT / "AoM_full/src/data/twitter2015"
data = {s: json.load(open(ddir / f"{s}.json")) for s in ("dev", "test")}
golds = {s: gold_word_spans(data[s]) for s in ("dev", "test")}
qwen = {s: [[tuple(x) for x in inst] for inst in json.load(open(GRAFT / f"qwen_t15_{s}.json"))] for s in ("dev", "test")}
aom = {}
for s in ("dev", "test"):
    samples = [json.loads(l) for l in open(GRAFT / f"dump_t15_{s}.jsonl")]
    off = calibrate_offset(samples, data[s], tok)
    aom[s] = load_aom_member(GRAFT / f"dump_t15_{s}.jsonl", data[s], tok, off)


def combine(qs, as_, rule):
    """qs, as_: per-sample lists of (ws,we,pol). Returns per-sample combined list."""
    out = []
    for q, a in zip(qs, as_):
        qmap = {(w0, w1): p for w0, w1, p in q}
        amap = {(w0, w1): p for w0, w1, p in a}
        qs_set, as_set = set(qmap), set(amap)
        res = []
        if rule == "qwen":
            res = list(q)
        elif rule == "aom":
            res = list(a)
        elif rule == "inter_qpol":            # spans both find, Qwen polarity
            for k in qs_set & as_set: res.append((k[0], k[1], qmap[k]))
        elif rule == "inter_apol":            # spans both find, AoM polarity
            for k in qs_set & as_set: res.append((k[0], k[1], amap[k]))
        elif rule == "inter_agree":           # spans both find AND polarity agrees
            for k in qs_set & as_set:
                if qmap[k] == amap[k]: res.append((k[0], k[1], qmap[k]))
        elif rule == "union_qpol":            # either finds; Qwen pol preferred
            for k in qs_set | as_set:
                res.append((k[0], k[1], qmap.get(k, amap.get(k))))
        elif rule == "aomspan_qpol":          # AoM extracts (stage1), Qwen classifies (stage2)
            for k in as_set:
                res.append((k[0], k[1], qmap.get(k, amap[k])))
        elif rule == "qwenspan_gated":        # Qwen spans confirmed by AoM span, Qwen pol
            for k in qs_set:
                if k in as_set: res.append((k[0], k[1], qmap[k]))
        out.append(res)
    return out


rules = ["qwen", "aom", "inter_qpol", "inter_apol", "inter_agree",
         "union_qpol", "aomspan_qpol", "qwenspan_gated"]
print(f"{'rule':16} {'devP':>6} {'devR':>6} {'devF':>6} | {'testP':>6} {'testR':>6} {'testF':>6}")
best = None
for r in rules:
    dp, dr, df = prf(combine(qwen["dev"], aom["dev"], r), golds["dev"])
    tp, tr, tf = prf(combine(qwen["test"], aom["test"], r), golds["test"])
    star = ""
    if r not in ("qwen", "aom") and (best is None or df > best[1]):
        best = (r, df, tf)
    print(f"{r:16} {dp:6.2f} {dr:6.2f} {df:6.2f} | {tp:6.2f} {tr:6.2f} {tf:6.2f}")
print(f"\nBEST-by-dev combo: {best[0]}  dev {best[1]}  -> TEST F {best[2]}  (bar 72.5)")
