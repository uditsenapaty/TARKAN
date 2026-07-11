"""Non-VL probe: student(DeBERTa+CLIP) x AoM(BART). Same combination rules as qwen_probe.
Establishes the non-VL ceiling on t2015. CPU-only, uses existing dumps."""
import json, sys
from pathlib import Path
GRAFT = Path("/teamspace/studios/this_studio/graft")
sys.path.insert(0, str(GRAFT))
from ns_offline import calibrate_offset
from ensemble_stack import load_aom_member, gold_word_spans, prf
from transformers import BartTokenizer
tok = BartTokenizer.from_pretrained(str(GRAFT / "bart-base"))
ddir = GRAFT / "AoM_full/src/data/twitter2015"
data = {s: json.load(open(ddir / f"{s}.json")) for s in ("dev", "test")}
golds = {s: gold_word_spans(data[s]) for s in ("dev", "test")}
stu = {s: [[tuple(x) for x in inst] for inst in json.load(open(GRAFT / f"student_t15_{s}.json"))] for s in ("dev","test")}
aom = {}
for s in ("dev","test"):
    samples=[json.loads(l) for l in open(GRAFT / f"dump_t15_{s}.jsonl")]
    aom[s]=load_aom_member(GRAFT / f"dump_t15_{s}.jsonl", data[s], tok, calibrate_offset(samples, data[s], tok))
def combine(qs, as_, rule):
    out=[]
    for q,a in zip(qs,as_):
        qm={(w0,w1):p for w0,w1,p in q}; am={(w0,w1):p for w0,w1,p in a}
        qsS,asS=set(qm),set(am); res=[]
        if rule=="student": res=list(q)
        elif rule=="aom": res=list(a)
        elif rule=="inter_spol": res=[(k[0],k[1],qm[k]) for k in qsS&asS]
        elif rule=="inter_apol": res=[(k[0],k[1],am[k]) for k in qsS&asS]
        elif rule=="inter_agree": res=[(k[0],k[1],qm[k]) for k in qsS&asS if qm[k]==am[k]]
        elif rule=="union_spol": res=[(k[0],k[1],qm.get(k,am.get(k))) for k in qsS|asS]
        elif rule=="aomspan_spol": res=[(k[0],k[1],qm.get(k,am[k])) for k in asS]
        out.append(res)
    return out
rules=["student","aom","inter_spol","inter_apol","inter_agree","union_spol","aomspan_spol"]
print(f"{'rule':14} {'devF':>6} | {'testP':>6} {'testR':>6} {'testF':>6}")
best=None
for r in rules:
    _,_,df=prf(combine(stu["dev"],aom["dev"],r),golds["dev"])
    tp,tr,tf=prf(combine(stu["test"],aom["test"],r),golds["test"])
    if r not in("student","aom") and (best is None or df>best[1]): best=(r,df,tf)
    print(f"{r:14} {df:6.2f} | {tp:6.2f} {tr:6.2f} {tf:6.2f}")
print(f"\nBEST-by-dev non-VL combo: {best[0]} dev {best[1]} -> TEST F {best[2]} (bar 72.5)")
