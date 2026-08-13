"""Offline neurosymbolic layer for AoM dumps (runs in the MAIN env, no GPU).

Consumes dump_t15_{dev,test}.jsonl ({'idx', 'pred': [[s,e,pol],...], 'gold': [...]})
plus the dataset's {dev,test}.json (words + aspects with from/to word indices).

Pipeline:
  1. pointer->word mapping: re-tokenize words word-by-word with BART BPE
     (add_prefix_space=True) exactly like their ConditionTokenizer, then
     SELF-CALIBRATE the constant pointer offset per split from gold pairs.
  2. rules (A13/A14 ported): SenticNet windowed polarity prior with negation flip
     -> strong-override of predicted polarity; duplicate-aspect consistency.
  3. metric: replicates AESCSpanMetric counting; VALIDATED by reproducing the
     run's own printed baseline F1 on the unmodified dump.
  4. grid-tune (tau, window, mode, consistency) on DEV, apply best to TEST.

Polarity ids in dumps: 3=POS, 4=NEU, 5=NEG (their label mapping).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

GRAFT = Path(__file__).resolve().parent
NEGATORS = {"not", "no", "never", "n't", "nt", "hardly", "barely", "cannot", "cant",
            "dont", "don't", "didnt", "didn't", "isnt", "isn't", "wasnt", "wasn't", "aint", "ain't"}
POS_ID, NEU_ID, NEG_ID = 3, 4, 5


def load_lexicon():
    lex = {}
    for line in open(GRAFT / "senticnet_polarity.tsv"):
        try:
            w, v = line.rstrip("\n").split("\t")
            lex[w] = float(v)
        except ValueError:
            continue
    return lex


def bpe_starts(words, tok):
    """cumulative BPE start position per word (their word-by-word scheme)."""
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(tok.tokenize(w, add_prefix_space=True) or [w])
    starts.append(pos)  # sentinel = total bpe len
    return starts


def calibrate_offset(samples, data, tok):
    """Solve pointer_offset s.t. gold_pair_start == bpe_start[aspect.from] + offset."""
    votes = Counter()
    for s in samples[:200]:
        raw = data[s["idx"]]
        starts = bpe_starts(raw["words"], tok)
        aspects = raw.get("aspects", [])
        if len(aspects) != len(s["gold"]):
            continue
        for g, a in zip(s["gold"], aspects):
            votes[g[0] - starts[a["from"]]] += 1
    off, n = votes.most_common(1)[0]
    total = sum(votes.values())
    print(f"  offset={off} (consistency {n}/{total} = {100*n/total:.1f}%)")
    return off


def pair_to_wordspan(pair, starts, offset):
    """map (bpe_s, bpe_e, pol) -> (word_s, word_e_incl) or None."""
    bs, be = pair[0] - offset, pair[1] - offset
    ws = we = None
    for wi in range(len(starts) - 1):
        if starts[wi] == bs:
            ws = wi
        if starts[wi] <= be < starts[wi + 1]:
            we = wi
    if ws is None or we is None or we < ws:
        return None
    return ws, we


def lex_prior(words, ws, we, lex, window):
    lo, hi = max(0, ws - window), min(len(words), we + 1 + window)
    toks = [w.lower() for w in words]
    vals = []
    for i in range(lo, hi):
        if ws <= i <= we:
            continue
        w = toks[i].strip("#@.,!?:;'\"()[]")
        p = lex.get(w) or lex.get(w.replace("-", "_"))
        if p is not None and abs(p) > 0.1:
            neg = any(toks[j] in NEGATORS for j in range(max(lo, i - 3), i))
            vals.append(-p if neg else p)
    return sum(vals) / len(vals) if vals else 0.0


def apply_rules(samples, data, tok, offset, lex, tau, window, mode, consistency):
    """mode: 'any' = override whenever strong prior disagrees; 'neu' = only NEU preds."""
    out = []
    for s in samples:
        raw = data[s["idx"]]
        words = raw["words"]
        starts = bpe_starts(words, tok)
        new_pairs = []
        for p in s["pred"]:
            p = list(p)
            span = pair_to_wordspan(p, starts, offset)
            if span is not None and tau < 10:
                m = lex_prior(words, span[0], span[1], lex, window)
                if abs(m) >= tau:
                    lex_pol = POS_ID if m > 0 else NEG_ID
                    if (mode == "any" and lex_pol != p[2]) or (mode == "neu" and p[2] == NEU_ID):
                        p[2] = lex_pol
            new_pairs.append(p)
        if consistency and len(new_pairs) > 1:
            by_text = {}
            for i, p in enumerate(new_pairs):
                sp = pair_to_wordspan(p, starts, offset)
                key = " ".join(words[sp[0]:sp[1] + 1]).lower() if sp else f"_{i}"
                by_text.setdefault(key, []).append(i)
            for key, idxs in by_text.items():
                if len(idxs) > 1:
                    maj = Counter(new_pairs[i][2] for i in idxs).most_common(1)[0][0]
                    for i in idxs:
                        new_pairs[i][2] = maj
        out.append({"idx": s["idx"], "pred": new_pairs, "gold": s["gold"]})
    return out


def aesc_prf(samples):
    """Replicates AESCSpanMetric counting (counter semantics, skip pol in (0,1,-1))."""
    tp = fn = fp = 0
    for s in samples:
        tgt, prd = {}, {}
        for t in s["gold"]:
            tgt[(t[0], t[1])] = t[2]
        for p in s["pred"]:
            if p[2] not in (0, 1, -1):
                prd[(p[0], p[1])] = p[2]
        t_list = [(k[0], k[1], v) for k, v in tgt.items()]
        p_list = [(k[0], k[1], v) for k, v in prd.items()]
        t_c, p_c = Counter(t_list), Counter(p_list)
        _tp = sum((t_c & p_c).values())
        tp += _tp
        fn += sum(t_c.values()) - _tp
        fp += sum(p_c.values()) - _tp
    pre = tp / (tp + fp + 1e-13)
    rec = tp / (tp + fn + 1e-13)
    f = 2 * pre * rec / (pre + rec + 1e-13)
    return round(100 * pre, 2), round(100 * rec, 2), round(100 * f, 2)


def main():
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else str(GRAFT / "AoM_full/src/data/twitter2015")
    tag = sys.argv[2] if len(sys.argv) > 2 else "t15"
    from transformers import BartTokenizer
    tok = BartTokenizer.from_pretrained(str(GRAFT / "bart-base"))
    lex = load_lexicon()

    def load(split):
        samples = [json.loads(l) for l in open(GRAFT / f"dump_{tag}_{split}.jsonl")]
        data = json.load(open(Path(dataset_dir) / f"{split}.json"))
        return samples, data

    dev, dev_data = load("dev")
    test, test_data = load("test")
    print("baseline (must match the run's own printed numbers):")
    print("  dev :", aesc_prf(dev))
    print("  test:", aesc_prf(test))

    print("calibrating pointer offset:")
    off = calibrate_offset(dev, dev_data, tok)

    grid = []
    for tau in (0.4, 0.5, 0.6, 0.7, 99):     # 99 = prior off (consistency-only rows)
        for window in ((4, 6) if tau < 10 else (4,)):
            for mode in (("neu", "any") if tau < 10 else ("neu",)):
                for cons in (False, True):
                    adj = apply_rules(dev, dev_data, tok, off, lex, tau, window, mode, cons)
                    p, r, f = aesc_prf(adj)
                    grid.append(((tau, window, mode, cons), f))
                    print(f"  dev tau={tau} w={window} mode={mode} cons={int(cons)} -> F {f}")
    (tau, window, mode, cons), best_dev = max(grid, key=lambda x: x[1])
    print(f"BEST dev: tau={tau} w={window} mode={mode} cons={cons} (dev F {best_dev})")
    adj_test = apply_rules(test, test_data, tok, off, lex, tau, window, mode, cons)
    print("TEST with best rules:", aesc_prf(adj_test))


if __name__ == "__main__":
    main()
