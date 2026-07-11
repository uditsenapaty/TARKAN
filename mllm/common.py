"""Shared utils for the Qwen2.5-VL MABSA fine-tune (patch A15).

Data source: AoM's canonical JSON splits (words / image_id / aspects[from,to,polarity,term]),
identical splits the paper (Table 2) and every Table-1 baseline are scored on.

Scoring matches the baselines exactly: (aspect-span, polarity) exact-match micro-F1 (joint/AESC).
The MLLM emits text, so predicted aspect surfaces are aligned back to token spans before counting,
giving numbers directly comparable to the 72.5 / 71.4 bars. MATE = span-only micro-F1;
MASC = polarity acc + macro-F1 over gold aspects the model also extracted.
"""
import json, os, re

AOM_ROOT = "/teamspace/studios/this_studio/graft/AoM_full/src/data"
IMG_ROOT = "/teamspace/studios/this_studio/graft/vlp_assets/VLP-MABSA/IJCAI2019_data"
IMG_DIR = {"twitter2015": f"{IMG_ROOT}/twitter2015_images",
           "twitter2017": f"{IMG_ROOT}/twitter2017_images"}

POL2WORD = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}
WORD2POL = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}

INSTRUCTION = (
    "You are an aspect-based sentiment analysis system for tweets. Read the tweet text and its "
    "image, then extract the aspect terms and their sentiment. An aspect term is a specific named "
    "entity the tweet actually comments on or expresses an opinion about — a person, organization, "
    "location, product, event, or group. Extract the exact span as it appears in the text. Do NOT "
    "extract generic words, hashtags, or @-usernames unless that item is itself the opinion target, "
    "and do NOT list entities that are only mentioned in passing. Give each aspect one sentiment: "
    "positive, negative, or neutral. Use neutral for factual or news-style mentions that carry no "
    "clear positive or negative stance — this is the most common case. Use the image only to help "
    "judge the sentiment of a text aspect, never to introduce aspects that are not in the text. "
    "Reply with ONLY a JSON array of objects like "
    '[{"aspect": "<exact text span>", "sentiment": "positive|negative|neutral"}]. '
    "If there are no aspects, reply with []."
)


def load_split(dataset, split):
    recs = json.load(open(f"{AOM_ROOT}/{dataset}/{split}.json"))
    out = []
    for r in recs:
        words = r["words"]
        img = os.path.join(IMG_DIR[dataset], r["image_id"])
        gold = [[a["from"], a["to"], a["polarity"]] for a in r["aspects"]]
        out.append({"words": words, "text": " ".join(words), "image": img, "gold": gold})
    return out


def target_json(words, gold):
    """Canonical assistant target: JSON array of {aspect surface, sentiment word}."""
    items = [{"aspect": " ".join(words[f:t]), "sentiment": POL2WORD[p]} for f, t, p in gold]
    return json.dumps(items, ensure_ascii=False)


# ---- parsing generated text back to (aspect_str, POL) ----
def parse_pairs(text):
    pairs = []
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            for it in arr:
                if isinstance(it, dict):
                    a = str(it.get("aspect", "")).strip()
                    s = str(it.get("sentiment", "")).strip().lower()
                    if a and s in WORD2POL:
                        pairs.append((a, WORD2POL[s]))
            return pairs
        except Exception:
            pass
    # fallback: line-based "<aspect> : <sentiment>"
    for line in text.splitlines():
        mm = re.match(r"\s*[-*]?\s*(.+?)\s*[:\-|]\s*(positive|negative|neutral)\s*$", line, re.I)
        if mm:
            pairs.append((mm.group(1).strip(), WORD2POL[mm.group(2).lower()]))
    return pairs


def _norm_tok(t):
    return t.lower()


def align_to_span(aspect_str, words, used):
    """Greedy first-unused token-span match of an emitted aspect surface. -1,-1 if unalignable."""
    at = [_norm_tok(t) for t in aspect_str.split()]
    if not at:
        return (-1, -1)
    wl = [_norm_tok(w) for w in words]
    n = len(at)
    for i in range(len(wl) - n + 1):
        if i in used:
            continue
        if wl[i:i + n] == at:
            for j in range(i, i + n):
                used.add(j)
            return (i, i + n)
    # relaxed: substring join match (handles punctuation splits)
    joined = "".join(at)
    for i in range(len(wl)):
        for k in range(i + 1, min(len(wl), i + 8) + 1):
            if i in used:
                break
            if "".join(wl[i:k]) == joined:
                for j in range(i, k):
                    used.add(j)
                return (i, k)
    return (-1, -1)


def pred_triples(pairs, words):
    used = set()
    tri = []
    for a, p in pairs:
        f, t = align_to_span(a, words, used)
        tri.append((f, t, p))
    return tri


def prf(pred_sets, gold_sets):
    tp = fp = fn = 0
    for pr, gd in zip(pred_sets, gold_sets):
        pr, gd = set(pr), set(gd)
        tp += len(pr & gd)
        fp += len(pr - gd)
        fn += len(gd - pr)
    P = tp / (tp + fp + 1e-13)
    R = tp / (tp + fn + 1e-13)
    F = 2 * P * R / (P + R + 1e-13)
    return P * 100, R * 100, F * 100


def score(records, gen_pairs):
    """records: load_split output; gen_pairs: list[list[(aspect_str,POL)]] aligned by index.
       Returns dict with joint P/R/F1, MATE P/R/F1, MASC acc/macF1."""
    joint_pred, joint_gold, mate_pred, mate_gold = [], [], [], []
    masc_correct = masc_total = 0
    from collections import Counter
    cm = Counter()  # (gold_pol, pred_pol) on matched aspects
    for rec, pairs in zip(records, gen_pairs):
        words, gold = rec["words"], rec["gold"]
        tri = pred_triples(pairs, words)
        gset = {(f, t, p) for f, t, p in gold}
        pset = {(f, t, p) for f, t, p in tri if f >= 0}
        joint_pred.append(pset); joint_gold.append(gset)
        mate_pred.append({(f, t) for f, t, _ in pset})
        mate_gold.append({(f, t) for f, t, _ in gold})
        gold_span2pol = {(f, t): p for f, t, p in gold}
        pred_span2pol = {(f, t): p for f, t, p in tri if f >= 0}
        for span, gp in gold_span2pol.items():
            if span in pred_span2pol:
                masc_total += 1
                pp = pred_span2pol[span]
                cm[(gp, pp)] += 1
                if pp == gp:
                    masc_correct += 1
    jP, jR, jF = prf(joint_pred, joint_gold)
    mP, mR, mF = prf(mate_pred, mate_gold)
    acc = 100 * masc_correct / (masc_total + 1e-13)
    # MASC macro-F1 over POS/NEU/NEG on matched aspects
    labs = ["POS", "NEU", "NEG"]
    f1s = []
    for L in labs:
        tp = cm[(L, L)]
        fp = sum(cm[(g, L)] for g in labs if g != L)
        fn = sum(cm[(L, pp)] for pp in labs if pp != L)
        p = tp / (tp + fp + 1e-13); r = tp / (tp + fn + 1e-13)
        f1s.append(2 * p * r / (p + r + 1e-13))
    macF1 = 100 * sum(f1s) / 3
    return {"joint_P": jP, "joint_R": jR, "joint_F": jF,
            "mate_P": mP, "mate_R": mR, "mate_F": mF,
            "masc_acc": acc, "masc_macF1": macF1, "masc_n": masc_total}
