"""Inference + evaluation (updated paper §3.8).

Joint MABSA uses a single unified BIO head (Eq. 21) run on the KAN-fused multimodal
token representation. Inference is two-stage (§3.8):
  Stage 1: predict BIO tags with no aspect evidence -> extract aspect spans.
  Stage 2: for each predicted span, compute relevance-filtered visual + filtered KG,
           re-fuse per token via KAN, re-run the BIO head -> final aspect-level polarity.
There is no separate ASC head; the MASC subtask reads the BIO polarity on gold aspects.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from config import CONFIG, ID2TAG, ID2POL
from metrics import joint_prf, mate_prf, masc_acc_f1
from utils import bio_to_spans


def _to_device(batch: Dict, device: str) -> Dict:
    out = dict(batch)
    for k in ("input_ids", "attention_mask", "bio_labels", "pixel_values"):
        if k in batch and torch.is_tensor(batch[k]):
            out[k] = batch[k].to(device)
    return out


def decode_word_tags(tag_logits_b: torch.Tensor, word_ids: List[int], n_words: int) -> List[str]:
    """Map subtoken tag predictions -> word-level BIO (tag at each word's first subtoken)."""
    sub = tag_logits_b.argmax(-1).tolist()
    word_tag = ["O"] * n_words
    seen = set()
    for i, wid in enumerate(word_ids):
        if wid == -1 or wid >= n_words or wid in seen:
            continue
        seen.add(wid)
        word_tag[wid] = ID2TAG[sub[i]]
    return word_tag


def decode_anchor_spans(anchor_logits_b, word_ids, n_words):
    """Paper §3.3/§3.10 — candidate anchors from the preliminary aspect-only head.

    Contiguous tokens predicted as aspect-bearing are grouped into candidates. These
    condition relevance / KG retrieval / fusion; they are NOT the final output, which the
    unified BIO head produces after evidence fusion.
    """
    from config import ANCHOR_TAGS
    sub = anchor_logits_b.argmax(-1).tolist()
    wt = ["O"] * n_words
    seen = set()
    for i, wid in enumerate(word_ids):
        if wid == -1 or wid >= n_words or wid in seen:
            continue
        seen.add(wid)
        wt[wid] = ANCHOR_TAGS[sub[i]]
    spans, start = [], None
    for i, t in enumerate(wt):
        if t == "B-ASP":
            if start is not None:
                spans.append((start, i))
            start = i
        elif t == "I-ASP":
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(wt)))
    return spans


def _majority_pol(word_tags: List[str], s: int, e: int) -> str:
    """Majority sentiment suffix over word tags in [s, e) (NEU if none)."""
    pols = [word_tags[i].split("-", 1)[1] for i in range(s, e) if 0 <= i < len(word_tags) and word_tags[i] != "O"]
    return max(set(pols), key=pols.count) if pols else "NEU"


def decode_spans(word_tags: List[str]) -> List[Tuple[int, int, str]]:
    """Boundary-first span decode (updated §3.8).

    Segments aspect spans on B/O boundaries while IGNORING the polarity suffix, then
    assigns each span its majority polarity. This avoids fragmenting a multi-word aspect
    when the BIO head flips polarity mid-span (which otherwise costs MATE precision).
    Gold spans are single-polarity by construction, so this only affects decoding.
    """
    collapsed = ["O" if t == "O" else (t.split("-", 1)[0] + "-X") for t in word_tags]
    return [(s, e, _majority_pol(word_tags, s, e)) for (s, e, _) in bio_to_spans(collapsed)]


def _pred_subspans_for(word_ids, span):
    """Map a predicted WORD span (s,e_excl) -> subtoken index range (mirrors data._subtoken_spans)."""
    s, e, *_ = span
    idxs = [i for i, wid in enumerate(word_ids) if isinstance(wid, int) and wid >= 0 and s <= wid < e]
    return (idxs[0], idxs[-1] + 1) if idxs else (0, 1)


@torch.no_grad()
def predict_joint(model, loader, device: str = None) -> Tuple[List, List]:
    """Joint MABSA (span, polarity) per instance via the two-stage unified-BIO pipeline (§3.8).

    Stage 1 extracts spans from a BIO pass with no aspect evidence; Stage 2 re-runs the
    KAN-enhanced BIO head over the predicted spans to assign the final polarity, so visual
    and KG evidence inform the joint metric.
    """
    device = device or CONFIG.device
    model.eval()
    from data import opinion_words, visual_concepts  # lazy: rebuild KG queries for predicted spans
    from kg_retrieval import AspectQuery

    ds = getattr(loader, "dataset", None)
    id2inst = {inst.id: inst for inst in (getattr(ds, "instances", None) or [])}
    captions = getattr(ds, "captions", None) or {}

    preds, golds = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        _amp = (str(device).startswith("cuda") and bool(getattr(model.cfg, "use_amp", True)))
        with torch.autocast("cuda", dtype=torch.float16, enabled=_amp):
            text_feats = model.text_encoder(batch["input_ids"], batch["attention_mask"])
            visual_feats = None
            if model.cfg.use_visual_stream and model.visual_encoder is not None and "pixel_values" in batch:
                visual_feats = model.visual_encoder(batch["pixel_values"])
        text_feats = text_feats.float()
        if visual_feats is not None:
            visual_feats = visual_feats.float()
        B = text_feats.size(0)

        # ---- ONE forward pass, ONE unified prediction (paper §3.3 + §3.8, §3.10) ----
        # anchors condition the evidence; the unified 7-tag CRF is the FINAL head and
        # emits span AND polarity together. Both ASC heads are auxiliary training signals
        # only -- there is no second inference stage and no separate polarity model.
        anc = model.anchor_head(text_feats)            # [B, n, 3] over O / B-ASP / I-ASP
        batch_spans = []
        for b in range(B):
            batch_spans.append(decode_anchor_spans(
                anc[b], batch["word_ids"][b], batch["n_words"][b]))
            golds.append([(s, e, pol) for (s, e, pol) in batch["gold_aspects"][b]])

        sub_spans, queries = [], []
        for b in range(B):
            inst = id2inst.get(batch["instance_id"][b])
            tokens = inst.tokens if inst is not None else None
            vc = visual_concepts(captions.get(inst.image_id, "")) if inst is not None else []
            subs, qs = [], []
            for sp in batch_spans[b]:
                subs.append(_pred_subspans_for(batch["word_ids"][b], sp))
                s_, e_ = sp[0], sp[1]
                qs.append(AspectQuery(aspect_term=" ".join(tokens[s_:e_]),
                                      opinion_words=opinion_words(tokens, (s_, e_)),
                                      visual_concepts=vc)
                          if tokens is not None else AspectQuery(aspect_term=""))
            sub_spans.append(subs)
            queries.append(qs)

        fwd = dict(batch)
        fwd["aspect_spans"] = sub_spans
        fwd["aspect_queries"] = queries
        fwd.pop("aspect_triples", None)
        out = model(fwd, text_feats=text_feats, visual_feats=visual_feats)

        # unified 7-tag decode: Viterbi when the CRF is on, else per-token argmax
        crf_paths = None
        crf_paths = None

        use_asc = (getattr(model.cfg, "polarity_source", "crf") == "asc"
                   and out.get("asc_logits") is not None and out["asc_logits"].numel() > 0)
        ascpol = out["asc_logits"].argmax(-1).tolist() if use_asc else None
        idx = 0
        for b in range(B):
            if crf_paths is not None:
                wt = [ID2TAG[t] for t in crf_paths[b]]
            else:
                wt = decode_word_tags(out["tag_logits"][b], batch["word_ids"][b],
                                      batch["n_words"][b])
            spans = decode_spans(wt)          # span + polarity from the ONE unified head
            if use_asc:
                # optional: take polarity from the auxiliary rich head instead. Only valid
                # where a final span coincides with an anchor, since the ASC rows are in
                # anchor order; elsewhere the unified tag stands.
                amap = {(a_, b_): ascpol[idx + j]
                        for j, (a_, b_) in enumerate(batch_spans[b])}
                idx += len(batch_spans[b])
                spans = [(s_, e_, ID2POL[amap[(s_, e_)]] if (s_, e_) in amap else p_)
                         for (s_, e_, p_) in spans]
            elif ascpol is not None:
                idx += len(batch_spans[b])
            
            preds.append(spans)
    return preds, golds


@torch.no_grad()
def predict_masc(model, loader, device: str = None) -> Tuple[List[str], List[str]]:
    """Polarity on GOLD aspects, read from the unified BIO head over gold spans (Table 3 MASC)."""
    device = device or CONFIG.device
    model.eval()
    ds = getattr(loader, "dataset", None)
    id2inst = {inst.id: inst for inst in (getattr(ds, "instances", None) or [])}
    y_true, y_pred = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)  # gold aspect_spans
        use_asc = getattr(model.cfg, "aux_asc_head", False) and out.get("asc_logits") is not None \
            and out["asc_logits"].numel() > 0
        if use_asc:
            logits = out["asc_logits"]
            
            asc = logits.argmax(-1).tolist(); idx = 0
            for b in range(len(batch["aspect_spans"])):
                for k in range(len(batch["aspect_spans"][b])):
                    y_pred.append(ID2POL[asc[idx]]); idx += 1
                    y_true.append(ID2POL[batch["aspect_polarity"][b][k]])
        else:
            tag = out["tag_logits"]
            for b in range(len(batch["aspect_spans"])):
                for k, sp in enumerate(batch["aspect_spans"][b]):
                    s, e = sp[0], sp[1]
                    sub_ids = tag[b, s:e].argmax(-1).tolist()
                    pols = [ID2TAG[t].split("-", 1)[1] for t in sub_ids if ID2TAG[t] != "O"]
                    y_pred.append(max(set(pols), key=pols.count) if pols else "NEU")
                    y_true.append(ID2POL[batch["aspect_polarity"][b][k]])
    return y_true, y_pred


def evaluate_all(model, loader, device: str = None) -> Dict[str, Dict]:
    preds, golds = predict_joint(model, loader, device)
    yt, yp = predict_masc(model, loader, device)
    return {
        "joint": joint_prf(preds, golds),
        "mate": mate_prf(preds, golds),
        "masc": masc_acc_f1(yt, yp),
    }


def _build_kg_and_entities(cfg):
    """Mirror train.py's model construction so eval uses the same KG stream that training did."""
    kg = None
    sqlite = cfg.paths.kg_index / "kg.sqlite"
    if sqlite.exists():
        from kg import KnowledgeGraph

        kg = KnowledgeGraph(sqlite_path=str(sqlite))
    ent = None
    nb = cfg.paths.conceptnet / "numberbatch-en.txt"
    # only the fallback TripleEncoder needs Numberbatch; the paper's verbalized encoder
    # (§3.6) does not, and loading 1.1 GB costs minutes per run
    if nb.exists() and getattr(cfg, "kg_encoder", "verbalized") != "verbalized":
        from kg_retrieval import EntityEmbedder

        ent = EntityEmbedder.from_txt(str(nb))
    return kg, ent


if __name__ == "__main__":
    import argparse

    from data import TarkanDataset, collate_fn, load_split
    from models import TarkanStudent
    from utils import load_checkpoint, get_logger

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    log = get_logger("evaluate")

    from torch.utils.data import DataLoader

    data_dir = CONFIG.paths.data / args.dataset
    images = CONFIG.paths.data / "images" / args.dataset
    insts = load_split(data_dir, args.split)
    ds = TarkanDataset(insts, CONFIG, images_dir=images)
    loader = DataLoader(ds, batch_size=CONFIG.batch_size, collate_fn=collate_fn)
    kg, ent = _build_kg_and_entities(CONFIG)
    model = TarkanStudent(CONFIG, kg=kg, entity_embedder=ent).to(CONFIG.device)
    load_checkpoint(model, args.checkpoint, map_location=CONFIG.device)
    log.info(evaluate_all(model, loader))
