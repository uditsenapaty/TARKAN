"""Student pipeline: predicted anchors, qualified evidence, KAN, unified BIO.

The linear ASC head supervises fused aspects only during training. Neither a
teacher nor counterfactual evidence is required for final JMASA inference.
"""
from __future__ import annotations
import torch
from torch import nn
from config import CONFIG, NUM_ANCHOR_TAGS, NUM_POLARITIES
from anchors import decode_anchor_spans, subtoken_span
from heads import BIOTaggingHead
from kan_fusion import build_fusion
from kg_filter import KGFilter
from kg_retrieval import AspectQuery, TripleEncoder, retrieve_triples
from relevance import AspectVisualRelevance, pool_aspect


class TarkanStudent(nn.Module):
    def __init__(self, config=CONFIG, build_encoders=True, kg=None,
                 entity_embedder=None, pool_mode=None):
        super().__init__()
        self.cfg, self.kg = config, kg
        self.pool_mode = pool_mode or config.pool_mode
        d, dp = config.hidden_dim, config.dropout
        self.text_encoder = self.visual_encoder = None
        if build_encoders:
            from encoders import TextEncoder, VisualEncoder
            self.text_encoder = TextEncoder(config.text_model_id, d, dp, frozen=config.freeze_text)
            if config.use_visual_stream:
                self.visual_encoder = VisualEncoder(config.visual_model_id, d, dp, frozen=config.freeze_visual)
        if config.kg_encoder == "verbalized" and self.text_encoder is not None:
            from transformers import AutoTokenizer
            from kg_retrieval import VerbalizedTripleEncoder
            kw = dict(use_fast=False, token=config.hf_token)
            if "bertweet" in config.text_model_id.lower():
                kw["normalization"] = True
            self.triple_encoder = VerbalizedTripleEncoder(
                self.text_encoder, AutoTokenizer.from_pretrained(config.text_model_id, **kw), d, dropout=dp)
        else:
            self.triple_encoder = TripleEncoder(d, embedder=entity_embedder, dropout=dp)
        self.anchor_head = nn.Linear(d, NUM_ANCHOR_TAGS)
        self.relevance = AspectVisualRelevance(d, dropout=dp)
        self.kg_filter = KGFilter(d, dropout=dp, eps=config.kg_eps)
        self.fusion = build_fusion(config.fusion, d, dropout=dp, cfg=config)
        self.fuse_proj = nn.Linear(2*d, d)
        self.bio_head = BIOTaggingHead(d, dropout=dp)
        self.asc_head_paper = nn.Linear(d, NUM_POLARITIES)

    def set_kg(self, kg):
        self.kg = kg

    def _anchors(self, batch, anchor_logits, use_gold_anchors):
        from data import opinion_words
        if use_gold_anchors:
            words = [[(s, e) for s, e, _ in gold
                      if any(s <= wid < e for wid in batch["word_ids"][b])]
                     for b, gold in enumerate(batch["gold_aspects"])]
        else:
            ids = anchor_logits.detach().argmax(-1).tolist()
            words = [decode_anchor_spans(row, wid, n) for row, wid, n in
                     zip(ids, batch["word_ids"], batch["n_words"])]
        spans, queries = [], []
        for b, row in enumerate(words):
            spans.append([subtoken_span(batch["word_ids"][b], sp) for sp in row])
            tokens = batch["tokens"][b]
            concepts = batch.get("visual_concepts", [[] for _ in words])[b]
            queries.append([AspectQuery(
                " ".join(tokens[s:e]),
                opinion_words(tokens, (s, e), window=self.cfg.opinion_window),
                concepts[:self.cfg.top_l_concepts]) for s, e in row])
        return words, spans, queries

    def forward(self, batch, text_feats=None, visual_feats=None, want_alpha=False,
                use_gold_anchors=False):
        cfg = self.cfg
        if text_feats is None:
            text_feats = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        if visual_feats is None and cfg.use_visual_stream:
            visual_feats = self.visual_encoder(batch["pixel_values"])
        B, n, d = text_feats.shape
        anchor_logits = self.anchor_head(text_feats)
        words, spans, queries = self._anchors(batch, anchor_logits, use_gold_anchors)
        triples = []
        aspects, owners = [], []
        for b, row in enumerate(spans):
            pooled = pool_aspect(text_feats[b], row, self.pool_mode)
            for k, _ in enumerate(row):
                aspects.append(pooled[k])
                owners.append(b)
                triples.append(retrieve_triples(queries[b][k], self.kg, cfg.top_m_triples, cfg=cfg)
                               if cfg.use_kg_stream and self.kg is not None else [])
        sizes = [len(row) for row in triples]
        flat = [tr for row in triples for tr in row]
        encoded = self.triple_encoder(flat) if flat else text_feats.new_zeros((0, d))
        gs = list(encoded.split(sizes)) if sizes else []
        scores, rs, zs, alphas = [], [], [], []
        cf_rs, cf_valid, cf_kg = [], [], []
        make_cf = self.training and cfg.use_teacher and cfg.lambda_cf > 0
        z_tok = text_feats.new_zeros((B, n, d))
        image_ids = batch.get("image_id", batch.get("instance_id", list(range(B))))
        for a, (t, b, g) in enumerate(zip(aspects, owners, gs)):
            if cfg.use_visual_stream:
                _, vbar, r, v, alpha = self.relevance(t.unsqueeze(0), visual_feats[b])
                r, v = r[0], v[0]
                if not cfg.use_relevance:
                    r, v = r.new_ones(()), vbar[0]
                alphas.append(alpha)
            else:
                r, v = t.new_zeros(()), t.new_zeros(d)
            if cfg.use_kg_stream and g.numel():
                if cfg.use_kg_filter:
                    s, gv = self.kg_filter(t, g)
                else:
                    s, gv = g.new_zeros(len(g)), g.mean(0)
            else:
                s, gv = t.new_zeros(0), t.new_zeros(d)
            scores.append(s)
            rs.append(r)
            z = self.fusion(t.unsqueeze(0), v.unsqueeze(0), gv.unsqueeze(0))[0]
            zs.append(z if cfg.use_kan_tag_representation else t)
            # Score a different image against the SAME aspect; singleton/duplicate
            # image batches have no eligible visual counterfactual.
            candidates = [j for j in range(B) if j != b and image_ids[j] != image_ids[b]]
            valid = make_cf and cfg.use_visual_stream and cfg.use_relevance and bool(candidates)
            if valid:
                other = candidates[torch.randint(len(candidates), ()).item()]
                cf_rs.append(self.relevance(t.unsqueeze(0), visual_feats[other])[2][0])
            else:
                cf_rs.append(r.new_zeros(()))
            cf_valid.append(valid)
            # Fallback KG negatives must be re-scored for this aspect, not use
            # another aspect's usefulness score. Exclude factual triples.
            current = {(tr.head, tr.relation, tr.tail) for tr in triples[a]}
            negatives = [(j, q) for j, row in enumerate(triples) if j != a
                         for q, tr in enumerate(row)
                         if (tr.head, tr.relation, tr.tail) not in current]
            if make_cf and cfg.use_kg_stream and cfg.use_kg_filter and negatives:
                j, q = negatives[torch.randint(len(negatives), ()).item()]
                cf_kg.append(self.kg_filter(t, gs[j][q:q+1])[0])
            else:
                cf_kg.append(t.new_zeros(0))
        z = torch.stack(zs) if zs else text_feats.new_zeros((0, d))
        offset = 0
        for b, row in enumerate(spans):
            for s, e in row:
                z_tok[b, s:e] = z[offset]
                offset += 1
        if self.training and cfg.evidence_dropout:
            keep = (torch.rand(B, 1, 1, device=text_feats.device) >= cfg.evidence_dropout)
            z_tok = z_tok * keep
        h = self.fuse_proj(torch.cat((text_feats, z_tok), -1))
        return dict(
            tag_logits=self.bio_head(h), anchor_logits=anchor_logits,
            asc_paper_logits=self.asc_head_paper(z), z=z,
            relevance=torch.stack(rs) if rs else text_feats.new_zeros(0),
            kg_scores=scores, kg_triples=triples,
            cf_relevance=torch.stack(cf_rs) if cf_rs else text_feats.new_zeros(0),
            cf_relevance_mask=torch.tensor(cf_valid, dtype=torch.bool, device=text_feats.device),
            cf_kg_scores=cf_kg, aspect_word_spans=words, aspect_spans=spans,
            aspect_batch_idx=torch.tensor(owners, device=text_feats.device, dtype=torch.long),
            alpha=alphas if want_alpha else None)

