"""TARKAN student model — assembles the updated methodology (Fig. 1, §3.2-§3.8).

Per image-text pair the student:
  1. encodes text/image (Eqs. 4-5),
  2. pools aspect reps over spans (Eq. 6) — gold spans in training, predicted in inference,
  3. estimates aspect-visual relevance and filters the image (Eqs. 7-10),
  4. retrieves + encodes aspect-centered KG triples (Eqs. 12-14),
  5. predicts KG usefulness and aggregates filtered KG evidence (Eqs. 15-17),
  6. fuses [h^t_i ; v_tilde ; g_tilde] PER TOKEN via KAN -> h̃_i (Eqs. 18-20),
  7. predicts the unified BIO aspect-sentiment tag from h̃_i (Eq. 21).

Updated paper §3.6: the BIO head performs BOTH aspect extraction and sentiment
classification (one 7-class sequence-labeling task) and runs on the KAN-fused
multimodal token representation h̃_i — there is no separate ASC head. Each aspect's
relevance-filtered visual (v_tilde_k) and teacher-filtered KG (g_tilde_k) evidence is
broadcast to that aspect's token positions; tokens outside any aspect get zero evidence.
Setting cfg.use_kan_tag_representation=False feeds the BIO head text-only features
(Table-6 ablation "w/o KAN-enhanced tag representation").

The offline LLM teacher never enters the forward pass — its signals (r^T, s^T) are
only *targets* consumed by losses.py. Ablation toggles (config) switch streams on/off
to reproduce Table 6.

Batch dict (from data.py collate) — forward consumes:
  input_ids [B,n], attention_mask [B,n], pixel_values [B,3,H,W] (optional if feats given),
  aspect_spans: List[B] of List[(start,end)]   (gold in training / predicted in inference;
    empty lists -> stage-1 extraction with zero aspect evidence),
  aspect_queries: List[B] of List[AspectQuery]   (for KG retrieval),
  aspect_triples: Optional List[B] of List[List[Triple]]  (precomputed/cached retrieval).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from config import CONFIG
from heads import BIOTaggingHead
from kan_fusion import build_fusion
from kg import KnowledgeGraph
from kg_filter import KGFilter
from kg_retrieval import AspectQuery, EntityEmbedder, TripleEncoder, retrieve_triples
from relevance import AspectVisualRelevance, pool_aspect


class TarkanStudent(nn.Module):
    def __init__(
        self,
        config=CONFIG,
        build_encoders: bool = True,
        kg: Optional[KnowledgeGraph] = None,
        entity_embedder: Optional[EntityEmbedder] = None,
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.cfg = config
        d = config.hidden_dim
        self.pool_mode = pool_mode
        self.kg = kg

        if build_encoders:
            from encoders import TextEncoder, VisualEncoder

            # Pass THIS model's config explicitly — the encoders' no-arg defaults read the
            # global CONFIG, which silently diverges from a replace()-constructed cfg (e.g.
            # the per-dataset bertweet-large override: tokenizer used large ids while the
            # encoder loaded base → vocabulary mismatch → garbage training).
            self.text_encoder = TextEncoder(model_id=config.text_model_id, hidden_dim=d, dropout=config.dropout)
            self.visual_encoder = VisualEncoder(model_id=config.visual_model_id, hidden_dim=d, dropout=config.dropout)
        else:  # tests/feature-precompute: feed text_feats/visual_feats to forward()
            self.text_encoder = None
            self.visual_encoder = None

        dp = config.dropout
        self.relevance = AspectVisualRelevance(d, dropout=dp)
        # paper §3.6: verbalize "h r t" and encode with the STUDENT'S encoder. Falls back to
        # the Numberbatch entity-embedding form when no text encoder exists (unit tests).
        if getattr(config, "kg_encoder", "verbalized") == "verbalized" and self.text_encoder is not None:
            from transformers import AutoTokenizer
            from kg_retrieval import VerbalizedTripleEncoder
            _kw = {"use_fast": False, "token": config.hf_token}
            if "bertweet" in config.text_model_id.lower():
                _kw["normalization"] = True
            self.triple_encoder = VerbalizedTripleEncoder(
                self.text_encoder, AutoTokenizer.from_pretrained(config.text_model_id, **_kw),
                d, dropout=dp)
        else:
            self.triple_encoder = TripleEncoder(d, embedder=entity_embedder, dropout=dp)
        self.kg_filter = KGFilter(d, dropout=dp)
        self._conf_extra = 3 if (getattr(config, "fusion_conf_append", False) and config.fusion == "kan") else 0
        self.fusion = build_fusion(config.fusion, d, dropout=dp, in_extra=self._conf_extra)
        # A10 (opt-in): feature-wise evidence gates, init 0 -> identity at start of training
        if getattr(config, "fusion_feat_gate", False):
            self.gate_gamma = nn.Parameter(torch.zeros(d))
            self.gate_delta = nn.Parameter(torch.zeros(d))
        else:
            self.gate_gamma = None
            self.gate_delta = None
        # A11 Evidence Reliability Learning: per-token softmax reliability over [text, vision, KG].
        # Final layer zero-init so w≈uniform (identity-ish) at start → stable warm start.
        if getattr(config, "fusion_reliability", False):
            self.reliability_mlp = nn.Sequential(
                nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dp), nn.Linear(d, 3)
            )
            nn.init.zeros_(self.reliability_mlp[-1].weight)
            nn.init.zeros_(self.reliability_mlp[-1].bias)
        else:
            self.reliability_mlp = None
        # tag_norm removed: paper Eq. 25 is h̃ = W_f[h^t; z_k] + b_f with no LayerNorm
        self.bio_head = BIOTaggingHead(d, dropout=dp)
        # paper §3.3 Eq. 6: the preliminary aspect-anchor generator. A lightweight
        # aspect-ONLY tagging head over h^t; its candidate spans are what conditions
        # relevance / KG retrieval / fusion at inference (never the gold spans).
        from config import NUM_ANCHOR_TAGS
        self.anchor_head = nn.Linear(d, NUM_ANCHOR_TAGS)
        # paper §3.8 Eq. 22: h̃_i = W_f [h^t_i ; z_k] + b_f, where z_k is the KAN-fused
        # representation of the candidate covering token i (zeros for uncovered tokens).
        self.fuse_proj = nn.Linear(2 * d, d)
        # Task-specific projections (minimal change #1). ONE shared TARKAN fusion, then a
        # small bottleneck per task so extraction and polarity stop competing for identical
        # features. This is NOT a second stage: both are computed in the same forward pass
        # from the same fused representation. §D PACS measured that forcing both behaviours
        # through one representation costs 2.22 joint F1.
        if getattr(config, "task_split", False):
            pd_ = getattr(config, "task_proj_dim", 384)
            self.p_ext = nn.Sequential(nn.Linear(d, pd_), nn.GELU(), nn.Linear(pd_, d))
            self.p_sent = nn.Sequential(nn.Linear(d, pd_), nn.GELU(), nn.Linear(pd_, d))
        else:
            self.p_ext = self.p_sent = None

        # Lightweight polarity experts over the SAME fused aspect representation z_k.
        # The old 19-member ensemble worked because its members made DIFFERENT mistakes
        # (§B.8: oracle 92.7 over members individually far lower). Three tiny branches with
        # a learned gate recreate that decorrelation inside one graph: one student, one
        # forward pass, one final prediction. Everything before z_k is shared.
        # Polarity-aware aspectness: one scalar coupling the polarity signal into the
        # aspect-bearing emissions. Zero-init means the model starts EXACTLY at the
        # baseline and has to learn that the coupling is worth anything.
        if getattr(config, "polarity_aware", False):
            init = float(getattr(config, "pa_lambda_init", 0.0))
            t = torch.tensor(init)
            self.pa_lambda = (nn.Parameter(t)
                              if getattr(config, "pa_lambda_learnable", True)
                              else nn.Parameter(t, requires_grad=False))
        else:
            self.pa_lambda = None

        n_exp = int(getattr(config, "polarity_experts", 0) or 0)
        if n_exp > 0:
            ed = getattr(config, "expert_dim", 256)
            self.experts = nn.ModuleList(
                nn.Sequential(nn.Linear(d, ed), nn.GELU(), nn.Dropout(dp), nn.Linear(ed, d))
                for _ in range(n_exp))
            self.expert_gate = nn.Linear(d, n_exp)
            # zero-init the gate so the mixture starts uniform, and residual so the model
            # begins exactly where it would without experts and has to earn any change
            nn.init.zeros_(self.expert_gate.weight)
            nn.init.zeros_(self.expert_gate.bias)
            self.expert_alpha = nn.Parameter(torch.tensor(0.1))
        else:
            self.experts = None
            self.expert_gate = None
        self.asc_head = None
        # The PAPER's auxiliary span-ASC head: a plain linear on the MEAN-pooled aspect
        # representation. Kept alongside the rich head deliberately — they see different
        # representations, so they are not redundant. Both contribute a loss; the rich head
        # is the inference polarity source (evaluate.py), this one is supervision only.
        if getattr(config, "aux_asc_head_paper", False):
            from config import NUM_POLARITIES
            self.asc_head_paper = nn.Linear(d, NUM_POLARITIES)
        else:
            self.asc_head_paper = None
        # NOVEL (not in the paper) — PDS: evidence-effect DIRECTION supervision.
        # r^T and s^T are both *relevance* signals: they say whether to look at a piece of
        # evidence, never what it does. This head predicts, per aspect, the distribution
        # over {NEG-shift, no-shift, POS-shift} that the teacher observed when the image
        # was added to the text — i.e. a constraint on how evidence may MOVE the decision.
        # It reads exactly the fusion input of Eq. 18, [t_k ; v~_k ; g~_k], so the
        # constraint lands on the evidence path rather than on the tagger.
        if getattr(config, "use_pds", False):
            self.pds_head = nn.Sequential(
                nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dp), nn.Linear(d, 3)
            )
        else:
            self.pds_head = None
        self.crf = None

    def set_kg(self, kg: KnowledgeGraph) -> None:
        self.kg = kg

    # ------------------------------------------------------------------ #
    def _gather_triples(self, spans_b, queries_b, triples_b, B):
        """Retrieve the triples for EVERY aspect in the batch, in (b, k) order.

        Retrieval is cached in kg_retrieval, so after the first epoch this is dictionary
        lookups. Kept separate from encoding so the encoding can be done in one shot.
        """
        out = []
        for b in range(B):
            per = []
            if self.cfg.use_kg_stream and self.kg is not None:
                qs = queries_b[b] if queries_b else []
                cached = triples_b[b] if triples_b is not None else None
                for k in range(len(spans_b[b])):
                    if cached is not None:
                        per.append(cached[k])
                    elif k < len(qs):
                        per.append(retrieve_triples(qs[k], self.kg, self.cfg.top_m_triples))
                    else:
                        per.append([])
            else:
                per = [[] for _ in spans_b[b]]
            out.append(per)
        return out

    def _encode_all_triples(self, triples_bk, d, device):
        """ONE encoder call for every triple in the batch, then split back per aspect.

        Encoding per aspect instead meant ~24 separate forwards of ~10 short sequences each
        per step — kernel-launch bound, and the reason mixed precision bought nothing.
        The maths is identical; only the batching changes.
        """
        flat, sizes = [], []
        for per in triples_bk:
            for trs in per:
                sizes.append(len(trs))
                flat.extend(trs)
        if not flat:
            return [[torch.zeros((0, d), device=device) for _ in per] for per in triples_bk]
        G = self.triple_encoder(flat)                     # [sum M, d], deduped internally
        outs, off, it = [], 0, iter(sizes)
        for per in triples_bk:
            row = []
            for _ in per:
                n = next(it)
                row.append(G[off:off + n])
                off += n
            outs.append(row)
        return outs

    def _aspect_evidence(self, t_k_all, V, queries, triples_cached, want_alpha,
                         g_pre=None):
        """Per-aspect evidence for ONE instance (Eqs. 7-17).

        Returns:
          v_tilde [K, d]   relevance-filtered visual (Eq. 10)
          r_k     [K]      aspect-visual relevance scores (Eq. 9; supervises L_rel)
          g_list  list[K]  filtered KG vector g_tilde_k [d] (Eq. 17)
          s_list  list[K]  per-triple KG usefulness scores (Eq. 15; supervises L_kg)
          tr_list list[K]  retrieved Triple lists
          alpha            attention weights if requested
        """
        cfg = self.cfg
        K = t_k_all.size(0)
        device = t_k_all.device
        d = cfg.hidden_dim

        # ---- visual stream (Eqs. 7-10) ----
        if cfg.use_visual_stream and K > 0:
            _, v_bar, r_k, v_tilde, alpha = self.relevance(t_k_all, V)
            if not cfg.use_relevance:
                # keep aspect-conditioned visual but drop the learned gate (Table 6)
                v_tilde = v_bar
                r_k = torch.ones(K, device=device)
        else:
            v_tilde = torch.zeros((K, d), device=device)
            r_k = torch.zeros((K,), device=device)
            alpha = None

        g_list, s_list, tr_list = [], [], []
        for k in range(K):
            t_k = t_k_all[k]
            # ---- KG stream (Eqs. 12-17) ----
            if cfg.use_kg_stream and self.kg is not None:
                triples = triples_cached[k] if triples_cached is not None else retrieve_triples(
                    queries[k], self.kg, cfg.top_m_triples
                )
                # embeddings come from the single batched encode (see _encode_all_triples)
                g = (g_pre[k] if g_pre is not None
                     else self.triple_encoder(triples))       # [M, d]
                if cfg.use_kg_filter:
                    s, g_tilde = self.kg_filter(t_k, g)       # Eqs. 15, 17
                else:
                    s = g.new_zeros((g.size(0),))
                    g_tilde = g.mean(dim=0) if g.size(0) > 0 else g.new_zeros((d,))  # unfiltered mean
            else:
                triples, s, g_tilde = [], torch.zeros((0,), device=device), torch.zeros((d,), device=device)
            g_list.append(g_tilde)
            s_list.append(s)
            tr_list.append(triples)

        return v_tilde, r_k, g_list, s_list, tr_list, (alpha if want_alpha else None)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        batch: Dict,
        text_feats: Optional[torch.Tensor] = None,
        visual_feats: Optional[torch.Tensor] = None,
        want_alpha: bool = False,
    ) -> Dict:
        if text_feats is None:
            text_feats = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        if visual_feats is None and self.cfg.use_visual_stream:
            visual_feats = self.visual_encoder(batch["pixel_values"])

        cfg = self.cfg
        B, n, d = text_feats.shape

        # ---- §3.3 Eq. 6: preliminary aspect-anchor distributions ----
        anchor_logits = self.anchor_head(text_feats)          # [B, n, 3] over O/B-ASP/I-ASP

        # z_k broadcast to the tokens of its candidate; zeros elsewhere (§3.8 Eq. 22)
        z_tok = text_feats.new_zeros((B, n, d))
        conf_tok = text_feats.new_zeros((B, n, 3)) if self._conf_extra else None  # A9 [r, mean(s), max(s)]

        all_r, all_s, all_tr, all_alpha, owner = [], [], [], [], []
        all_z = []    # per-aspect KAN-fused z_k in (b, k) order (§3.7)
        pds_in = []   # per-aspect [t_k ; v~_k ; g~_k] in (b, k) order, for the PDS head
        spans_b = batch["aspect_spans"]
        queries_b = batch.get("aspect_queries", [[] for _ in range(B)])
        triples_b = batch.get("aspect_triples", None)

        # ---- one retrieval pass + ONE encoder call for the whole batch ----
        triples_bk = self._gather_triples(spans_b, queries_b, triples_b, B)
        g_bk = self._encode_all_triples(triples_bk, d, text_feats.device)

        for b in range(B):
            spans = spans_b[b]
            t_k_all = pool_aspect(text_feats[b], spans, self.pool_mode)  # [K, d]
            V = visual_feats[b] if (visual_feats is not None) else text_feats.new_zeros((1, d))
            v_tilde, r_k, g_list, s_list, tr_list, alpha = self._aspect_evidence(
                t_k_all, V, queries_b[b] if queries_b else [], triples_bk[b], want_alpha,
                g_pre=g_bk[b]
            )
            K = t_k_all.size(0)
            # ---- §3.7 Eqs. 19-21: fuse PER ASPECT, u_k = [t_k ; v~_k ; g~_k] -> z_k ----
            if K:
                g_stack = torch.stack(g_list, 0)                       # [K, d]
                if cfg.use_kan_tag_representation:
                    z_k = self.fusion(t_k_all, v_tilde, g_stack)       # [K, d]
                else:  # Table-6 "w/o KAN-enhanced tag representation": text only reaches the head
                    z_k = t_k_all
                if self.experts is not None:
                    # gate and mixture both read the SAME z_k -> one forward pass
                    w = torch.softmax(self.expert_gate(z_k), dim=-1)   # [K, n_exp]
                    mix = sum(w[:, e:e + 1] * self.experts[e](z_k)
                              for e in range(len(self.experts)))       # [K, d]
                    z_k = z_k + self.expert_alpha * mix                # residual
                    self.last_gate = w
                all_z.append(z_k)
            # broadcast z_k to the tokens of its candidate (§3.8 Eq. 22)
            for k in range(K):
                s_, e_ = spans[k][0], spans[k][1]
                z_tok[b, s_:e_] = z_k[k]
                if conf_tok is not None:
                    conf_tok[b, s_:e_, 0] = r_k[k]
                    if s_list[k].numel():
                        conf_tok[b, s_:e_, 1] = s_list[k].mean()
                        conf_tok[b, s_:e_, 2] = s_list[k].max()
            if K:
                all_r.append(r_k)
                all_s.extend(s_list)
                all_tr.extend(tr_list)
                owner.extend([b] * K)
                if alpha is not None:
                    all_alpha.extend(list(alpha))
                if self.pds_head is not None:
                    pds_in.append(torch.cat(
                        [t_k_all, v_tilde, torch.stack(g_list, 0)], dim=-1))   # [K, 3d]

        # ---- §3.7 ACEQ: construct counterfactual evidence for training ----
        cf_r = None
        cf_kg = None
        if self.training and getattr(cfg, 'lambda_cf', 0.0) > 0 and all_r:
            # Visual counterfactual: for each aspect, compute relevance with a
            # mismatched image from another instance in the batch
            cf_r_list = []
            for b in range(B):
                K_b = len(spans_b[b])
                if K_b == 0:
                    continue
                # Pick a different instance's visual features
                other_b = (b + 1) % B
                V_other = visual_feats[other_b] if (visual_feats is not None) else text_feats.new_zeros((1, d))
                t_k_b = pool_aspect(text_feats[b], spans_b[b], self.pool_mode)
                if cfg.use_visual_stream and K_b > 0:
                    _, _, cf_r_k, _, _ = self.relevance(t_k_b, V_other)
                    cf_r_list.append(cf_r_k)
            cf_r = torch.cat(cf_r_list, 0) if cf_r_list else text_feats.new_zeros((0,))
            
            # KG counterfactual: for each aspect, the teacher-rejected triples
            # are already in kg_scores (s_kq scores); we just need to identify
            # which are approved vs rejected using teacher labels. The actual
            # pairing happens in aceq_loss in losses.py.
            # We pass cf_kg_scores = kg_scores (same scores, different masking in loss)

        # ---- §3.8 Eq. 22: evidence-enhanced token representation ----
        #   h̃_i = W_f [h^t_i ; z_k] + b_f     for tokens covered by candidate a_k
        #   z_k = 0                            for tokens under no candidate, so their
        #                                      representation stays primarily text-conditioned
        # Evidence dropout (train only): zero a random subset of instances' z_k so the tag
        # head also learns to work from text alone, which is the regime it meets whenever the
        # anchor generator proposes nothing.
        if self.training and getattr(cfg, "evidence_dropout", 0.0) > 0:
            keep = (torch.rand(B, 1, 1, device=text_feats.device) >= cfg.evidence_dropout).to(text_feats.dtype)
            z_tok = z_tok * keep
            if conf_tok is not None:
                conf_tok = conf_tok * keep
        h_tilde = self.fuse_proj(torch.cat([text_feats, z_tok], dim=-1))

        h_ext = self.p_ext(h_tilde) if self.p_ext is not None else h_tilde
        tag_logits = self.bio_head(h_ext)    # Eq. 23 -> [B, n, 7]

        # ---- polarity-aware aspectness ----
        # p^pol_i is the token's polarity belief RENORMALISED over {POS, NEU, NEG}, so it
        # measures "is there a coherent sentiment here" independently of how aspect-y the
        # token already looks (P(O) drops out). c_i is its normalised confidence.
        # c_i is DETACHED: it conditions the emissions as a feature, and cannot be gamed by
        # the model making polarity artificially confident to win aspectness.
        if self.pa_lambda is not None:
            import math as _math
            p7 = torch.softmax(tag_logits, dim=-1)
            pol = torch.stack([p7[..., 1] + p7[..., 2],      # POS = B-POS + I-POS
                               p7[..., 3] + p7[..., 4],      # NEU
                               p7[..., 5] + p7[..., 6]], -1)  # NEG
            pol = pol / pol.sum(-1, keepdim=True).clamp_min(1e-9)
            ent = -(pol * pol.clamp_min(1e-9).log()).sum(-1)
            c_i = (1.0 - ent / _math.log(3.0)).detach()      # [B, n] in [0, 1]
            bump = torch.zeros_like(tag_logits)
            bump[..., 1:] = (self.pa_lambda * c_i).unsqueeze(-1)   # aspect-bearing tags only
            tag_logits = tag_logits + bump
            self.last_pa_c = c_i

        # A7: dedicated ASC polarity logits over the pooled aspect rep of h̃, in the SAME
        # (b, k) aspect order that teacher.build_targets uses (so losses/eval align).
        z_cat = torch.cat(all_z, 0) if all_z else text_feats.new_zeros((0, d))

        # §3.8 Eq. 25 — the paper's auxiliary span-ASC head: a linear on the KAN-fused z_k.
        z_sent = self.p_sent(z_cat) if (self.p_sent is not None and z_cat.numel()) else z_cat
        asc_paper_logits = None
        if self.asc_head_paper is not None:
            asc_paper_logits = self.asc_head_paper(z_sent)

        # The rich head (ours): mean+max+first pooling of h̃ over the span -> MLP. It sees a
        # different representation from the paper head above, so the two are not redundant.
        asc_logits = None
        if self.asc_head is not None:
            h_sent_tok = self.p_sent(h_tilde) if self.p_sent is not None else h_tilde
            reps = []
            for b in range(B):
                for sp in spans_b[b]:
                    s_, e_ = sp[0], sp[1]
                    chunk = h_sent_tok[b, s_:e_]
                    if chunk.numel():
                        rep = torch.cat([chunk.mean(dim=0), chunk.max(dim=0).values, chunk[0]], dim=-1)
                    else:
                        rep = h_tilde.new_zeros((3 * d,))
                    reps.append(rep)
            asc_logits = (self.asc_head(torch.stack(reps, 0)) if reps
                          else h_tilde.new_zeros((0, 3)))

        r_cat = torch.cat(all_r, 0) if all_r else text_feats.new_zeros((0,))

        pds_logits = None
        if self.pds_head is not None:
            pds_logits = (self.pds_head(torch.cat(pds_in, 0)) if pds_in
                          else text_feats.new_zeros((0, 3)))

        return {
            "tag_logits": tag_logits,
            "anchor_logits": anchor_logits,   # [B, n, 3] §3.3 Eq. 6 -> L_anc
            "z": z_cat,                       # [sumK, d] §3.7 fused aspect representations
            "asc_logits": asc_logits,  # [sumK, 3] rich head — the inference polarity source
            "asc_paper_logits": asc_paper_logits,  # [sumK, 3] paper's linear auxiliary
            "pds_logits": pds_logits,  # [sumK, 3] in (b,k) order, or None if PDS off
            "relevance": r_cat,
            "cf_relevance": cf_r,      # [sumK] counterfactual visual relevance for ACEQ
            "kg_scores": all_s,        # list[sumK] of [M_k]
            "cf_kg_scores": all_s,
            "kg_triples": all_tr,      # list[sumK] of list[Triple]
            "aspect_batch_idx": torch.tensor(owner, dtype=torch.long, device=text_feats.device),
            "alpha": all_alpha if want_alpha else None,
        }
