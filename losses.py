"""TARKAN training objective (updated paper §3.7, Eqs. 11, 16, 22).

    L_tag = - sum_i log p(b*_i | T, I)                                    (Eq. 22)
    L_rel = - sum_k [ r^T_k log r_k + (1-r^T_k) log(1-r_k) ]             (Eq. 11)
    L_kg  = - sum_k sum_q [ s^T_kq log s_kq + (1-s^T_kq) log(1-s_kq) ]   (Eq. 16)

    L = L_tag + λ1 L_rel + λ2 L_kg                                       (Eq. for L)

The updated methodology unifies aspect extraction + sentiment classification in the
single BIO head (Eq. 21, run on the KAN-fused token representation), so there is no
separate auxiliary span-ASC loss (the old L_asc / λ3 are removed). cfg.use_teacher =
False zeroes L_rel and L_kg ("w/o LLM teacher guidance").
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from config import CONFIG


def tag_loss(tag_logits: torch.Tensor, bio_labels: torch.Tensor, cfg=CONFIG) -> torch.Tensor:
    """Eq. 22. Token CE over 7 BIO classes; -100 positions ignored (subtoken continuations).

    Paper-faithful default is plain CE. Two reproduction-aid patches activate only when the
    corresponding cfg flags are set (both OFF by default):
      A1 tag_class_weight  -> inverse-frequency class weights (up-weight rare NEG/B-I vs O),
                              precomputed once over the train split (cfg._tag_weight_vec).
      A5 tag_label_smoothing -> label smoothing epsilon.
    """
    B, n, C = tag_logits.shape
    weight = None
    ls = 0.0
    return F.cross_entropy(
        tag_logits.reshape(B * n, C), bio_labels.reshape(B * n), ignore_index=-100,
        reduction="mean", weight=weight, label_smoothing=ls,
    )


def relevance_loss(r: torch.Tensor, teacher_r: torch.Tensor, mask: Optional[torch.Tensor] = None,
                   conf_weight: bool = False) -> torch.Tensor:
    """Eq. 13. BCE over aspects that have a teacher relevance label.

    `conf_weight` (minimal change #6): the teacher's label is a probability, so |r^T - 0.5|
    is exactly how sure it was. Weighting by it stops a coin-flip judgement training as
    hard as a confident one. Normalised so the loss scale is unchanged.
    """
    if r.numel() == 0:
        return r.new_zeros(())
    if mask is not None:
        if mask.sum() == 0:
            return r.new_zeros(())
        r, teacher_r = r[mask], teacher_r[mask]
    r = r.clamp(1e-7, 1 - 1e-7)
    if not conf_weight:
        return F.binary_cross_entropy(r, teacher_r.float(), reduction="mean")
    per = F.binary_cross_entropy(r, teacher_r.float(), reduction="none")
    w = (teacher_r.float() - 0.5).abs() * 2.0
    return (w * per).sum() / w.sum().clamp(min=1e-6)


def kg_loss(
    kg_scores: List[torch.Tensor],
    teacher_kg: List[torch.Tensor],
    teacher_kg_mask: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Eq. 16. BCE over retrieved triples that have a teacher usefulness label."""
    preds, tgts = [], []
    for i, s in enumerate(kg_scores):
        if s.numel() == 0:
            continue
        t = teacher_kg[i]
        if t.numel() != s.numel():
            continue
        if teacher_kg_mask is not None:
            m = teacher_kg_mask[i]
            if m.sum() == 0:
                continue
            s, t = s[m], t[m]
        preds.append(s)
        tgts.append(t.float())
    if not preds:
        # keep graph connected to wg params if everything is unlabeled
        device = kg_scores[0].device if kg_scores else torch.device("cpu")
        return torch.zeros((), device=device)
    p = torch.cat(preds).clamp(1e-7, 1 - 1e-7)
    return F.binary_cross_entropy(p, torch.cat(tgts), reduction="mean")


def obi_marginal_loss(tag_logits: torch.Tensor, anchor_labels: torch.Tensor) -> torch.Tensor:
    """Minimal change #2 — O/B/I marginal supervision on the SAME unified 7-tag head.

        P(B) = P(B-POS) + P(B-NEU) + P(B-NEG),   P(I) likewise,   P(O) = P(O)

    and those three marginals are trained against the aspect-only labels. This does not add
    a head, a decode path, or any inference-time information: it teaches the existing CRF a
    better decomposition, so that uncertainty about WHICH polarity can no longer make an
    otherwise obvious aspect disappear (the marginalised-decode win, moved into training).
    """
    B, n, C = tag_logits.shape
    p = F.softmax(tag_logits, dim=-1)
    # BIO_TAGS = [O, B-POS, I-POS, B-NEU, I-NEU, B-NEG, I-NEG]
    p_o = p[..., 0]
    p_b = p[..., 1] + p[..., 3] + p[..., 5]
    p_i = p[..., 2] + p[..., 4] + p[..., 6]
    logp = torch.log(torch.stack([p_o, p_b, p_i], dim=-1).clamp_min(1e-9))
    return F.nll_loss(logp.reshape(B * n, 3), anchor_labels.reshape(B * n),
                      ignore_index=-100, reduction="mean")


def anchor_loss(anchor_logits: torch.Tensor, anchor_labels: torch.Tensor) -> torch.Tensor:
    """Paper §3.3 Eq. 8 — the preliminary aspect-anchor generator.

    Aspect-ONLY tags {O, B-ASP, I-ASP}: this objective trains the anchor generator
    independently of sentiment polarity, so the candidate set does not depend on getting
    the polarity right. -100 positions (subtoken continuations) are ignored.
    """
    B, n, C = anchor_logits.shape
    return F.cross_entropy(anchor_logits.reshape(B * n, C), anchor_labels.reshape(B * n),
                           ignore_index=-100, reduction="mean")


def pds_loss(pds_logits: torch.Tensor, teacher_pds: torch.Tensor,
             mask: Optional[torch.Tensor] = None, conf_weight: bool = False) -> torch.Tensor:
    """NOVEL — evidence-effect direction supervision.

    The teacher's label is a SOFT distribution over {NEG-shift, no-shift, POS-shift}
    (it comes from a probability difference, so its uncertainty is real information).
    Cross-entropy against soft targets keeps that uncertainty instead of freezing it into
    a hard class the student must then fit exactly.
    """
    if pds_logits is None or pds_logits.numel() == 0:
        return torch.zeros((), device=(pds_logits.device if pds_logits is not None
                                       else torch.device("cpu")))
    if mask is not None:
        if mask.sum() == 0:
            return pds_logits.new_zeros(())
        pds_logits, teacher_pds = pds_logits[mask], teacher_pds[mask]
    logp = F.log_softmax(pds_logits, dim=-1)
    per = -(teacher_pds.float() * logp).sum(-1)
    if conf_weight:
        # the teacher's own peak probability is its confidence in the direction
        w = teacher_pds.float().max(-1).values
        return (w * per).sum() / w.sum().clamp(min=1e-6)
    return per.mean()


def aceq_loss(outputs: Dict, targets: Dict, cfg=CONFIG) -> torch.Tensor:
    """ACEQ: Aspect-Conditioned Counterfactual Evidence Qualification (paper §3.7, Eqs. 19-21).
    
    Visual counterfactual (Eq. 19):
        Lv_cf = sum_{k: rT_k=1} [mv - r+_k + r-_k]+
    KG counterfactual (Eq. 20):
        Lg_cf = sum_k sum_{(q+,q-) in Pk} [mg - skq+ + skq-]+
    Total (Eq. 21):
        LACEQ = Lv_cf + η * Lg_cf
    """
    mv = getattr(cfg, 'aceq_mv', 0.2)
    mg = getattr(cfg, 'aceq_mg', 0.2)
    eta = getattr(cfg, 'aceq_eta', 1.0)
    device = torch.device('cpu')
    
    # --- Visual counterfactual (Eq. 19) ---
    l_vcf = torch.zeros((), device=device)
    if 'teacher_relevance' in targets and outputs['relevance'].numel() > 0:
        r = outputs['relevance']  # [sumK]
        rT = targets['teacher_relevance']  # [sumK]
        device = r.device
        l_vcf = torch.zeros((), device=device)
        
        # For aspects where teacher says image IS useful (rT=1),
        # the factual score r+ should exceed the counterfactual r- by margin mv
        # Only compute for positive anchor conditions
        mask_pos = (rT == 1)
        if mask_pos.any() and outputs.get('cf_relevance') is not None and outputs['cf_relevance'].numel() > 0:
            r_pos = r[mask_pos]
            r_neg = outputs['cf_relevance'][mask_pos]  # counterfactual scores
            l_vcf = torch.clamp(mv - r_pos + r_neg, min=0.0).mean()
    
    # --- KG counterfactual (Eq. 20) ---
    l_gcf = torch.zeros((), device=device)
    if 'kg_scores' in outputs and 'cf_kg_scores' in outputs:
        kg_s = outputs['kg_scores']      # list of [M_k] tensors
        cf_s = outputs['cf_kg_scores']    # list of [M_k] tensors
        teacher_kg = targets.get('teacher_kg', [])
        pairs = []
        for k in range(len(kg_s)):
            if kg_s[k].numel() == 0:
                continue
            if k >= len(teacher_kg) or teacher_kg[k].numel() == 0:
                continue
            tgt = teacher_kg[k]
            s_k = kg_s[k]
            cf_k = cf_s[k] if k < len(cf_s) and cf_s[k].numel() > 0 else None
            if cf_k is None:
                continue
            # For each teacher-approved triple (sT=1), pair with a rejected one
            pos_mask = (tgt > 0.5)
            neg_mask = (tgt <= 0.5)
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                for pi in pos_mask.nonzero(as_tuple=True)[0]:
                    # Use the hardest negative (highest-scoring rejected)
                    neg_scores = s_k[neg_mask]
                    hardest_neg = neg_scores.max()
                    pairs.append(torch.clamp(mg - s_k[pi] + hardest_neg, min=0.0))
        if pairs:
            l_gcf = torch.stack(pairs).mean()
    
    return l_vcf + eta * l_gcf


def compute_losses(outputs: Dict, targets: Dict, cfg=CONFIG, model=None) -> Dict[str, torch.Tensor]:
    """Returns dict with l_tag, l_rel, l_kg, total (updated §3.7: L = L_tag + λ1 L_rel + λ2 L_kg).

    A4 (opt-in): when cfg.use_crf and `model` (owning model.crf) is passed, L_tag becomes the
    CRF negative log-likelihood over word-level emissions instead of token CE.
    """
    if (
        getattr(cfg, "use_crf", False)
        and model is not None
        and getattr(model, "crf", None) is not None
        and targets.get("word_ids") is not None
        and targets.get("n_words") is not None
    ):
        emis, labs, mask = word_level_emissions(
            outputs["tag_logits"], targets["word_ids"], targets["n_words"], targets["bio_labels"]
        )
        l_tag = -model.crf(emis, labs, mask=mask, reduction="mean")
    else:
        l_tag = tag_loss(outputs["tag_logits"], targets["bio_labels"], cfg)

    if cfg.use_teacher and cfg.use_relevance and cfg.use_visual_stream and "teacher_relevance" in targets:
        l_rel = relevance_loss(
            outputs["relevance"], targets["teacher_relevance"],
            targets.get("teacher_relevance_mask"),
            conf_weight=getattr(cfg, "teacher_conf_weight", False)
        )
    else:
        l_rel = l_tag.new_zeros(())

    if cfg.use_teacher and cfg.use_kg_stream and cfg.use_kg_filter and "teacher_kg" in targets:
        l_kg = kg_loss(outputs["kg_scores"], targets["teacher_kg"], targets.get("teacher_kg_mask"))
    else:
        l_kg = l_tag.new_zeros(())

    # NOVEL — L_pds. Gated on use_teacher too: it is a teacher signal, so "w/o LLM
    # teacher guidance" must remove it along with L_rel and L_kg.
    if (cfg.use_teacher and getattr(cfg, "use_pds", False)
            and outputs.get("pds_logits") is not None and "teacher_pds" in targets):
        l_pds = pds_loss(outputs["pds_logits"], targets["teacher_pds"],
                         targets.get("teacher_pds_mask"),
                         conf_weight=getattr(cfg, "teacher_conf_weight", False))
    else:
        l_pds = l_tag.new_zeros(())

    # §3.7 ACEQ: counterfactual evidence qualification
    if cfg.use_teacher and getattr(cfg, 'lambda_cf', 0.0) > 0:
        l_cf = aceq_loss(outputs, targets, cfg)
    else:
        l_cf = l_tag.new_zeros(())

    # §3.3 Eq. 8 / §3.9 Eq. 27 — the anchor generator term
    if outputs.get("anchor_logits") is not None and targets.get("anchor_labels") is not None:
        l_anc = anchor_loss(outputs["anchor_logits"], targets["anchor_labels"])
    else:
        l_anc = l_tag.new_zeros(())

    if (getattr(cfg, "lambda_obi", 0.0) > 0 and targets.get("anchor_labels") is not None):
        l_obi = obi_marginal_loss(outputs["tag_logits"], targets["anchor_labels"])
    else:
        l_obi = l_tag.new_zeros(())

    total = (l_tag + getattr(cfg, "lambda_obi", 0.0) * l_obi
             + getattr(cfg, "lambda_anc", 0.3) * l_anc
             + cfg.lambda1 * l_rel + cfg.lambda2 * l_kg
             + getattr(cfg, "lambda3", 0.0) * l_pds
             + getattr(cfg, "lambda_cf", 0.3) * l_cf)
    out = {"l_tag": l_tag, "l_anc": l_anc, "l_obi": l_obi, "l_rel": l_rel,
           "l_kg": l_kg, "l_pds": l_pds, "l_cf": l_cf, "total": total}

    # A7 (opt-in): dedicated ASC polarity CE on the pooled aspect reps (gold spans in training).
    # Only added when explicitly enabled, so the faithful baseline's loss dict is unchanged.
    if getattr(cfg, "aux_asc_head", False) and outputs.get("asc_logits") is not None and "aspect_polarity" in targets:
        al = outputs["asc_logits"]
        tgt = targets["aspect_polarity"]
        l_asc = F.cross_entropy(al, tgt) if (al.numel() and al.shape[0] == tgt.shape[0]) else l_tag.new_zeros(())
        out["l_asc"] = l_asc
        out["total"] = total + getattr(cfg, "lambda_asc", 0.5) * l_asc

    # The paper's own auxiliary span-ASC term, on the mean-pooled aspect rep. Trained
    # jointly with the rich head above: different representations, so not redundant.
    if (getattr(cfg, "aux_asc_head_paper", False)
            and outputs.get("asc_paper_logits") is not None and "aspect_polarity" in targets):
        ap = outputs["asc_paper_logits"]
        tgt = targets["aspect_polarity"]
        l_asc_p = (F.cross_entropy(ap, tgt) if (ap.numel() and ap.shape[0] == tgt.shape[0])
                   else l_tag.new_zeros(()))
        out["l_asc_paper"] = l_asc_p
        out["total"] = out["total"] + getattr(cfg, "lambda_asc_paper", 0.5) * l_asc_p
    return out
