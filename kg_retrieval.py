"""Aspect-centered KG retrieval + triple encoding (paper Eqs. 12-14).

    Q_k = {a_k} ∪ O_k ∪ C_k                                              (Eq. 12)
        O_k = sentiment/opinion words near the aspect (spaCy)
        C_k = visual concepts (caption nouns / CLIP zero-shot)
    G_k = top-M retrieved triples (e_p, r, e_q)                          (Eq. 13)
    g_kq = phi([e_p ; r ; e_q])                                          (Eq. 14)

Retrieval/scoring is non-differentiable preprocessing; only TripleEncoder has
parameters. Entity embeddings come from ConceptNet Numberbatch (English, 300-d)
with a deterministic hash fallback for OOV / offline tests.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from config import CONFIG
from kg import KnowledgeGraph, Triple, normalize

# Canonical relation vocabulary (ConceptNet 34 + SenticNet edge types). Unknown -> <unk>.
CONCEPTNET_RELATIONS = [
    "RelatedTo", "FormOf", "IsA", "PartOf", "HasA", "UsedFor", "CapableOf", "AtLocation",
    "Causes", "HasSubevent", "HasFirstSubevent", "HasLastSubevent", "HasPrerequisite",
    "HasProperty", "MotivatedByGoal", "ObstructedBy", "Desires", "CreatedBy", "Synonym",
    "Antonym", "DistinctFrom", "DerivedFrom", "SymbolOf", "DefinedAs", "MannerOf",
    "LocatedNear", "HasContext", "SimilarTo", "EtymologicallyRelatedTo",
    "EtymologicallyDerivedFrom", "CausesDesire", "MadeOf", "ReceivesAction", "ExternalURL",
]
SENTICNET_RELATIONS = ["HasPolarity", "HasMood", "SemanticallyRelated"]
RELATION_VOCAB = ["<unk>"] + CONCEPTNET_RELATIONS + SENTICNET_RELATIONS
REL2ID = {r: i for i, r in enumerate(RELATION_VOCAB)}


@dataclass
class AspectQuery:
    aspect_term: str
    opinion_words: List[str] = field(default_factory=list)   # O_k
    visual_concepts: List[str] = field(default_factory=list)  # C_k

    def terms(self) -> List[str]:
        seen, out = set(), []
        for t in [self.aspect_term, *self.opinion_words, *self.visual_concepts]:
            k = normalize(t)
            if k and k not in seen:
                seen.add(k)
                out.append(t)
        return out


def _triple_key(t: Triple) -> str:
    return f"{t.head}|{t.relation}|{t.tail}"


_RETRIEVAL_CACHE: Dict[tuple, List[Triple]] = {}


def retrieve_triples(
    query: AspectQuery,
    kg: KnowledgeGraph,
    top_m: int = None,
    teacher_scores: Optional[Dict[str, float]] = None,
) -> List[Triple]:
    """Memoized wrapper. Retrieval is deterministic preprocessing keyed entirely by the
    query terms, and every training aspect is retrieved again on every epoch — ~380
    sqlite round-trips per step, all identical after epoch 1. Caching them is
    result-neutral and removes about a third of the step time. Skipped when teacher
    scores are supplied (those vary per call)."""
    if teacher_scores is None:
        key = (tuple(query.terms()), top_m or CONFIG.top_m_triples)
        hit = _RETRIEVAL_CACHE.get(key)
        if hit is not None:
            return hit
        out = _retrieve_triples_uncached(query, kg, top_m, None)
        if len(_RETRIEVAL_CACHE) < 400_000:
            _RETRIEVAL_CACHE[key] = out
        return out
    return _retrieve_triples_uncached(query, kg, top_m, teacher_scores)


def _retrieve_triples_uncached(
    query: AspectQuery,
    kg: KnowledgeGraph,
    top_m: int = None,
    teacher_scores: Optional[Dict[str, float]] = None,
) -> List[Triple]:
    """Gather neighbours of all query terms, score, keep top-M (Eq. 13).

    score = weight + lexical_match + affective_relevance + relation_prior (+ teacher)
    (equal-weight combination — Open-Q #8; deterministic tie-break).
    """
    top_m = top_m or CONFIG.top_m_triples
    cand: Dict[str, Triple] = {}
    aspect_key = normalize(query.aspect_term)
    qkeys = {normalize(t) for t in query.terms()}
    # Gather a quota PER SOURCE. kg.neighbors() truncates by raw weight, and SenticNet
    # edges all carry weight 1.0 while ConceptNet assertions go up to 16 — so a single
    # pooled fetch deletes SenticNet before the affective-relevance term in score()
    # below ever sees it (measured: 6.9% SenticNet against the paper's 41.6%, Table 8).
    # Retrieving each source separately lets the scorer, not the weight scale, decide.
    sources = CONFIG.kg_sources or (None,)
    for term in query.terms():
        for src in sources:
            for tr in kg.neighbors(term, top=top_m * 4, sources=(src,) if src else None):
                cand[_triple_key(tr)] = tr

    # ---- Eq. 15: s_ret = alpha*s_sem + beta*s_aff + gamma*s_rel, all on a common [0,1]
    # scale, with (alpha, beta, gamma) = (0.5, 0.3, 0.2) from Table 5. ----
    alpha, beta, gamma = getattr(CONFIG, "kg_rank_weights", (0.5, 0.3, 0.2))
    items = list(cand.values())
    if not items:
        return []

    # s_sem — semantic correspondence with the query context, min-max normalised to [0,1].
    # (The differentiable cosine(t_k, g_kq) of the paper is unavailable at retrieval time,
    # which is preprocessing; lexical overlap with Q_k is its non-parametric stand-in.)
    raw_sem = []
    for tr in items:
        hit = 0.0
        if tr.head == aspect_key:
            hit += 1.0
        if normalize(tr.tail) in qkeys:
            hit += 1.0
        raw_sem.append(hit + 0.1 * float(tr.weight))
    lo, hi = min(raw_sem), max(raw_sem)
    rng = (hi - lo) or 1.0

    # s_aff — SenticNet polarity magnitude normalised to [0,1]; ConceptNet triples with no
    # affective score get the neutral prior of 0.5, exactly as the paper specifies.
    def s_aff(tr: Triple) -> float:
        pol = kg.polarity(tr.tail)
        if pol is None:
            pol = kg.polarity(tr.head)
        return min(1.0, abs(float(pol))) if pol is not None else 0.5

    # s_rel — fixed relation weights: sentiment-bearing and descriptive relations rank above
    # weakly informative ones.
    SENTIMENT_REL = {"HasPolarity", "HasProperty", "Causes", "CausesDesire", "Desires",
                     "SemanticallyRelated", "HasMood"}
    DESCRIPTIVE_REL = {"IsA", "RelatedTo", "PartOf", "HasA", "MadeOf", "AtLocation",
                       "UsedFor", "CapableOf", "SimilarTo", "Synonym"}

    def s_rel(tr: Triple) -> float:
        if tr.relation in SENTIMENT_REL:
            return 1.0
        if tr.relation in DESCRIPTIVE_REL:
            return 0.6
        return 0.2

    def score(i: int, tr: Triple) -> float:
        s = (alpha * ((raw_sem[i] - lo) / rng) + beta * s_aff(tr) + gamma * s_rel(tr))
        if teacher_scores is not None:
            s += float(teacher_scores.get(_triple_key(tr), 0.0))
        return s

    order = sorted(range(len(items)),
                   key=lambda i: (-score(i, items[i]), items[i].relation, items[i].tail))
    return [items[i] for i in order[:top_m]]


class EntityEmbedder:
    """Surface term -> 300-d vector via Numberbatch, with deterministic hash fallback."""

    def __init__(self, numberbatch: Optional[Dict[str, np.ndarray]] = None, dim: int = None):
        self.dim = dim or CONFIG.entity_emb_dim
        self.table = numberbatch or {}

    @classmethod
    def from_txt(cls, path: str, dim: int = None, vocab: Optional[set] = None) -> "EntityEmbedder":
        """Load numberbatch-en-19.08.txt(.gz), STREAMING line-by-line (never reads the
        whole file at once). Pass `vocab` (a set of normalized terms) to keep only the
        embeddings you need — cuts memory from ~600 MB to just the KG vocabulary.
        """
        import gzip

        table: Dict[str, np.ndarray] = {}
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            first = f.readline().split()
            d = int(first[1]) if len(first) == 2 else CONFIG.entity_emb_dim
            if len(first) != 2:  # first line was already a vector
                key = normalize(first[0].split("/")[-1])
                if vocab is None or key in vocab:
                    table[key] = np.asarray(first[1:], dtype=np.float32)
            for line in f:
                parts = line.rstrip().split(" ")
                key = normalize(parts[0].split("/")[-1])  # /c/en/word -> word
                if vocab is not None and key not in vocab:
                    continue
                table[key] = np.asarray(parts[1:], dtype=np.float32)
        return cls(table, dim=d)

    def embed(self, term: str) -> np.ndarray:
        key = normalize(term)
        if key in self.table:
            return self.table[key]
        # deterministic pseudo-embedding so OOV / offline still works (unit-norm).
        h = hashlib.sha256(key.encode("utf-8")).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
        v = rng.randn(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)


def verbalize(t: Triple) -> str:
    """Paper §3.6: each retrieved triple (h, r, t) is verbalized as "h r t"."""
    rel = re.sub(r"(?<!^)(?=[A-Z])", " ", str(t.relation)).replace("_", " ").lower().strip()
    return f"{str(t.head).replace('_', ' ')} {rel} {str(t.tail).replace('_', ' ')}"


class VerbalizedTripleEncoder(nn.Module):
    """Paper §3.6 — encode the verbalized triple with the STUDENT'S text encoder.

    "Each retrieved triple (h, r, t) is verbalized as 'h r t' and encoded using the
    encoder. The pooled representation is projected to the common d-dimensional space to
    obtain g_kq."

    Verbalizations are deduplicated per forward pass and encoded once: a batch of 8
    instances asks for ~120 triples, of which many repeat, and each is only a handful of
    tokens. An LRU cache of tokenized ids avoids re-tokenizing the same string every step.
    """

    def __init__(self, text_encoder, tokenizer, d: int, max_len: int = 24,
                 dropout: float = None, freeze_backbone: bool = False):
        super().__init__()
        self.enc = text_encoder            # SHARED with the student, not a copy
        self.tok = tokenizer
        self.max_len = max_len
        # freeze_backbone=False (default): L_kg backpropagates through the triple
        # representations into the shared text encoder, so the encoder actually learns to
        # represent KG evidence. With BERTweet-base this fits on a 16 GB card. Setting it
        # True detaches the KG branch — an engineering compromise that changes the model
        # (the encoder then receives no gradient from L_kg), kept only as an OOM escape.
        self.freeze_backbone = freeze_backbone
        dropout = CONFIG.dropout if dropout is None else dropout
        self.proj = nn.Sequential(nn.Linear(text_encoder.out_dim, d), nn.GELU(),
                                  nn.Dropout(dropout))
        self._ids_cache = {}

    def _encode_ids(self, text: str):
        ids = self._ids_cache.get(text)
        if ids is None:
            ids = self.tok(text, truncation=True, max_length=self.max_len)["input_ids"]
            if len(self._ids_cache) < 200_000:
                self._ids_cache[text] = ids
        return ids

    def forward(self, triples: Sequence[Triple]) -> torch.Tensor:
        device = self.proj[0].weight.device
        d = self.proj[0].out_features
        if not triples:
            return torch.zeros((0, d), device=device)
        texts = [verbalize(t) for t in triples]
        uniq = list(dict.fromkeys(texts))
        idx = {u: i for i, u in enumerate(uniq)}
        seqs = [self._encode_ids(u) for u in uniq]
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
        att = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, device=device)
            att[i, :len(s)] = 1
        if self.freeze_backbone:
            with torch.no_grad():
                h = self.enc.bert(input_ids=ids, attention_mask=att).last_hidden_state
        else:
            h = self.enc.bert(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)          # mean-pool
        g = self.proj(pooled)                                     # [U, d]
        return g[torch.tensor([idx[t] for t in texts], device=device)]


class TripleEncoder(nn.Module):
    """phi([e_p ; r ; e_q]) -> g_kq in R^d  (Eq. 14) — Numberbatch entity-embedding form.

    Kept as the cheap fallback; the paper's own formulation is VerbalizedTripleEncoder.
    """

    def __init__(self, d: int = None, entity_dim: int = None, embedder: Optional[EntityEmbedder] = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        entity_dim = entity_dim or CONFIG.entity_emb_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.embedder = embedder or EntityEmbedder(dim=entity_dim)
        self.entity_proj = nn.Linear(entity_dim, d)
        self.relation_emb = nn.Embedding(len(RELATION_VOCAB), d)
        self.phi = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d)
        )

    def forward(self, triples: Sequence[Triple]) -> torch.Tensor:
        """Returns g [M, d]; empty [0, d] if no triples."""
        device = self.entity_proj.weight.device
        d = self.entity_proj.out_features
        if not triples:
            return torch.zeros((0, d), device=device)
        head_e, tail_e, rel_ids = [], [], []
        for t in triples:
            head_e.append(self.embedder.embed(t.head))
            tail_e.append(self.embedder.embed(t.tail))
            rel_ids.append(REL2ID.get(t.relation, 0))
        he = torch.from_numpy(np.stack(head_e)).to(device)
        te = torch.from_numpy(np.stack(tail_e)).to(device)
        ep = self.entity_proj(he)                                  # [M, d]
        eq = self.entity_proj(te)                                  # [M, d]
        r = self.relation_emb(torch.tensor(rel_ids, device=device))  # [M, d]
        g = self.phi(torch.cat([ep, r, eq], dim=-1))              # [M, d]
        return g
