"""Offline Llama-3-8B-Instruct teacher, 8-bit on A100; training labels only."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import CONFIG
from kg import Triple
from kg_retrieval import _triple_key as triple_key

RELEVANCE_PROMPT = (
    "Given a tweet, an aspect term, and an image description, decide whether the image "
    "provides useful evidence for inferring sentiment toward the aspect. "
    "Return 1 if useful and 0 otherwise."
)
KG_PROMPT = (
    "Given a tweet, an aspect term, and a candidate KG triple, decide whether the triple "
    "is useful for aspect-level sentiment reasoning. Return 1 if useful and 0 otherwise."
)
# Graded scoring below is experimental; the paper uses the binary prompts.
KG_SCORE_PROMPT = (
    "Given a tweet, an aspect term, and a candidate KG triple, rate how useful the triple is "
    "for reasoning about sentiment toward the aspect, on a scale from 0 (useless) to 10 "
    "(highly useful). Answer with a single integer."
)
_BIN_RE = re.compile(r"[01]")
_INT_RE = re.compile(r"\d+")


def _parse_binary(text: str) -> int:
    m = _BIN_RE.search(text or "")
    if not re.fullmatch(r"[01][.!]?", text.strip()):
        raise ValueError(f"Teacher did not return a binary label: {text!r}")
    return int(text.strip()[0])


def _parse_int10(text: str) -> int:
    m = _INT_RE.search(text or "")
    return max(0, min(10, int(m.group(0)))) if m else 0


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def cache_identity(dataset, cfg=CONFIG):
    """Invalidate old checkpoint/precision labels and changed training/KG inputs."""
    import hashlib
    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                result.update(chunk)
        return result.hexdigest()
    train_file = cfg.paths.data / dataset / "train.tsv"
    if not train_file.exists():
        train_file = train_file.with_suffix(".txt")
    return dict(
        schema=1, teacher=cfg.teacher_llm_id, precision=cfg.teacher_precision,
        captioner=cfg.captioner_id, prompts=[RELEVANCE_PROMPT, KG_PROMPT],
        top_m=cfg.top_m_triples, top_l=cfg.top_l_concepts, opinion_window=cfg.opinion_window,
        rank_weights=list(cfg.kg_rank_weights), sources=list(cfg.kg_sources),
        training_sha256=digest(train_file),
        kg_sha256=digest(cfg.paths.kg_index / "kg.sqlite"),
        concepts_sha256=digest(cfg.paths.data / "visual_concepts" / f"{dataset}.json"))


class TeacherCache:
    """In-memory lookup over the cached teacher labels for one dataset."""

    def __init__(self, rel: Dict[Tuple[str, int], float] = None,
                 kg: Dict[Tuple[str, int, str], float] = None,
                 pds: Dict[Tuple[str, int], List[float]] = None):
        self.rel = rel or {}
        self.kg = kg or {}
        # NOVEL — evidence-effect direction, a 3-vector over {NEG-shift, no-shift, POS-shift}
        self.pds = pds or {}

    @classmethod
    def load(cls, dataset: str, cfg=CONFIG) -> "TeacherCache":
        import pandas as pd
        import json

        base = cfg.paths.teacher_labels
        metadata = base / f"{dataset}_metadata.json"
        if not metadata.exists():
            raise FileNotFoundError(f"Missing teacher provenance: {metadata}; regenerate with prepare.py teacher.")
        if json.loads(metadata.read_text(encoding="utf-8")) != cache_identity(dataset, cfg):
            raise ValueError("Teacher cache provenance differs from this configuration/data. Regenerate it.")
        rel, kg = {}, {}
        rp = base / f"{dataset}_relevance.parquet"
        kp = base / f"{dataset}_kg.parquet"
        if rp.exists():
            df = pd.read_parquet(rp)
            for _, r in df.iterrows():
                rel[(str(r["instance_id"]), int(r["aspect_idx"]))] = float(r["label"])
        if kp.exists():
            df = pd.read_parquet(kp)
            for _, r in df.iterrows():
                kg[(str(r["instance_id"]), int(r["aspect_idx"]), str(r["triple_key"]))] = float(r["label"])
        pds = {}
        pp = base / f"{dataset}_pds.parquet"
        if pp.exists():
            df = pd.read_parquet(pp)
            for _, r in df.iterrows():
                pds[(str(r["instance_id"]), int(r["aspect_idx"]))] = [
                    float(r["p_neg"]), float(r["p_none"]), float(r["p_pos"])]
        return cls(rel, kg, pds)

    @staticmethod
    def save(dataset: str, rel_rows: List[dict], kg_rows: List[dict],
             pds_rows: List[dict] = None, cfg=CONFIG) -> None:
        import pandas as pd

        base = cfg.paths.teacher_labels
        base.mkdir(parents=True, exist_ok=True)
        if rel_rows:
            pd.DataFrame(rel_rows).to_parquet(base / f"{dataset}_relevance.parquet", index=False)
        if kg_rows:
            pd.DataFrame(kg_rows).to_parquet(base / f"{dataset}_kg.parquet", index=False)
        if pds_rows:
            pd.DataFrame(pds_rows).to_parquet(base / f"{dataset}_pds.parquet", index=False)
        from utils import save_json
        save_json(cache_identity(dataset, cfg), base / f"{dataset}_metadata.json")


# --------------------------------------------------------------------------- #
# LLM teacher
# --------------------------------------------------------------------------- #
class LLMTeacher:
    def __init__(self, model_id: str = None, device: str = None, cfg=CONFIG):
        self.cfg = cfg
        self.model_id = model_id or cfg.teacher_llm_id
        self.device = device or cfg.device
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not str(self.device).startswith("cuda"):
            raise ValueError("Offline teacher requires CUDA for the 8-bit protocol.")
        from transformers import BitsAndBytesConfig
        kwargs = dict(token=self.cfg.hf_token,
                      quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                      device_map={"": self.device})
        self._tok = AutoTokenizer.from_pretrained(self.model_id, token=self.cfg.hf_token)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()
        # We decode greedily (do_sample=False). Clear the model's default sampling params
        # so transformers doesn't warn about temperature/top_p/top_k on every call.
        gc = self._model.generation_config
        gc.do_sample = False
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None
        if gc.pad_token_id is None and self._tok.pad_token_id is not None:
            gc.pad_token_id = self._tok.pad_token_id

    def _render(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{system}\n\n{user}\nAnswer:"

    def _ask(self, system: str, user: str) -> int:
        import torch

        self._load()
        text = self._render(system, user)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=4, do_sample=False)
        gen = self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _parse_binary(gen)

    def _ask_batch(self, system: str, users: List[str], max_new_tokens: int = 4, parser=None) -> List[int]:
        """P1 (OBEYING, result-neutral): score many INDEPENDENT prompts in one greedy
        generate() via left-padding + attention mask. Per-sequence argmax is identical to
        the unbatched call (padding is masked; left-padding right-aligns real tokens), so
        the {0,1} labels match — this is pure GPU throughput, not a methodology change.
        Triples are scored independently (no joint-prompt collapse) to preserve semantics.
        """
        import torch

        if not users:
            return []
        self._load()
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
            if self._model.generation_config.pad_token_id is None:
                self._model.generation_config.pad_token_id = self._tok.eos_token_id
        prev_side = self._tok.padding_side
        self._tok.padding_side = "left"  # decoder-only models must left-pad for batched generation
        try:
            texts = [self._render(system, u) for u in users]
            enc = self._tok(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self._tok.pad_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1]:]
            decoded = self._tok.batch_decode(gen, skip_special_tokens=True)
        finally:
            self._tok.padding_side = prev_side
        parse = parser or _parse_binary
        return [parse(d) for d in decoded]

    def relevance_label(self, tweet: str, aspect: str, image_description: str) -> int:
        return self._ask(RELEVANCE_PROMPT, self._rel_user(tweet, aspect, image_description))

    def kg_label(self, tweet: str, aspect: str, triple: Triple) -> int:
        return self._ask(KG_PROMPT, self._kg_user(tweet, aspect, triple))

    @staticmethod
    def _rel_user(tweet: str, aspect: str, image_description: str) -> str:
        return f"Tweet: {tweet}\nAspect: {aspect}\nImage description: {image_description}\nAnswer (0 or 1):"

    @staticmethod
    def _kg_user(tweet: str, aspect: str, triple: Triple) -> str:
        tr = f"({triple.head}, {triple.relation}, {triple.tail})"
        return f"Tweet: {tweet}\nAspect: {aspect}\nKG triple: {tr}\nAnswer (0 or 1):"

    @staticmethod
    def _kg_user_graded(tweet: str, aspect: str, triple: Triple) -> str:
        """Graded variant. The binary user prompt above ends with "Answer (0 or 1):",
        which CONTRADICTS the 0-10 system prompt — the teacher obeys the nearer
        instruction and answers 0/1, so every score collapses to 0.0 or 0.1 and nothing
        is ever retained. Measured: 0 of 31,790 triples scored >= 0.5.
        """
        tr = f"({triple.head}, {triple.relation}, {triple.tail})"
        return (f"Tweet: {tweet}\nAspect: {aspect}\nKG triple: {tr}\n"
                f"Usefulness (an integer from 0 to 10):")

    def relevance_label_batch(self, items: List[Tuple[str, str, str]], batch_size: int = 16) -> List[int]:
        users = [self._rel_user(tw, asp, desc) for (tw, asp, desc) in items]
        return self._run_batched(RELEVANCE_PROMPT, users, batch_size)

    def kg_label_batch(self, items: List[Tuple[str, str, Triple]], batch_size: int = 16) -> List[int]:
        users = [self._kg_user(tw, asp, tr) for (tw, asp, tr) in items]
        return self._run_batched(KG_PROMPT, users, batch_size)

    def kg_score_batch(self, items: List[Tuple[str, str, Triple]], batch_size: int = 16) -> List[int]:
        """Graded (0-10) usefulness scores for KG triples (see KG_SCORE_PROMPT)."""
        users = [self._kg_user_graded(tw, asp, tr) for (tw, asp, tr) in items]
        return self._run_batched(KG_SCORE_PROMPT, users, batch_size, parser=_parse_int10)

    def _run_batched(self, system: str, users: List[str], batch_size: int, parser=None) -> List[int]:
        out: List[int] = []
        for i in range(0, len(users), batch_size):
            out.extend(self._ask_batch(system, users[i: i + batch_size], parser=parser))
        return out


# --------------------------------------------------------------------------- #
# align cached labels to a live batch (consumed by losses.compute_losses)
# --------------------------------------------------------------------------- #
def build_targets(batch: Dict, outputs: Dict, cache: Optional[TeacherCache] = None, cfg=CONFIG) -> Dict:
    import torch
    from anchors import match_anchors
    from config import POL2ID
    device = outputs["tag_logits"].device
    matches = [match_anchors(pred, gold, cfg.anchor_iou_threshold)
               for pred, gold in zip(outputs["aspect_word_spans"], batch["gold_aspects"])]
    order = [(b, k, g) for b, row in enumerate(matches) for k, g in enumerate(row)]
    targets = dict(bio_labels=batch["bio_labels"], anchor_labels=batch["anchor_labels"])
    targets["aspect_polarity"] = torch.tensor([
        POL2ID[batch["gold_aspects"][b][g][2]] if g >= 0 else -100 for b, k, g in order
    ], dtype=torch.long, device=device)
    if cache is None:
        return targets
    rvals, rmask, kg_t, kg_m = [], [], [], []
    for a, (b, k, g) in enumerate(order):
        iid = str(batch["instance_id"][b])
        v = cache.rel.get((iid, g)) if g >= 0 else None
        rvals.append(v if v is not None else 0.0)
        rmask.append(v is not None)
        tv, tm = [], []
        for tr in outputs["kg_triples"][a]:
            v = cache.kg.get((iid, g, triple_key(tr))) if g >= 0 else None
            tv.append(v if v is not None else 0.0)
            tm.append(v is not None)
        kg_t.append(torch.tensor(tv, dtype=torch.float, device=device))
        kg_m.append(torch.tensor(tm, dtype=torch.bool, device=device))
    targets.update(
        teacher_relevance=torch.tensor(rvals, dtype=torch.float, device=device),
        teacher_relevance_mask=torch.tensor(rmask, dtype=torch.bool, device=device),
        teacher_kg=kg_t, teacher_kg_mask=kg_m)
    return targets

