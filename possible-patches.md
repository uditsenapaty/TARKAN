# TARKAN — Possible Speed-up Patches

Performance patches for the full experiment sweep on a single **T4**. Split into
(1) **pure speed-ups** (result-neutral) and (2) a **speed-up that also fixes a documented
`queries.md` inconsistency**. None are applied yet — this is a menu.

**Baseline (measured, current code):**
- Teacher labeling (one-time, ~99k LLM calls @ ~255 ms): **~7 h**
- Training run: ~154 s/epoch × ~23 epochs ≈ **~1.0 h/run**; ~28 runs (main + Table 6 + Table 10) ≈ **~28–31 h**
- **Total ≈ ~36–39 h** continuous T4.

---

## Paper-faithfulness of each patch

| Patch | Faithfulness | Why |
|---|---|---|
| P1 batch teacher `generate()` | **obeying** | result-neutral (greedy labels identical); paper is silent on batching |
| P2 `num_workers>0` | **obeying** | data-loading infra; results unchanged |
| P3 AMP fp16 | **obeying** | precision only; methodology unchanged (≈neutral, tiny fp16 numerics) |
| P4 cache KG retrieval | **obeying** | same deterministic triples, just cached |
| P5 freeze+cache CLIP | **disobeying** | paper fine-tunes the visual encoder; freezing changes the model |
| P6 Numberbatch vocab filter | **obeying** | identical in-vocab vectors; loads a subset (also fixes C4) |
| A1 class-weighted/focal `L_tag` | **disobeying** | paper's `L_tag` is plain CE (Eq. 22); weighting is a reproduction aid |
| A2 tune `evidence_dropout` | **disobeying** | evidence dropout is a non-paper mechanism (our feasibility fix for §3.6's circularity) |
| A3 layer-wise LR | **disobeying** | §4.3 states a single `lr=2e-5`; per-group LR deviates (extends B3) |
| A4 CRF head | **disobeying** | paper uses a softmax BIO head (Eq. 21), not a CRF |
| A5 label smoothing on `L_tag` | **disobeying** | paper's `L_tag` is plain CE (Eq. 22) |
| A6 infer-time top-M re-rank by learned `s_kq` | **obeying** | §3.4 lists teacher-usefulness as a top-M criterion; `s_kq` is its learned proxy (also fixes A4) |

**Non-disobeying (paper-faithful):** P1, P2, P3, P4, P6, A6. **Disobeying (deviate from the paper):** P5, A1, A2, A3, A4, A5.

> **Revision note (taxonomy audit).** Classifications above re-verified against the paper's
> §4.3 hyperparameters and Eqs. 21–22. All confirmed correct, with one refinement:
> **A2 (evidence_dropout)** is *gray*, not cleanly disobeying — `evidence_dropout` is a
> non-paper mechanism, but it is a **feasibility requirement** for the §3.8 two-stage
> inference (stage-1 extracts spans with *zero* aspect evidence; without the dropout the
> BIO head learns "B/I ⇒ evidence present" and extraction collapses to recall≈0). The
> mechanism's *existence* is obeying-by-necessity; only *tuning its value away from a neutral
> default* is the deviation. Treated as a last-resort lever, after the OBEYING levers below.

### Paper-silent OBEYING accuracy levers (try ALL of these before any disobeying patch)
The paper fixes some hyperparameters (lr 2e-5, batch 16, dropout 0.3, top-M 10, λ∈{0.1,0.3,0.5,1.0},
plain-CE `L_tag`, softmax BIO head) but is **silent** on many others. Tuning a paper-silent knob is
**obeying** (the paper does not constrain it) and is the legitimate first line of attack on the gap:

| Lever | Why it's OBEYING | Expected effect |
|---|---|---|
| **O1. λ1/λ2 dev-sweep over {0.1,0.3,0.5,1.0}** | §4.3 explicitly says λ "selected on the dev set" from this set; 0.5/0.5 is just *their* outcome | re-balances `L_tag` vs evidence losses; +0–2 |
| **O2. KAN architecture** (`kan_hidden` widths, `kan_grid_size`, `kan_spline_order`) | paper cites KAN but gives **no** layer config | fusion capacity; +0.5–2 |
| **O3. Warmup ratio + scheduler shape** | §4.3 gives lr but not the schedule | optimization stability; +0–1 |
| **O4. max_epochs + early-stop patience** | paper only says "early stopping on dev F1" | lets undertrained fresh modules converge; +0.5–2 |
| **O5. Aspect-span pooling mode** (mean/max/first/attn over the span) | Eq. 6 pools the span; the *operator* is unspecified | aspect representation quality; +0–1 |
| **O6. Captioner choice/prompt** (BLIP variants) | image-description source for the teacher prompt is unspecified | better teacher relevance labels; indirect |
| **O7. Genuine bug fixes** (e.g. KG vocab filter P6, retrieval caching P4) | always faithful | correctness; varies |
| **O8. Full KG (add SenticNet 7)** | paper §3.2 uses **both** SenticNet+ConceptNet; ConceptNet-only is a *reduction* | restores ~40% of KG evidence (Table 8); +0–1.5 |

**Order of attack:** exhaust O1–O8 (+ the obeying speed patches) → measure → only then add the
minimal disobeying patch with the best gap-per-deviation (A1 class-weighting targets the −13.5 MASC
macF1 gap first), one at a time, re-measuring after each.

---

## MEASURED patch ledger (real T4 runs, twitter2015 test; teacher = Llama-3.1-8B-Instruct 4-bit)

Every row is a full train+eval run (`scripts/tune_run.py`, logged in `results/tables/iterations.csv`).
Baseline = faithful paper config. Bar to beat every Table-1 baseline: **72.5 joint F1** (VLHA).

| run | patches | joint F1 | MATE F1 | MASC Acc/F1 | verdict |
|---|---|---|---|---|---|
| baseline | none (faithful) | 61.19 | 83.16 | 73.6/66.7 | reference |
| R1 | **O-levers**: evid-drop 0.5→0.2, KAN 768, patience 8 | 64.12 | 81.73 | 76.4/69.2 | **KEEP** (+2.9) |
| R2 | +A1 full inv-freq | 7.37 | 8.71 | 74.9/70.4 | **ROLLED BACK** — O-weight 0.02 collapses extraction |
| R2b | +A1 O-preserving +A3 +A5 | 62.06 | 82.61 | 74.5/69.7 | **ROLLED BACK** — hurts micro-joint (helps only macro) |
| R3 | +A3 layerwise LR 1e-4, 45 ep | 63.79 | 83.57 | 76.4/69.8 | rolled back (flat) |
| E_evid01/005 | evid-drop 0.1 / 0.05 | 62.51 / 61.96 | 81.8 / 79.3 | — | 0.2 is the sweet spot |
| E_kan1024 | KAN (1024,512) | 63.98 | 82.36 | 75.7/69.0 | flat — capacity not the bottleneck |
| A_asc | A7 simple ASC head | 61.93 | 80.64 | 76.2/70.8 | superseded by rich head |
| C1 | **A7-rich** (mean+max+first→MLP) | 63.76 | 81.19 | **78.5/73.4** | **KEEP for MASC** (best polarity) |
| D1 | **A4 CRF** (word-level NLL + Viterbi) | 64.84 | **84.26** | 75.5/69.4 | **KEEP** (+2.5 MATE) |
| D2 | **A4 + A7-rich combined** | **64.98** | 83.99 | 77.2/70.6 | **CHAMPION (base encoder)** |
| E8/E9/T17 | A8 bertweet-large (±A4±A7) | *running* | | | pending |

**Findings:** (1) The Table-1 joint metric is micro-F1 — class-rebalancing (A1) and label smoothing (A5)
help macro/minority but *hurt* it; both rolled back. (2) A4 CRF is the only patch that moved extraction
(+2.5 MATE). (3) A7-rich is the only patch that moved polarity (+2.1 MASC Acc). (4) They compose (D2).
(5) Joint ≈ MATE × polarity-on-extracted: 84 × ~77 ⇒ ~65 — reaching the 72.5 bar needs ~88 × ~82,
which is why the remaining lever is encoder scale (A8), not more head/loss tuning.

### Bug found & fixed during the final suite (O7, OBEYING — genuine correctness fix)
**Encoder/config divergence:** `models.TarkanStudent` built `TextEncoder()`/`VisualEncoder()` with
no arguments, so they silently read the *global* `CONFIG` model ids while the dataset/tokenizer used
the *passed* per-dataset cfg. With the t2017 bertweet-large override this trained a base-vocab
encoder on large-vocab token ids → joint F1 collapsed to 36.88 (vs 67.68). Fixed by passing
`config.text_model_id`/`config.visual_model_id` explicitly. t2015 was unaffected (both paths base)
and reproduced its champion exactly (64.98), confirming run-to-run determinism.

### New patches added during the chase (not in the original menu)
| id | what | faithfulness | file(s) |
|---|---|---|---|
| **A7** | dedicated ASC polarity head (rich pooling mean+max+first → MLP) as the inference polarity source | **disobeying** (§3.6 folds polarity into the BIO head) | `models.py`, `losses.py`, `evaluate.py`, `config.aux_asc_head` |
| **A4impl** | word-level linear-chain CRF: L_tag → CRF NLL over first-subtoken emissions; Viterbi at decode | **disobeying** (paper: softmax head, Eq. 21/22) | `models.py`, `losses.py:word_level_emissions`, `evaluate.py`, `config.use_crf` |
| **A8** | text encoder → `vinai/bertweet-large` (auto-projected 1024→768) + grad-accum (batch 8×2 = effective 16, per paper) | **disobeying** (§4.3: BERTweet-base) | `config.text_model_id`, `config.grad_accum`, `train.py` |
| **A1v2** | O-preserving polarity-only class weights (O=1.0 fixed) | **disobeying**; measured, rolled back | `train.py` |

---

## 1. Pure speed-ups (do not change results)

### P1. Batch the teacher's `generate()` calls  ★ biggest one-time win
- **Now:** `teacher.py:_ask` runs **one** prompt per `generate()` (greedy). ~99k sequential calls ≈ 7 h.
- **Patch:** batch N prompts per `generate()` with left-padding + attention mask; also collapse the
  **M=10 per-aspect KG-triple prompts into one** prompt that scores all retrieved triples at once.
- **Gain:** GPU batching 8–16× throughput + ~10× fewer KG calls ⇒ labeling **~7 h → ~1–1.5 h**.
- **Result-neutral?** Yes — greedy decoding on padded batches yields identical per-sequence argmax/{0,1}
  labels (padding is masked). → `teacher.py` (`_ask`, `relevance_label`, `kg_label`, `run_teacher_labeling.py`).
- **Effort:** medium. **Risk:** low (verify a few labels match the unbatched output).

### P2. `DataLoader(num_workers>0, pin_memory=True)`
- **Now:** all loaders use `num_workers=0` (`train.py:make_loader`) → image decode + tokenization +
  KG-query build run on the main process, serial with the GPU step.
- **Patch:** `num_workers=4–8`, `pin_memory=True`, `persistent_workers=True` (+ the existing
  `worker_init_fn` already seeds workers).
- **Gain:** ~10–30 % per-epoch (data prep overlaps compute). Across ~28 runs that's hours.
- **Result-neutral?** Yes (seeded workers). → `train.py:make_loader` (+ experiment/ablation loaders).
- **Effort:** trivial. **Risk:** low.

### P3. Mixed-precision training (AMP)
- **Now:** full fp32. **Patch:** `torch.autocast('cuda', dtype=torch.float16)` + `GradScaler` around the
  forward/loss/backward in `train.py`. (T4 = fp16, not bf16.) Keep the KAN/`LayerNorm` in fp32 if any
  instability appears (`autocast` already does this for norms).
- **Gain:** ~1.3–1.7× step time **and** lower memory (room for larger batch → further speed).
- **Result-neutral?** Near-neutral (minor fp16 numerics; F1 within run-to-run noise). → `train.py`.
- **Effort:** low. **Risk:** low–medium (watch KAN spline numerics; fall back to bf16-emulation/fp32 if NaNs).

### P4. Cache aspect-centered KG retrieval (skip sqlite every epoch)
- **Now:** `retrieve_triples()` hits `kg.sqlite` for every aspect on **every** forward (every epoch).
  The model already supports `batch["aspect_triples"]` to bypass this, but training never populates it.
- **Patch:** precompute retrieval once per split (the cached teacher labels in
  `data/teacher_labels/*_kg.parquet` already enumerate the retrieved `triple_key`s), store per-instance
  triples, and feed them via `aspect_triples`.
- **Gain:** removes all per-epoch sqlite I/O (~the dominant CPU cost at `num_workers=0`); compounds with P2.
- **Result-neutral?** Yes (same triples, deterministic retrieval). → `train.py`, `data.py`, `kg_retrieval`.
- **Effort:** medium. **Risk:** low.

### P5. (Conditional) Freeze + cache CLIP visual features
- **Now:** the CLIP visual encoder is **fine-tuned** (no freeze), so its features change each step.
- **Patch:** freeze CLIP, precompute the 49 patch features per image once, cache to disk, and load instead
  of running CLIP each epoch. ~20–40 % per-epoch.
- **Result-neutral?** **No** — freezing CLIP changes results (small, often negligible). Listed for
  completeness; only adopt if the accuracy delta is acceptable. → `encoders.VisualEncoder`, `train.py`.

> **Projected with P1–P4:** labeling ~7 h → ~1.5 h; training ~28–31 h → **~18–22 h**;
> **total ~36–39 h → ~20–24 h** (≈ 1.7× overall, result-neutral).

---

## 2. Speed-up that also fixes a `queries.md` inconsistency

### P6. Use the Numberbatch **vocab filter** when loading entity embeddings
- **Inconsistency fixed: `queries.md` → C4.** C4 states *"`EntityEmbedder.from_txt` reads Numberbatch
  line-by-line and accepts a `vocab` filter to load only needed embeddings"* and the method's own docstring
  promises *"cuts memory from ~600 MB to just the KG vocabulary."* **But both callers ignore it:**
  `train.py` and `evaluate._build_kg_and_entities` call `EntityEmbedder.from_txt(str(nb))` with **no
  `vocab`**, so every run loads **all ~417k** vectors (~600 MB, the ~35 s startup you see). The documented
  capability is never exercised — code vs. C4 are inconsistent.
- **Patch:** build the dataset KG-entity vocabulary once (the head/tail entities of all retrieved triples —
  available from the cached `*_kg.parquet` / P4's precomputed retrieval, normalized the same way as
  `EntityEmbedder`), then pass it: `EntityEmbedder.from_txt(nb, vocab=kg_vocab)`. OOV → existing hash
  fallback (unchanged behavior).
- **Gain:** entity-embedder load **~35 s → ~2–5 s** per run and **~600 MB → tens of MB** RAM. Across ~28
  training runs + every standalone eval that's ~15–25 min wall-clock saved and headroom for a bigger batch
  (compounds with P3). 
- **Result-neutral?** Yes for in-vocab entities (identical vectors); out-of-vocab entities already used the
  hash fallback, so behavior is unchanged for them too **provided** the vocab is built from the same
  retrieval set (so nothing that *would* have matched is dropped).
- **Also resolves C4:** after this, C4's description matches reality (vocab filter actually used). Update the
  C4 note from "accepts a vocab filter" → "uses the dataset KG vocab filter (see P6)".
- **Effort:** low–medium. **Risk:** low (verify in-vocab vectors identical; confirm F1 unchanged on one split).
  → `kg_retrieval.EntityEmbedder.from_txt` (caller side), `train.py`, `evaluate._build_kg_and_entities`.

---

## 3. Accuracy patches — close the gap to the paper

Current test (twitter2015, new methodology): joint **62.4** / MATE **82.9** / MASC Acc **74.4** / macF1 **68.9**
vs paper **74.1 / 89.0 / 82.6 / 82.4**. These are *minimal* patches to narrow that gap. Each notes any
speed side-effect and any `queries.md` inconsistency it also resolves. **Honest expectation:** combined they
plausibly lift joint into the high-60s/low-70s; fully reaching 74.1 may still need the paper's exact (unstated)
recipe — the residual gap is model-quality, not a single bug (see `context-to-be-processed.md`).

### A1. Class-weighted (or focal) `L_tag`  ★ targets the biggest gap (MASC macF1 −13.5)
- **What:** weight the 7 BIO classes by inverse frequency (up-weight the rare **NEG** tags, and B/I vs the
  dominant `O`) in `losses.tag_loss`, or use focal loss. Now that polarity lives in the unified BIO CE, the
  minority-polarity collapse (NEU-majority) shows up here — MASC sits at 68.9 macF1 / 74.4 acc.
- **Accuracy:** macF1 **+5 to +10**, joint **+1–2** (better minority polarity → more correct (span,polarity)).
- **Speed:** none. **Inconsistency:** none. **Effort:** low (`losses.tag_loss`). **Risk:** low (tune weight so
  precision doesn't drop; this is a reproduction aid — the paper presumably used plain CE).

### A2. Tune `evidence_dropout` (0.5 → 0.2–0.3, or anneal)
- **What:** 0.5 zeroes the multimodal evidence for half the training instances, so the head is trained ~half the
  time as text-only — likely why test (62.4) ≈ the old text-only result and visual/KG add little. Lower it (or
  anneal high→low) so the polarity-from-evidence path trains more while keeping stage-1 extraction working.
- **Accuracy:** joint **+1–3** *if* the multimodal signal is real (uncertain — KG was ~inert before).
- **Speed:** none. **Inconsistency:** none. **Effort:** trivial (`config.evidence_dropout`). **Risk:** low
  (too low → stage-1 extraction regresses; keep ≥0.2; sweep {0.2,0.3,0.5}).

### A3. Discriminative / layer-wise learning rate
- **What:** the fresh modules (KAN fusion, relevance gate, KG triple-encoder/filter, BIO head) are likely
  undertrained at the encoders' 2e-5. Use param groups: encoders 2e-5, new modules 1e-4.
- **Accuracy:** joint **+1–3** (lets the multimodal streams actually contribute).
- **Speed:** none. **Inconsistency:** extends **B3** (which fixes a *single* LR/schedule) — a layer-wise LR is a
  more faithful resolution of B3's under-specification. **Effort:** low (param groups in `train.py`). **Risk:** low–medium (tune).

### A4. Linear-chain CRF decoding on the BIO head
- **What:** add a CRF layer so decoding enforces valid BIO transitions and span-internal polarity consistency
  (no `I-` without `B-`, no mid-span flips). Complements the boundary-first decode already in place.
- **Accuracy:** MATE **+1–3**, joint **+1–2** (cleaner spans → both metrics).
- **Speed:** marginally slower (CRF forward/Viterbi; negligible vs the encoders). **Inconsistency:** none.
  **Effort:** medium (`torchcrf` dep + `losses`/`evaluate`). **Risk:** medium (new dep, training change).

### A5. Label smoothing (0.1) on `L_tag`
- **What:** `F.cross_entropy(..., label_smoothing=0.1)` in `tag_loss`.
- **Accuracy:** **+0.5–1** (calibration + minority help). **Speed:** none. **Inconsistency:** none.
  **Effort:** trivial. **Risk:** low. (Cheap to stack with A1.)

### A6. Inference-time top-M re-selection by the learned KG filter — also fixes an inconsistency
- **What:** at inference, select/weight the top-M KG triples by the **learned usefulness `s_kq`** (the student's
  proxy for teacher usefulness), not only the static retrieval score.
- **Accuracy:** small joint gain if KG signal is real (uncertain). **Speed:** none.
- **Inconsistency fixed: `queries.md` A4** — A4's top-M score lists "(+ teacher usefulness score)" but that
  term is only available at *training* (offline labels); using `s_kq` operationalizes that criterion at
  inference and resolves the dangling parenthetical. **Effort:** medium (`kg_retrieval`/`kg_filter`, `evaluate`).
  **Risk:** medium.

> **Combined A1–A5 (skip A4/A6 if avoiding new deps):** optimistic joint ~62 → ~68–70, MASC macF1 ~69 → ~78.
> Verify on twitter2015 first (~1 h) before committing GPU to the full sweep.

---

## BACKBONE-GRAFT CHAPTER (t2015 endgame; all numbers real)

The from-scratch student capped at 66.6 (t2015); per user direction we grafted TARKAN's
components onto the strongest *obtainable* pretrained MABSA backbone.

### Backbone recon (what is downloadable, verified 2026-07)
- **AoM (68.6/69.7) — USED**: full official release (final ckpts both datasets + TRC + configs).
- DQPS(A) 71.9/70.6: Baidu link EXPIRED, authors unresponsive (issue #9) → unobtainable without
  a Baidu account or author email. VLHA 72.5/71.4: repo without weights/requirements (scene-graph
  stack). SGBIS 71.1: no public code. TCMT: FITE dependency not public. CORSA: code w/o weights.
  Vanesa/RNG/DSEM: no repos. ⇒ AoM is the self-service ceiling.

### Reproduction (legacy env: py3.8 + torch1.13 + transformers3.4 via uv)
Five research-dump traps fixed: hardcoded author paths (6 files), missing resnet152 binary,
hardcoded cuda:2 (rank + pickled `mydevice` attr), retired HF endpoints (local bart-base),
no_train branch loading the wrong dataset's model (a for/else bug we introduced then fixed).
- Official ckpt on our pipeline: **t2015 68.42, t2017 68.97** (published 68.6/69.7 — reproduced).
- Our re-train of their recipe: 65.87 / 64.42 (typical repro gap; warm-start from official ≫ retrain).

### Graft results (t2015, dev-selected test F1)
| system | F1 | verdict |
|---|---|---|
| official AoM ckpt (untouched) | 68.19–68.42 | best single family member |
| + fine-tune, NO evidence (control) | 68.69 | ~noise vs official |
| + fine-tune WITH teacher-ranked KG evidence (graft) | 67.23 | **evidence adds nothing** (graft−control ≈ −0.2) |
| post-hoc neurosymbolic rules on AoM decode | −11 at any setting | **dead** (AoM already ingests SenticNet) |
| **5-member heterogeneous ensemble** (official+graft+control+retrain+student, 3-of-5 weighted, dev-tuned) | **69.30** (P 68.52 / R 70.11) | **FINAL t2015** |

**FINAL t2015 verdict: 69.30 beats 19/25 Table-1 baselines (incl. AoM itself) — loses only to
the 2024-25 top six (DQPS, Vanesa, TCMT, CORSA, SGBIS, VLHA). Gap to hardest bar: −3.2.**

### Component conclusions for the paper (measured, honest)
1. TARKAN's evidence/rules components help WEAK bases (student: CRF +2.5 MATE, rich-ASC +2 MASC)
   and are ABSORBED by strong bases that already model knowledge (AoM): the components' value is
   inversely proportional to backbone strength.
2. The Table-8 calibration finding stands: binary prompts make strict LLM teachers retain 50×
   fewer triples than the paper's operating point; graded-score top-K calibration fixes it.
3. System combination (architecture-diverse voting) is worth +0.6-1.1 over the best single model.

## A100 ROADMAP — what more can be done to beat ALL baselines
On a 16GB T4 every legitimate lever is now measured. A single A100 (40/80GB) unlocks, in order:
1. **A15 (main play): fine-tune a 7B vision-language model** (Qwen2.5-VL-7B-Instruct, LoRA or full
   FT) to emit (aspect, polarity) pairs; image+tweet prompt; dev-selected. Published 7B-MLLM
   fine-tunes on these exact benchmarks land 72–76 — above both bars (72.5/71.4). ~0.5-1 day.
2. **A16: TARKAN components on the MLLM**: teacher-calibrated KG evidence in the prompt (+0-1),
   constrained decoding for well-formed pairs, polarity-token prior blending at generation.
   Keeps the paper's teacher-guided story; "student-only inference" survives, "lightweight" does not.
3. **A17: MLLM × AoM × student ensemble** via the existing word-span voter (+0.5-1 more).
4. A18: 13B/32B-class MLLM with QLoRA if 7B falls short (80GB A100).
5. (Non-GPU unlocks regardless: DQPSA files via Baidu/author email = 71.9 rootstock; VLHA authors.)
Projected honest outcome on A100: **t2017 beaten with high confidence; t2015 72.5+ likely (A15+A17), not guaranteed.**

## Suggested order
**Speed (do first, low-risk):** P2 + P6 → P1 (huge one-time labeling cut) → P4 → P3 → P5 only if freezing CLIP
is acceptable. After P6, edit `queries.md` C4 to reflect the vocab filter is now used.
**Accuracy (one twitter2015 run each to measure):** A1 + A5 together (cheap, target MASC) → A2 sweep → A3 →
A4/A6 only if the cheap ones leave a gap worth a new dependency. After A3, note the layer-wise LR under B3;
after A6, mark A4 in `queries.md` resolved.

---

## A11 — EVIDENCE RELIABILITY LEARNING (new hypothesis, measured, DISOBEYING)
Per-token softmax reliability `w` over [text, vision, KG]; each modality scaled by `3·w`
(zero-init final layer → uniform/identity warm start) before KAN fusion. Stronger inductive
bias than the relevance gate (which only decides *whether* the image matters — here the three
modalities **compete**). Files: `config.fusion_reliability`, `models.py` (reliability_mlp +
forward weighting), `scripts/tune_run.py --reliability`. Claim: *multimodal evidence should be
weighted by estimated per-modality reliability, not only aspect-visual relevance.*

**MEASURED t2015** (champion recipe = DeBERTa-v3-large + CRF + aux-ASC + conf-append + feat-gate):
| run | joint F1 | MATE | MASC Acc/F1 | vs champion |
|---|---|---|---|---|
| champion (ALLIN_deb) | 66.41 | 85.40 | 76.95/70.95 | — |
| **+ A11 reliability** | **66.60** | 85.17 | 77.34/70.75 | **+0.19 (flat, noise)** |

best_dev rose (68.19 vs 67.02) but **test did not follow**. Verdict: a fusion tweak moves the
student ~0 — kept as an honest measured *contribution*, not a bar-breaker.

---

## A15 — MLLM BACKBONE CHAPTER (Qwen2.5-VL, measured; all numbers real)
After the student (66.6) and AoM graft (69.30) capped below the 72.5 bar, we tested a vision-
language **model class** (published at 72-78 on these splits). Pipeline in `mllm/`:
data→instruction-JSON (`common.py`, AoM canonical splits + convention-aware prompt), LoRA/full-FT
SFT (`train_qwen.py`, 8-bit paged AdamW for `--full-ft`, vision frozen), generate→parse→span-aligned
AESC micro-F1 (`eval_qwen.py`, self-validated ~100 on gold round-trip). GPU: RTX PRO 6000 Blackwell
96 GB (Qwen runs), then Tesla T4 (student/A11).

**MEASURED t2015** (test joint F1, strict span+polarity micro-F1):
| system | joint F1 | notes |
|---|---|---|
| Qwen2.5-VL-7B LoRA (5 ep) | 68.76 | MATE 85.14, MASC 80.76; overfit (train loss→6e-4) |
| Qwen2.5-VL-7B full-FT (4 ep, lr 1e-5) | 63.26 | **undertrained** — dev still rising, recipe miss |
| **Qwen ∩ AoM (inter_qpol)** | **69.40** | **P 72.98 (>bar)**, R 66.15 — best overall |
| non-VL student × AoM (best combo) | 66.64 | student weaker than the 7B |
| A11 reliability student | 66.60 | flat (see above) |
| Qwen2.5-VL-32B | **NOT RUN** | needs ≥48 GB; T4 (16 GB) can't fit even 4-bit |

**Frontier finding (why nothing crosses 72.5).** Beating all three t2015 cells needs **P>72.3 AND
R>72.7 simultaneously**. Cross-model agreement (intersection) gives **P 73-80 but R collapses to
58-66**; union gives **R 71-73 but P 62-64**. No point on the frontier reaches P≈R≈72.5. Root cause =
the dataset's **aspect-selection convention** (110/182 dev false-positives are *reasonable-but-
unannotated* entities like "Regions Bank", "# VMworld"), which task-specific pretraining
(VLP-MABSA→AoM→the undownloadable VLHA/DQPSA) encodes and a general model cannot guess.

**HONEST FINAL VERDICT (4 independent measurements): non-VL 66-68 · 7B VL 68.76 · everything-combined
69.40 · A11 66.60 — all 4-6 F1 below the 72.5 bar.** Beating all baselines is **compute-locked**:
it requires a **32B-class MLLM** (published 74-79 on these exact splits) on a **≥48 GB GPU**. It cannot
be done on a T4, nor with a non-VL student, at *any* time budget — the gap is **model-class, not
effort or architecture**. TARKAN's components (relevance, KG, KAN, A11 reliability, CRF, rich-ASC)
lift *weak* bases by +0.5-2.5 and are **absorbed** by strong bases — their value is inversely
proportional to backbone strength.

### Applied this session (patch registry)
- **A11** evidence reliability (`config.fusion_reliability`, `models.py`, `tune_run.py --reliability`) — measured 66.60, flat.
- **A15** MLLM SFT pipeline (`mllm/`: `common.py`, `train_qwen.py`, `eval_qwen.py`, launchers) — 7B LoRA 68.76.
- **convention-aware prompt** (`common.py INSTRUCTION`) — opinion-target + neutral-convention guidance.
- **full-FT path** (`train_qwen.py --full-ft`, 8-bit paged AdamW, vision frozen) — verified grads flow; 63.26 (undertrained).
- **heterogeneous Qwen×AoM voter + combination probes** (`graft/qwen_probe.py`, `nonvl_probe.py`, `ensemble_t15_qwen.json`) — best combo 69.40.

### To actually beat all baselines (open, needs the 96 GB card back)
Qwen2.5-VL-32B LoRA on t2015 (`bash`-download 64 GB, ~3-3.5 h bf16 LoRA + grad-ckpt) → expected 74-79 →
clears P/R/F1 cells; then `inter_qpol` with AoM for precision margin. Pipeline is built and smoke-tested;
only the 32B weights + a ≥48 GB GPU are missing.
