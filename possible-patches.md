# TARKAN — Patch / Hypothesis Ledger

**READ THIS SECTION FIRST.** Chapters below are in chronological order; the newest is at the
top. Identity used throughout: `joint = MATE_F1 × a`, where `a` = polarity accuracy on
*correctly-extracted* spans (holds to ±0.1 on every measured run).

## ⚠ THE BAR MOVED (2026-08-12) — `TARKAN_new.pdf` supersedes `TARKAN_old.pdf`
The new manuscript's Table 1 adds **MADSC (PR 2026)** and **SMCAF (Inf. Fusion 2026)**.
MADSC now dominates VLHA on t2015, so the beat-everything target is **MADSC, not VLHA**:

| t2015 target | P | R | **F1** | source |
|---|---|---|---|---|
| VLHA (PR 2025) — the OLD bar | 72.3 | 72.7 | 72.5 | new Table 1 |
| **MADSC (PR 2026) — the NEW bar** | **72.8** | **73.1** | **72.9** | new Table 1 |
| DQPSA (AAAI 2024) | 71.7 | 72.0 | 71.9 | new Table 1 |
| SGBIS 71.1 · CORSA 69.9 · TCMT/Vanesa 69.8 · SMCAF 69.5 · AoM 68.6 | | | | new Table 1 |
| TARKAN claimed (3-run avg) | 74.4 | 75.8 | 75.1† | new Table 1 |

**To beat EVERY t2015 baseline we now need P > 72.8 AND R > 73.1 AND F1 > 72.9** (was
72.3/72.7/72.5). Sub-task bars also rise: **MATE > 88.2** (VLHA; DQPSA 87.7, MADSC 86.60)
and **MASC Acc > 82.34** (MADSC; VLHA 81.50 Acc / 81.70 Mac-F1).

**Required (MATE, `a`) pairs for joint > 72.9:** MATE 88.0 → `a` **82.8** · MATE 88.5 →
`a` **82.4** · MATE 89.0 → `a` **81.9** · MATE 90.0 → `a` **81.0**.
Chapter B's best was MATE 88.0 × `a` 81.3 = 71.26, so **the gap grew from 1.24 to 1.64.**

## ★ The new manuscript ADOPTS two patches Chapter B discovered empirically
This is the most useful thing in the new PDF — two of our best-measured "disobeying"
patches are now **the paper's own architecture**, so they are reclassified **obeying**:

| New-paper mechanism (§3.3, Eq. 6–8, Eq. 27) | Our patch | Chapter-B measurement |
|---|---|---|
| **Candidate aspect-anchor generator**: a preliminary token-level head over aspect-only **B-ASP / I-ASP / O**, supervised by `L_anc`, whose spans condition evidence selection — "trains the anchor generator **independently of sentiment polarity**" | **B1** dedicated O/B/I head | MATE 85.40 → **87.00**; the +1.6 joint came from exactly this |
| **Auxiliary span-level sentiment head** `L_asc` on the fused aspect representation | **A7-rich** dedicated ASC head | **+2.1 MASC Acc** (78.5 vs 76.95) |

Total objective: `L = L_tag + λ_anc L_anc + λ_asc L_asc + λ_rel L_rel + λ_kg L_kg`, with
`(λ_anc, λ_asc, λ_rel, λ_kg) = (0.3, 0.5, 0.5, 0.3)`. Paper ablation: **w/o `L_anc` → 74.2
(−0.9)**. Other now-specified details: BERTweet-base + CLIP-ViT-B/32, max text len 128,
hidden 768, AdamW 2e-5, batch 16, dropout 0.3, **patience 3**, **3-run average**, KAN
2 layers × width 256 / grid 5 / spline order 3, top **M = 10** KG triples, teacher
**Llama-3.2-3B-Instruct**, captions **BLIP-base** (teacher-side only).

**Reading this honestly.** The new paper's own inference flow — *preliminary anchors →
evidence → fusion → final tagging* — is the architecture-level version of the two-stage
workaround Chapter B arrived at independently, and it retires the §3.6 circularity that
forced `evidence_dropout`. That is a genuine convergence and it means our strongest
extractor is now paper-faithful. It does **not** by itself explain the claimed numbers:
75.1 joint / 90.2 MATE on **BERTweet-base + CLIP-ViT-B/32** sits ~8 F1 above every
measured system in that backbone class (Chapter A: 61–67; our best student 66.6), and
above every VL-pretrained baseline. Chapter A/B also measured the evidence streams this
objective supervises (`L_rel`, `L_kg`) at ≈0 on the joint metric, and the fusion family
(KAN vs MLP vs gated) as flat. So we implement the new architecture and **report what it
measures**, without assuming the headline is reachable on that backbone.

| Chapter | Hardware | Best t2015 joint F1 | Status |
|---|---|---|---|
| **C. T4 restart (2026-08-12, current)** | 1× T4 16 GB | *in progress* | rebuilding + 3 un-pulled levers |
| B. A100 decomposition (2026-07-11→14) | 5× A100-40 (shared) | **71.26** | ceiling = backbone error-correlation |
| A. T4 era (student → AoM graft → 7B MLLM) | 1× T4 16 GB | 69.40 | model-class-bound |

## ★ MASTER VERDICT TABLE — which patches actually helped (all measured, no projections)

**HELPED — keep (ranked by size of gain):**
| Patch | Gain | Where | Chapter |
|---|---|---|---|
| **B1** marginalized MATE decode (sum polarity out *before* O/B/I argmax) | **+1.6 joint**, all recall | `evaluate.py`, `experts/mate_expert.py` | B |
| **B1** dedicated O/B/I head + word-level CRF extractor (drops the 7-tag joint head) | MATE 85.4 → **87.0** single, 88.0 ens | `experts/mate_expert.py` | B |
| **diverse MASC ensemble** (4 Qwen2.5-VL-7B + 4 text: twitter-roberta×3, bertweet) | 66.4 → **71.26** | `experts/masc_enc.py` | B |
| **B3** log-avg (geometric) MASC combiner + dev-tuned span-τ on a bar-margin objective | locked 71.26 | `experts/assemble.py` | B |
| **B2** MASC = Qwen2.5-VL-7B 4-bit LoRA, aspect marked **in place**, verbalizer head | ~79/member | `experts/masc_expert.py` | B |
| **A4** word-level CRF (on the weak student) | +2.5 MATE | `models.py`, `losses.py` | A |
| **A7-rich** dedicated ASC head (mean+max+first→MLP) | +2.1 MASC Acc | `models.py`, `evaluate.py` | A |
| **O-levers** evidence-dropout 0.5→0.2, KAN 768, patience 8 | +2.9 joint | `config.py` | A |
| **hi-res images** for MASC (max_pixels 100k→200k) | dev 78.3 → 80.0 | `masc_expert.py` | B |
| **B7 EMA** (ramped, eval-on-EMA) | ep6 ema 79.86 > raw 79.32 | `masc_expert.py --ema` | B |

**DID NOT HELP — measured, do not re-try blindly:**
| Patch | Result | Why it failed |
|---|---|---|
| **B4** pool t2015+t2017 train | MATE −3.2 (P −4.1) | t2017 has 2.04 aspects/sent vs t2015's 1.51 → prior shift → over-prediction. Also *unfair* (baselines are t2015-only). |
| **B10** rationale distillation | **strongest single members** (test 79.2, macF1 74.6) but ensemble 71.17 < 71.26 | correlated with the Qwen block — best member ≠ best ensemble |
| cross-family LLaVA-1.5-7B member | single 78.6 (on par) but ensemble 71.04 | errors **correlated** with Qwen: both_wrong 0.149 vs indep 0.043 → **ratio 3.43** |
| advanced combiners (Dawid–Skene, spectral, dev-confusion, dev-weighting, per-class log-bias, confidence-gate, max-conf route) | all **below** log-avg on test | dev +0.9 → test −0.6 every time; dev n=1122, binomial σ≈1.2 |
| MATE arch-diverse re-selection (19 members, dev-greedy) | dev 88.54 → **test 87.61** | weak members give greedy selection noise to overfit |
| B9 same-tweet SupCon (444 within-tweet diff-pol pairs) | weak members | — |
| caption-augmented text (BLIP caption → text model) | 71.17 | dilutes |
| A1 class-weighting / A5 label smoothing | joint −2 to −54 | Table-1 metric is **micro**-F1; these help macro only |
| A3 layer-wise LR, A11 evidence-reliability, KAN capacity, EMA/SWA on MATE, vision-attn LoRA | flat (±0.2) | fusion/optimizer tweaks move the student ~0 |
| post-hoc neurosymbolic rules on AoM | −11 | AoM already ingests SenticNet (absorption) |
| KG evidence grafted into AoM | graft − control ≈ −0.2 | components' value ∝ 1/backbone strength |
| Qwen2.5-VL-32B | stalled 72–77 (below the 7B) | recipe-locked (bs1/cold-head/tune-vision); never a clean capacity test |

**The one structural conclusion (Chapter B, §7v/§7x):** the 71.26 ceiling is **backbone
error-correlation**, not tuning. Every strong member inherits Qwen2.5-VL's blind spots;
oracle-over-members is 93 but no *realizable* selector extracts it, because member confidence
≠ correctness on the disagreement cases. Only a **mechanism-decorrelated** strong member, or
new training signal, can move it.

---

# ===== CHAPTER C — T4 RESTART (2026-08-12) =====

## C.0 Situation on arrival (all verified, not assumed)
- **Hardware regression: 1× Tesla T4 16 GB, idle.** The 5×A100 server is gone. T4 is compute
  capability 7.5 → **fp16 only, no bf16** (matters for Qwen2.5-VL numerics).
- **`data/` lost AGAIN** (4.7 MB, senticnet only). **RESTORED + verified**: cloned
  `CopotronicRifat/TwitterDataMABSA` (2.0 GB) → 8288 t2015 + 4819 t2017 images;
  record counts match paper Table 2 exactly (**3179/1122/1037 · 3562/1176/1234**).
  Images symlinked (`data/images/<year>` → raw clone) to avoid a 2 GB copy.
- **`experts/` does not exist.** The entire Chapter-B pipeline (`mate_expert.py`,
  `masc_expert.py`, `assemble.py`, `masc_enc.py`, `vl_masc.py`, `common.py`,
  `gen_rationales.py`) was never committed and is **gone**. Chapter B must be rebuilt from
  this ledger. Surviving assets: `graft/` (7 MB — AoM official prediction dumps
  `dump_t15_*.jsonl`, `qwen_t15_*.json`, evidence JSONs) and `mllm/` (A15 Qwen SFT pipeline).
- **git `main` has ZERO commits**; every file untracked. Remote `uditsenapaty/TARKAN` set.
- **No env, no HF cache.** Lightning blocks venvs (1 conda env per Studio) → installed into
  the default env: **torch 2.8.0+cu128, transformers 4.56.2, peft 0.20.0, bitsandbytes 0.50.0**.
- **NEW ASSET this session:** `referred_clones/` — 16 baseline implementations, incl. **DQPSA
  full source** and **VLP-MABSA**. Chapter B never had these.

## C.1 ★ FORENSIC AUDIT — VLHA (the 72.5 bar) IS NOT REPRODUCIBLE FROM ITS RELEASED CODE
`referred_clones/VLHA/` is 312 KB, 13 files, **model code only** — no training driver, no
evaluation script, no weights, no requirements. Reading `VLHA/model.py` (the only entry point):

| Defect | Evidence in `VLHA/model.py` | Consequence |
|---|---|---|
| No test/dev evaluation at all | loads only `train.txt` via hardcoded `E:/PythonProject2/...` | the released artifact **never computes a test number** |
| Metric is not Table-1's | `aspect_accuracy = (tags_pred==seq).sum()/numel()` — token-level binary agreement over 2 tags | **joint AESC micro-F1 is never implemented anywhere in the repo** |
| Polarity read off gold spans | `fcmodel(output, aspect_term_seq)` uses the **gold** binary aspect sequence | not an end-to-end pipeline |
| Optimizer rebuilt per sample | `optimizer = Adam(all_parameters, lr=2e-5)` **inside** the per-sample loop | Adam moments reset every sample; `StepLR` never steps |
| Fake batching | slices `batch_samples` then iterates one sample, `.backward()` per sample | `batch_size=32` is cosmetic |
| Untrained random head in the loss path | `NNmodel = nn.Sequential(Linear(768,2), Softmax)` re-instantiated **inside the loop**, never in the optimizer; its output feeds `crfmodel(output_tensor, …)` | the CRF loss consumes a fresh random projection every step |
| Loss CRF ≠ decode CRF | loss uses `crfmodel`; `tags_pred` comes from `crfmodel2`, which receives no loss | the decoder is never trained |
| Silent sample dropping | bare `except Exception: continue` around the whole forward, but denominator stays `len(samples)` | reported accuracies are over an unknown subset |
| Off-by-one span labelling | `binary_sequence[start_token_idx-1:end_token_idx-1]` | labels shifted |

**Honest statement of the finding (R8).** This is **not** a claim that VLHA fabricated numbers,
and **not** a claim that their metric differs from ours — no metric of theirs exists to compare.
The defensible claim: **the top baseline's 72.5 cannot be independently verified, reproduced, or
rescored from its public release**, whereas AoM's 68.6 *was* reproduced here (68.42 their
harness / 68.19 ours). Our metric was separately shown **0.2–0.4 STRICTER** than the baselines'
own on identical predictions, so no margin we report is a lenient-scorer artifact.
**Consequence for the campaign:** 72.5 stays the bar (R7 — baselines enter as published
numbers), but this audit is itself a paper-grade reproducibility contribution, and it retires the
"go read VLHA's eval script to find our missing +1" plan — **there is no eval script to read.**

## C.2 ★ FORENSIC AUDIT — DQPSA (71.9 / MATE 87.7 / MASC 81.1): full source, mechanism recovered
`referred_clones/DQPSA/` is complete and readable (1556 lines). What it actually does — and
every point below is a mechanism Chapter B never tried:

1. **Two separate models, assembled in two stages** — `eval_MABSA(MATE_model, MASC_model, …)`.
   **Independently confirms our decomposition** (`joint = MATE × a`) is the SOTA-standard design,
   not our workaround.
2. **Two thresholds, not one:** `MATE_limit=0.5, MASC_limit=0.3`. Chapter B tuned a span-τ only;
   **the polarity score is also thresholded** by the SOTA baseline. → lever **C3** below.
3. **Metric is plain micro P/R/F1** (`total_correct/total_pred`, `/total_label`) — same family as
   `metrics.py`, so cross-checking is meaningful.
4. **PDQ = "Prompt as Dual Query"** (`PDQ/PDQ.py`): a BLIP-2 Q-Former where the 32 query slots are
   **not** BLIP-2's learned queries but `Qformer.bert.embeddings(query_ids)` — *the prompt's own
   token embeddings*. The prompt therefore (a) cross-attends the image inside the Q-Former and
   (b) is prepended to the text encoder. Losses **ITC + ITM** on the Q-Former.
5. **Frozen, pre-cached visual features** — `samples["image_embeds"]` is loaded from `.pkl`
   (`image_feature` numpy). The ViT is never fine-tuned. **This is what makes it T4-feasible.**
6. **EPE = GlobalPointer span head** (`Text_encoder/epe.py`, `sparse_attn_model.py`): a RoPE
   bi-affine `einsum('bmhd,bnhd->bhmn')` producing an **L×L (start,end) score matrix**, lower
   triangle masked, `BCEWithLogitsLoss(reduction='sum') / num_prompt`. **No BIO tagging at all** —
   so the entire class of BIO decode pathologies (the +1.6 B1 bug) cannot occur, and every span
   gets a *directly calibrated* probability.
7. **MASC is span extraction over the prompt's own option list.** `prompt_mask` restricts the L×L
   matrix to the sub-block covering the literal string **`[ positive, neutral, negative ]`** in the
   prompt; the model points at the correct option. For MATE, `prompt_mask` instead covers the tweet.
   **One architecture, two tasks, switched by prompt + mask.**
8. Unavailable: their `Text_encoder/model_best` (FSUIE-pretrained BERT) and
   `checkpoints/pretrain_ckp/MASC_best_model.pt` (MABSA pretraining) — Baidu, expired.
   **Public substitute identified:** `Salesforce/blip2-itm-vit-g` ships exactly the BLIP-2 stage-1
   Q-Former with pretrained `vision_proj` / `text_proj` / `itm_head` — i.e. the same ITC+ITM
   rootstock PDQ builds on, ~129M image-text pairs of VL grounding.

**Why this is the lever Chapter B was missing.** §7v/§7x concluded the ceiling is
error-correlation and that the one un-pulled lever is a strong member from a *genuinely different*
mechanism. LLaVA failed that test because LLaVA **is** architecturally Qwen-like (CLIP-ViT +
autoregressive LLM, instruction-tuned) → correlation ratio 3.43. A PDQ member differs on every
axis that matters: **EVA ViT-g not CLIP-ViT**, **discriminative bi-affine span scoring not
autoregressive verbalizer**, **ITC/ITM contrastive pretraining not instruction tuning**. And it is
~230 M trainable params on cached features → **minutes per member on a T4**, versus ~3 h for one
Qwen-7B member.

## C.3 PATCH MENU — Chapter C (status tracked here; measured results appended as they land)
| id | patch | faithfulness | T4 cost | status |
|---|---|---|---|---|
| **C0** | rebuild canonical data layer + scorer; regression-gate on AoM's dumped preds (must re-score **68.19**) | obeying | minutes | ✅ **PASSED** |
| **C1** | rebuild B1 MATE expert = the new paper's anchor generator (O/B/I + word-CRF), DeBERTa-v3-large ×seeds | **obeying** (now §3.3/Eq. 8) | ~16 min/seed | ⚠ s42 = **85.77** (target 87.0) |
| **C2** | **NEW — PDQ port**: BLIP-2 `blip2-itm-vit-g` Q-Former, prompt-as-query, cached frozen ViT-g feats, EPE GlobalPointer head, prompt-masked option-scoring MASC + span MATE | gray — new backbone, report explicitly | ~6 min feature cache + ~20 min/member | — |
| **C3** | **NEW — joint-expected-F1 decoding**: keep a pair iff `P(span)·P(polarity) > τ`, τ dev-tuned on the bar-margin objective (DQPSA thresholds both stages; we only thresholded spans) | obeying (decode rule unspecified) | free | — |
| **C4** | **NEW — annotation-policy span reranker**: high-recall stage-1 → candidate spans → "would an annotator mark this?" classifier fit on **train-set out-of-fold** predictions (NOT dev — every Chapter-B combiner overfit dev's n=1122) | disobeying | ~1 h | — |
| **C2b** | **PDQ variants** — swap the text encoder to Twitter-domain (bertweet / twitter-roberta), vary `n_query` and seeds. PDQ decorrelates, so N diverse PDQ members is the cheap route to `a` | gray | ~12 min/member | running |
| **C2c** | **PDQ + ITC/ITM losses** (DQPSA uses `itc=itm=epe=1.0`; we run epe-only via their own `no_its_and_itm` path). Needs HF `vision_proj`/`text_proj`/`itm_head`/`temp` kept from the checkpoint and a text-only Q-Former pass. Untried faithfulness gap | obeying-to-DQPSA | ~15 min/member | TODO |
| **C4b** | **MATE recipe probe** — C1 overfit (loss 0.013 @ ep9) and sits 1.2 under Chapter B's single-seed 87.00. Probe lr 1e-5/2e-5 × head-lr 1e-4 × dropout 0.3 × 16 epochs before building more modules | obeying | ~16 min/run | queued |
| **C5** | 1–2 Qwen2.5-VL-7B 4-bit LoRA MASC members on T4 (fp16-only risk), only if `a` still short | gray (backbone) | ~3–4 h/member | deferred — our best member (79.75) already matches Chapter B's Qwen members |
| **C6** | **AADG** (`experts/aadg.py`) — BLIP caption → spaCy object mentions → dual similarity (CLIP direct + grid-region-mediated) → replace mentions with the grounded aspect → feed `D_aspect` to every MASC member | disobeying TARKAN / faithful-to-MADSC | ~15 min one-off + free per member | running |
| **C7** | **Calibrated modality gate** (`experts/masc_gated.py`) — MADSC Eqs. 9/13/14/16/19: `u=σ(w_u·sim+b_u)`, `g=σ(W_g·u+b_g)`, `z=g·v_a+(1−g)·t_a`, plus `L_conf` BCE on deterministic pseudo-alignment labels. **Four learnable scalars.** Their biggest ablation (−5.74 Mac-F1 without) | disobeying TARKAN / faithful-to-MADSC | ~15 min/member | queued |

**Why C7 is not a repeat of Chapter B's dead relevance gate (§7c).** TARKAN's `r^T` gate and
MADSC's `g_a` are the same *mechanism*, and Chapter B measured TARKAN's at ≈0 (stage-2
encoder 75.41 vs 78.5 for a plain head). The difference is the **driver**: Chapter B learned
the gate from scratch on 3.2k aspects from a CLIP+DeBERTa trunk with no grounding signal,
whereas C7 drives it from an *externally computed, calibrated* CLIP alignment score with a
deterministic pseudo-label supervising the calibrator. Same gate, different input — that is
the part Chapter B never had, and it is why this is worth one more measurement rather than
being pre-emptively retired.

## C.4 MEASURED — Chapter C results log (append one row per completed run)

**C0 regression gate — PASSED, exactly.** Scoring AoM's own dumped predictions
(`graft/dump_t15_test.jsonl`) with the rebuilt scorer reproduces Chapter B to the decimal:
**joint P 67.45 / R 68.95 / F1 68.19 on exactly 1037 gold pairs, 674 sentences**
(dev: 1122 pairs / 727 sentences). Our reconstruction from the raw `$T$` tsv is therefore
**bijective with AoM's canonical split**, and `metrics.py` is unchanged from Chapter B.
Everything downstream is measured on the same footing as the published baselines.

**Deterministic battery (before any GPU spend) — all green.**
* aspect-only O/B/I anchor tagging is **lossless: oracle F1 = 100.00** on train/dev/test
  (independently reproduces Chapter B, §B.2).
* vendored CRF verified against brute-force enumeration over all 3^5 paths:
  `logZ` matches to 5e-7 and Viterbi returns the exact arg-max path; marginals sum to 1.

**C1 — anchor generator / MATE, DeBERTa-v3-large, seed 42** (t2015):
| split | P | R | F1 |
|---|---|---|---|
| dev | 85.69 | 88.06 | 86.86 (best @ ep9) |
| **test** | **84.14** | **87.46** | **85.77** |

Below Chapter B's single-seed 87.00 by 1.23. Train loss reached 0.013 by ep9 (heavily
overfit), and unlike Chapter B ("MATE has no dev/test gap") this run shows a **1.1 dev→test
gap**. Next levers, in order: multi-seed marginal averaging (Chapter B's proven +0.3–1.3),
then LR/epoch retune. Not yet a like-for-like reproduction — flagged, not papered over.

**C2 — PDQ member (BLIP-2 Q-Former + EPE GlobalPointer), seed 42, bert-base text encoder.**
Trains in ~10 min on the T4 on cached ViT-g features (3502 images × 257 × 1408, 0 missing;
caching took ~4 min). Smoke test confirmed gradients reach the Q-Former, i.e. the
visual→text bridge is actually being fine-tuned (Chapter B §7m's lesson).

| MASC member (t2015, gold-span) | dev Acc | **test Acc** |
|---|---|---|
| twitter-roberta-base-sentiment | 77.09 | 77.34 |
| bertweet-base | 75.31 | **78.11** |
| **PDQ (BLIP-2 Q-Former, bert-base)** | 73.71 | 76.37 |

**★ The decorrelation test — PDQ PASSES where LLaVA failed (test, n=1037):**
| pair | both_wrong | independence | **ratio** |
|---|---|---|---|
| twitter-roberta ∥ bertweet (text ∥ text) | 0.150 | 0.050 | **3.03** |
| twitter-roberta ∥ PDQ | 0.145 | 0.054 | 2.70 |
| **bertweet ∥ PDQ** | 0.131 | 0.052 | **2.54** |

Unique-right share (correct where ALL others are wrong): twitter-roberta 2.03% ·
bertweet 3.38% · **PDQ 3.95%**. 3-member oracle **88.91**, all-wrong 11.09.

So the **weakest** member contributes the **most unique signal** and forms the **least
correlated** pair — the opposite of Chapter B's LLaVA result (ratio 3.43, and it hurt).
This is the first positive evidence that the mechanism axis (Q-Former cross-attention +
discriminative bi-affine span scoring) decorrelates where the "different VL family" axis
did not. **Next: PDQ's weakness is its vanilla `bert-base-uncased` text encoder on tweets
(DQPSA used a task-pretrained BERT there) — swapping to bertweet/twitter-roberta should
keep the mechanism diversity while adding strength.**

**C3 — joint-expected-F1 decoding: small, real, and dev-consistent.**
| members | mode | τ | MATE@τ | `a` | dev joint | **test joint** |
|---|---|---|---|---|---|---|
| 1 MATE + 1 text | span (Chapter B) | 0.88 | 85.74 | 77.46 | 66.52 | 66.41 |
| 1 MATE + 1 text | **joint (C3)** | 0.55 | 85.05 | 77.98 | 66.40 | 66.32 |
| 1 MATE + 3 MASC | span (Chapter B) | 0.88 | 85.74 | 77.68 | 64.74 | 66.60 |
| 1 MATE + 3 MASC | **joint (C3)** | 0.55 | 85.75 | 77.85 | **65.04** | **66.76** |

C3 is **negative with one weak polarity member (−0.09) and positive with three (+0.16)**,
and it moves dev and test in the same direction, so it is not a test-set artifact. It
works by trading MATE precision for `a` at fixed MATE F1 — exactly the intended mechanism.
Keep it, but it is a decimals-level lever, not a bar-breaker.

**C.4b ORACLE-ANCHOR DECOMPOSITION (free, CPU) — where the loss actually is.**
| t2015 test | value |
|---|---|
| oracle-anchor joint F1 (= MASC ensemble acc on **gold** spans) | **78.11** |
| predicted-anchor joint F1 (measured, 3 members) | 66.76 |
| **anchor-error propagation cost** | **11.35** |

**Do not compare this to the paper's 77.0 → 75.1 (cost 1.9).** Per §3.3 the manuscript's
gold anchors replace only the *conditioning query* for evidence selection — "these anchors
serve only as conditioning queries … rather than as final aspect–sentiment outputs" — so
that ablation is structurally incapable of costing much. Ours replaces extraction outright.

What it tells *us*: with perfect extraction our current polarity stack caps at **78.11**,
and `a` on correctly-extracted spans (77.85) is essentially the same number — so MASC
member quality moves both terms. **Ranked gaps to the MADSC bar: polarity +4.5 (77.9 →
82.4) ahead of extraction +2.75 (85.75 → 88.5).** Chapter B reached `a` = 81.3 only with
4 Qwen-7B members at 79–80 each; clearing 82.4 needs members stronger than anything either
chapter has produced, so **72.9 is not a realistic target for this T4 session** — the
honest aim is to re-reach and then exceed Chapter B's 71.26 with cheaper, decorrelated
members.

**C.4c MASC MEMBER LADDER (t2015, gold-span acc) — the text encoder was PDQ's bottleneck**
| member | dev | **test** |
|---|---|---|
| PDQ + bert-base-uncased | 73.71 | 76.37 |
| **PDQ + twitter-roberta** | 76.56 | **78.78** |
| PDQ + bertweet-base | — | 70.78 |
| twitter-roberta (text only) | 77.09 | 77.34 |
| bertweet-base (text only) | 75.31 | 78.11 |
| **deberta-v3-large (text only)** | 77.90 | **79.36** |

Swapping PDQ's text encoder from `bert-base-uncased` to `twitter-roberta` bought **+2.41**
(76.37 → 78.78), confirming the diagnosis that PDQ's weakness was domain mismatch in the
text tower, not the Q-Former mechanism. `deberta-v3-large` is the single strongest member
(79.36), consistent with Chapter A's 78.5 for a dedicated ASC head on the same backbone.

**★ Decorrelation scales INVERSELY with member strength (6-member matrix, test n=1037):**
| pair | ratio | | pair | ratio |
|---|---|---|---|---|
| pdq_twrob ∥ pdq_btw | **1.62** | | twrob ∥ pdq | 2.70 |
| twrob ∥ pdq_btw | **1.65** | | pdq ∥ pdq_twrob | 2.69 |
| deb ∥ pdq_btw | **1.76** | | deb ∥ pdq_twrob | 2.91 |
| btw ∥ pdq_btw | **1.79** | | twrob ∥ btw | 3.03 |
| pdq ∥ pdq_btw | 1.91 | | btw ∥ pdq_twrob | 3.03 |
| btw ∥ pdq | 2.54 | | twrob ∥ pdq_twrob | 3.29 |

Unique-right share: **pdq_btw 2.60%** (the *weakest* member, 70.78) vs deb 1.06%,
twrob/btw 0.58%, pdq_twrob 0.29%, pdq 0.10%. **6-member oracle = 93.44**, which reproduces
Chapter B's 93.4 almost exactly. So the diversity is real and the oracle headroom is
intact — the binding problem remains extraction of that headroom, not its existence.

**C.4d MEMBER-SUBSET SWEEP — dev-selection ANTI-correlates with test (reproduces §7x)**
| dev F1 | members | mode | τ | test P | test R | **test F1** | `a` |
|---|---|---|---|---|---|---|---|
| **67.94** | strong4 | joint | 0.54 | 68.01 | 69.91 | 68.95 | 80.65 |
| 67.76 | strong4 | span | 0.65 | 68.79 | 69.91 | 69.34 | 80.47 |
| 67.59 | no_pdqbtw | span | 0.65 | 68.98 | 70.11 | 69.54 | 80.69 |
| 67.41 | **all6** | **span** | 0.65 | **69.17** | **70.30** | **69.73** | 80.91 |
| 67.17 | all6 | joint | 0.55 | 68.95 | 69.82 | 69.38 | 81.44 |
| **67.05** | deb+pdqtw+pdqbtw | span | 0.65 | 69.26 | 70.40 | **69.82** | 81.02 |

Dev spans 67.05–67.94 (0.89) while test spans 68.95–69.82 (0.87) in **reverse order**:
the dev-best config is the test-worst-but-one, and the dev-worst is the test-best. Dev
(n=1122, binomial σ ≈ 1.2) cannot discriminate between these. **Reported number therefore
uses Chapter B's robust rule — a fixed curated set (ALL members), log-avg, span-τ, no
dev-greedy selection: joint 69.73.** Dev-greedy selection would report 68.95; the 69.82
row is NOT claimable (that is test peeking).

### ★ CHAPTER C BEST SO FAR (t2015 test, R6-clean)
```
MATE   P=~86.2  (3-seed marginal ens, dev-tuned tau=0.65)
a      = 80.91
JOINT  P=69.17  R=70.30  F1=69.73        bar 72.8 / 73.1 / 72.9  ->  margin -3.17
```
Trajectory: 66.41 (1 MATE + 1 text) → 66.95 (3 MATE + 3 MASC) → **69.73** (3 MATE + 6 MASC).
`a` = 80.91 already approaches Chapter B's 81.3 **without a single 7B model** — the whole
polarity stack here is ≤435M params and trains in minutes on one T4. The remaining deficit
is now concentrated in **MATE (85.75 vs the 88.0 Chapter B reached on this exact
architecture)**, which is a recipe gap, not an architecture gap → C4b probes running.

## C.5 ★ MADSC (the new bar) DISSECTED — the gap is 100% polarity, and our extractor already wins

Reading the MADSC paper (Pattern Recognition 179, 2026) changes the target arithmetic
completely. **MADSC's own t2015 MATE is 86.60 — BELOW DQPSA (87.7) and VLHA (88.2).**
Its JMASA 72.9 therefore implies `a` = 72.9 / 86.60 ≈ **84.2**, against a gold-span MASC
accuracy of 82.34. So the SOTA system does not win on extraction at all; it wins entirely
on polarity, using a **BART-base backbone with only 156.7M trainable params** — no 7B model
anywhere. That is very good news for a T4.

**MADSC's own ablations say where its polarity comes from (t2015):**
| removed | MATE F1 | MABSA Mac-F1 | JMASA F1 |
|---|---|---|---|
| full MADSC | 86.60 | 78.38 | 72.9 |
| w/o modality gating | 77.83 (**−8.8**) | 72.64 (**−5.7**) | 70.7 (−2.2) |
| w/o confidence calibration | 78.69 (−7.9) | 76.25 (−2.1) | 71.4 (−1.5) |
| GPT-4o → BLIP2 caption | 81.05 (−5.6) | 76.97 (−1.4) | 71.8 (−1.1) |
| max-product → soft multi-region | 84.13 (−2.5) | 76.62 (−1.8) | 71.9 (−1.0) |

Two things follow. (1) **Calibrated modality gating, not the description, is the load-bearing
part.** (2) Chapter B's "caption-augmented text DILUTES (71.17)" result is *not* evidence
against this: MADSC's central claim is precisely that **generic** captions are the wrong
signal ("granularity mismatch", scene-level saliency, multimodal laziness). The
aspect-**conditioned** rewrite is a different object, and their §4.6 shows it lifting three
frozen baselines as a pure plug-in (TomBERT +2.37/+2.81 Acc/Mac-F1, FITE +1.46/+1.35,
ITOAOF +1.44/+2.23). So this is a genuinely untried lever here, not a repeat.

**New patch C6 — AADG (`experts/aadg.py`), with deviations recorded honestly:**
| MADSC | ours | why |
|---|---|---|
| VinVL top-36 boxes | **3×3 grid crops + full image = 10 pseudo-regions** | detectron2/VinVL not installable here; keeps the property the paper argues matters (aspect and object must agree on the SAME local region), drops the detector |
| GPT-4o captions | `Salesforce/blip-image-captioning-base` | open weights; their ablation prices this at −1.4 MABSA Mac-F1 |
| learned calibrator `(w_u,b_u)` trained jointly | threshold `sim_final` directly | used as a plug-in (their §4.6 protocol); the learned gate needs joint training |
| α=0.7, β=0.3, τ=0.6 | same (their t2015 values) | — |

### C.6 ★★ MATE SOLVED — 87.01, ABOVE THE SOTA SYSTEM'S OWN EXTRACTOR
The C4b recipe probe worked. Two extra members at head-lr 1e-4 (vs the original 1e-3) plus
the three seeds, combined by **plain marginal averaging with NO selection of any kind**:

| MATE member (t2015 test) | P | R | F1 |
|---|---|---|---|
| s42 (lr 2e-5, head 1e-3) | 84.14 | 87.46 | 85.77 |
| s43 · s44 | 83.27 · 82.41 | 86.40 · 87.66 | 84.81 · 84.95 |
| **probeA** (lr 1e-5, head 1e-4, dropout 0.3) | 83.96 | 88.33 | **86.09** |
| probeB (lr 2e-5, head 1e-4) | 83.82 | 87.95 | 85.84 |
| **ALL-5 marginal ensemble (no selection)** | **85.00** | **89.10** | **87.01** |

**87.01 beats MADSC 86.60, CORSA 86.30, AoM 86.20, Atlantis 86.10, M2DF 86.30, CMMT 85.9,
VLP-MABSA 85.7.** Only DQPSA (87.7) and VLHA (88.2) remain above — and VLHA's number is the
one shown in C.1 to be unverifiable from its release. Exhaustive search over all 31 subsets
confirms the selection-free all-5 set (87.01) is within 0.01 of the best possible subset
(87.02), so **no dev-greedy selection is needed or used here**. The head-lr was the bug:
1e-3 on the CRF transition matrix was over-driving it.

**Consequence:** with MATE = 87.01 fixed, `joint = 87.01 × a`, so the bar F1 > 72.9 needs
**`a` = 83.8**, and the per-cell bar (P > 72.8) needs MATE_P 85.00 × a > 72.8 → `a` = 85.6
on precision. Current `a` = 80.91. **Everything now rides on polarity.**

Also new: **bertweet-large MASC = 79.75 test** (best single member so far), deberta-v3-large
79.36, PDQ+twitter-roberta 78.78.

### C.7 ★ THE COMBINER FAMILY IS NOW DEFINITIVELY CLOSED (C8, measured)
Chapter B killed seven combiners (Dawid–Skene, spectral, dev-confusion, dev-weighting,
per-class log-bias, confidence-gating, max-conf routing). All were fit on **dev**, so the
obvious rebuttal was "n=1122 with binomial σ≈1.2 is simply too little data". **C8 removes
that rebuttal.** A multinomial logistic stacker over member log-probabilities, fit on
**3179 train out-of-fold aspects** (2.8× dev, 2-fold, split by *instance* so sibling aspects
never straddle the split, full coverage):

| combiner (4 text members, gold-span acc) | dev | **test** |
|---|---|---|
| plain log-average (selection-free) | 78.34 | **80.91** |
| **OOF stacker** (3179 fitting points) | 78.34 | 79.85 (**−1.06**) |

Even with the correct protocol and 2.8× the data, the learned combiner does not beat the
plain geometric mean. Likely cause: the OOF predictions come from half-data members whose
calibration differs from the full-data members applied at test — the standard stacking
mismatch — and log-average is already near-optimal for near-exchangeable members.
**Conclusion: combination is DONE at log-average. Only better MEMBERS can move `a`.**
That retires the largest remaining "free" idea and redirects all remaining effort to C6/C7.

### C.8 RUNNING TALLY (t2015 test, all selection-free unless stated)
| stage | MATE@τ | `a` | **JOINT** | oracle |
|---|---|---|---|---|
| 1 MATE seed + 1 text member | 85.74 | 77.46 | 66.41 | — |
| 3 MATE + 3 MASC | 86.18 | 77.69 | 66.95 | 88.91 |
| 3 MATE + 6 MASC | ~86.2 | 80.91 | 69.73 | 93.44 |
| **5 MATE (87.01) + 9 MASC, τ=0.71** | **87.43** | 80.40 | **70.29** | **95.18** |
| bar (MADSC) | — | 83.8 needed | **72.9** | — |

`joint = 87.43 × 0.8040 = 70.29` (identity exact). MATE@τ **87.43** now also clears
DQPSA's 87.7? No — 87.43 < 87.7, but it clears MADSC 86.60, CORSA 86.30, AoM 86.20.
**Only 4.82% of test aspects are wrong in every member**, so the ceiling is not information;
it is extraction of information that no combiner can reach.

### C.9 PER-CLASS DIAGNOSIS — we beat MADSC on NEU and lose on POS/NEG (9-member ens, gold spans)
| class | n | P | R | F1 | MADSC F1 |
|---|---|---|---|---|---|
| POS | 317 | 81.21 | **72.24** | 76.46 | **84.4** |
| NEU | 607 | 81.78 | **89.46** | **85.44** | 78.8 |
| NEG | 113 | 76.92 | **61.95** | 68.63 | **74.3** |
| — | | | | acc **81.20** / macro 76.84 | acc 82.34 / macro 78.38 |

Confusion (rows gold): **86 POS → NEU**, 35 NEG → NEU, 45 NEU → POS. NEU is 58.5% of test
and it is swallowing both minority classes; our accuracy deficit vs MADSC (−1.14) is
entirely a class-balance deficit, not a general modelling deficit. Fixing POS recall alone
is worth ≈ +2.4 accuracy, which would put `a` near 83.
**Patch C9 — class-balanced members** (`masc_text.py --class-weight balanced|sqrt`).
Chapter B's A1 verdict ("class weighting hurts, joint −2 to −54") was measured on the
**joint 7-tag BIO head**, where up-weighting minority polarity tags drove the `O` weight to
0.02 and collapsed *extraction*. On a dedicated 3-way MASC head that failure mode is
structurally impossible. Added as extra ensemble members, never as replacements.

### C.10 AADG — first attempt DEGENERATE, fixed by data-driven calibration
Naively porting MADSC's `tau = 0.6` produced garbage:
`"a Facebook in a baseball uniform talking to a Facebook of Facebook"`. Cause: MADSC
thresholds `u * sim_final` where `u` is a **jointly-trained** calibrator; the raw CLIP
text-text cosine has almost no absolute scale (~0.78–0.93 for *any* pair), so 0.6 fires on
every mention, and nothing stopped one aspect claiming all of them. Two deviations fix it:
* **τ = 92nd percentile of the similarity distribution measured on TRAIN only** (never
  dev/test) → τ = 0.6256 over 9170 pairs, range [0.176, 0.732], median 0.563. The
  region-mediated route is what supplies the spread.
* **greedy ONE-TO-ONE matching** (an aspect claims at most one mention, and vice versa).
* `u` exported as a **TRAIN-ECDF rank in [0,1]** rather than a raw cosine, so C7's gate
  calibrator sees a score with usable dynamic range.

Result: **15.8% of train aspects grounded** (16.5% test), 84% receive an explicit
"… is not clearly visible" note, and the rewrites are correct
(`"a Lydia with green hair sitting on a couch"` ← woman).

**★ Independent corroboration of TARKAN's H1.** The paper's own teacher labels only
**10–16%** of images as aspect-relevant. Our CLIP dual-similarity route — a completely
different mechanism, no LLM teacher — grounds **15.8–16.5%**. Two unrelated methods agreeing
on the same relevance rate is real support for TARKAN's central premise that the image is
usually irrelevant, even though Chapter B measured the *gate built on it* as worth ≈0.

### C.11 PATCH C10 — sibling-aspect marking (input-level aspect conditioning)
Every member so far marks **only** the target aspect in place. But t2015 averages
1.51 aspects/sentence and contains **444 within-tweet pairs whose gold polarities differ**,
and the measured failure is exactly the one that predicts: 86 POS aspects predicted NEU,
i.e. the model backing off to the tweet's overall tone. `marked_text(mark_siblings=True)`
additionally brackets the other candidate aspects with `< >`:

```
plain    : Embattled [ Metro ] Councilman Dan Johnson to debate challenger John Witt , …
siblings : Embattled [ Metro ] Councilman < Dan Johnson > to debate challenger < John Witt > , …
```

Chapter B attacked this at the **loss** level (B9 same-tweet SupCon on the 444 pairs → weak
members). This is the **input** level version, costs nothing, and composes with C9.

### C.12 BASELINE STANDINGS — 70.29 already beats 15 of 19 t2015 baselines
Beaten: JML 64.1 · GMP 49.3 · CMMT 66.5 · VLP-MABSA 66.6 · M2DF 67.6 · MultiPoint 67.6 ·
Atlantis 67.3 · MCPL-VLP 68.2 · AoM 68.6 · RNG 68.6 · Vanesa 69.8 · TCMT 69.8 ·
SMCAF 69.5 · **CORSA 69.9** · (and every pre-2023 system).
**Still above us:** SGBIS 71.1 (−0.8) · DQPSA 71.9 (−1.6) · VLHA 72.5 (−2.2) ·
**MADSC 72.9 (−2.6)**.

### C.13 CODE AVAILABILITY OF THE FOUR REMAINING TARGETS
| system | t2015 | code | weights | reproducible? |
|---|---|---|---|---|
| SGBIS (KBS 2026) | 71.1 | none | none | no |
| DQPSA (AAAI 2024) | 71.9 | **full** | Baidu, expired | architecture yes (→ C2 PDQ port) |
| VLHA (PR 2025) | 72.5 | model only | none | **no — see §C.1** |
| MADSC (PR 2026) | 72.9 | **none** (searched 2026-08-13) | none | no |

Only **one** of the four systems still above us ships runnable code, and none ships weights.
Their mechanisms are therefore reimplemented from paper text (C2/C6/C7) with every
substitution recorded, never copied.

### C.14 MEASURED — C9 class-balanced member does exactly what it was designed to do
`vinai/bertweet-large`, identical recipe, only the CE weights change
(counts NEG 368 / NEU 1883 / POS 928 → weights 1.884 / 0.368 / 0.747):

| bertweet-large | NEG R | POS R | NEU R | acc |
|---|---|---|---|---|
| plain | 61.06 | 74.13 | 86.16 | **79.75** |
| **balanced CE** | **66.37 (+5.31)** | **76.34 (+2.21)** | 82.54 (−3.62) | 78.88 (−0.87) |

It buys minority recall at a small accuracy cost — and, critically, it is **uniquely right
on 4.63%** of test aspects where the plain member is wrong (plain is uniquely right on
5.50%). So it is a *targeted-diversity* member: the log-average can use it even though its
standalone accuracy is lower. **Chapter B's A1 verdict does not transfer** — there, class
weighting drove the `O` tag weight to 0.02 and destroyed extraction on the joint BIO head;
on a dedicated 3-way MASC head there is no `O` tag to destroy.

### C.15 MEASURED NEGATIVE — C11 non-parametric aspect/context memory (kNN) is DEAD
`experts/knn_memory.py`: frozen MiniLM embedding of `aspect + ±w context`, similarity-weighted
vote over the 3179 labelled TRAIN contexts. Attractive on paper — it does no gradient
training, so it *cannot* share the fine-tuned members' failure modes, and Twitter has
recurring entities/stock phrasings with memorable polarity.

Swept window ∈ {3,5,10} × k ∈ {5,15,40} × temp ∈ {0.02,0.05,0.15} (test):
| best cells | acc |
|---|---|
| w=5, k=40, temp=0.15 | **62.87** |
| w=10, k=40, temp=0.15 | 62.68 |
| w=3, k=40, temp=0.15 | 62.30 |
| **majority-class (NEU, 607/1037)** | **58.53** |

Only **+4.3 over the prior**, and accuracy *rises monotonically as k grows and temperature
flattens* — the signature of neighbours carrying no information, with the vote degenerating
into a smoothed class prior. A 62.9 member log-averaged against ~79 members can only dilute.
**Chapter B's one-line dismissal of kNN memory was correct**, and now it is measured rather
than assumed: 3179 aspects is too sparse for retrieval to find sentiment-discriminative
Twitter contexts. Not added to the ensemble.

### C.16 PATCHES C12–C14 — three surgical attacks on the one measured failure
All three change only the *training signal*; none adds a module, and the largest adds one
scalar hyper-parameter. Each is deliberately distinguished from a Chapter-B patch that
already failed, so the results are interpretable either way.

| id | patch | how it differs from the Chapter-B patch that failed |
|---|---|---|
| **C12** | **minority-margin loss** — `relu(m − (z_gold − z_NEU))` applied **only to POS/NEG examples**, nothing on NEU | A1 global class weighting rescales the *whole* distribution. The measured error is **one-directional** (POS/NEG collapse into NEU while NEU recall is already 89.5), so only that boundary is pushed |
| **C13** | **sibling-logit loss** — on the 444 within-tweet different-polarity pairs, `relu(m − [s_A(pA) − s_A(pB)] − [s_B(pB) − s_B(pA)])`, with an instance-grouped batch sampler so siblings are co-batched | B9 SupCon constrained **representations**; this constrains the **decision**, which is where the error lives |
| **C14** | **opinion dropout** — randomly mask SenticNet opinion words lying outside ±4 tokens of the target | no Chapter-B analogue; removes the tweet-level shortcut at the *input* rather than penalising it in the loss |

Verified before launch: the sibling loss returns 1.000 when both aspects collapse to one
polarity, 0.000 when each prefers its own, and 0.000 across different tweets; the
instance sampler yields 420 batches with full coverage and **799 batches containing a
multi-aspect tweet**, so the term actually fires.

**A use for CORSA's dead file.** C14's opinion detector is
`referred_clones/CORSA/CORSA/src/senticnet_word.txt` — the 39,891-entry lexicon that CORSA
ships and never reads. SenticNet scores function words too (`the` = 0.935), so it is
filtered to open-class words (spaCy stopwords + non-alpha + len<3 removed) → **28,914
entries, 1.91 opinion words per training aspect**.

### C.17 ★ THE ENSEMBLE IS SATURATED — and dev is anti-correlated with test AGAIN
Assemblies over increasingly large member pools, all selection-free log-average:
| members | test joint |
|---|---|
| 9 | **70.29** |
| 12 (+3 gated) | **70.39** |
| 12 (+3 class-balanced) | 69.52 |
| ~20 (everything) | 69.62 – 70.10 |

Sweeping a *dev-accuracy inclusion threshold* (a single scalar, far weaker than greedy
subset search) inverts cleanly: `dev≥0` (20 members) → dev 68.18 / **test 70.10**;
`dev≥77.1` (10 members) → dev **70.23** / test 69.19. **Dev says fewer members, test says
more.** Every operating point lands in 70.1–70.4. Combination, member count, member
selection and member weighting are all exhausted; `a` sits at 79.5–81.4 regardless.

### C.18 ★★ NOVEL DIAGNOSTIC — our extractor and classifier are ANTI-correlated in difficulty
| test subset | n | MASC ensemble acc |
|---|---|---|
| all gold aspects | 1037 | 81.20 |
| **extracted by MATE** | 924 | **80.52**  ← this is `a` |
| **missed by MATE** | 113 | **86.73** |

The spans our extractor **misses are EASIER** for the classifier than the ones it finds:
a **−0.68** penalty where MADSC's numbers imply a **+1.84** bonus (`a` 84.18 vs gold-span
accuracy 82.34) — a **2.5-point swing in `a`, which is most of the remaining 2.6 gap**.
Plausible cause: MADSC runs one backbone for both subtasks, so spans its extractor gets
right are spans its classifier also understands; our DeBERTa BIO-CRF extractor and
bertweet/PDQ classifiers are deliberately decorrelated, which is good for the MASC ensemble
and evidently bad for this coupling. Recorded as an open structural finding.

**Recovering the missed spans does NOT pay by itself** (measured): at `cand_thr` 0.3 test
recall rises 89.10 → 90.36 (+13 gold spans) but 21 false positives come with them, so with
`a`=80.5 joint precision *falls* 67.2 → 66.1. The extra recall is only worth having if
something can separate good new spans from junk — which is exactly what **C4** is for.

### C.19 PATCH C4 — annotation-policy span reranker (finally built)
`experts/span_rerank.py`. A binary judge over whole candidate spans ("would an annotator
mark this?"), because Chapter A showed **110/182 dev false positives are
reasonable-but-unannotated entities** ("Regions Bank", "# VMworld") — a policy question a
per-token tagger confidence cannot represent. Trained on **MATE out-of-fold** candidates
(`mate_expert.py --fold`), never on train predictions, which are memorised (~99 F1) and
contain almost no false positives to learn from. Consumed by `assemble.py --rerank`, which
substitutes the judge's score for `mean(1−P(O))` in the τ decision, with `--rerank-mix`
interpolating geometrically between the two.

**Target arithmetic with everything measured:** `joint = MATE@τ × a`. To clear 72.9 we need
roughly **MATE ≈ 89 AND `a` ≈ 82** together (87.4 × 83.4 also works, but 83.4 exceeds every
published gold-span MASC number on t2015 — MADSC 82.34, DEQA 82.10, VLHA 81.50). Our
gold-span accuracy of **81.20 is already 3rd best published**, above DQPSA/CORSA (81.10)
and VLHA (81.50 acc). So the remaining route runs through **MATE precision**, not polarity.

### C.20 MEASURED — C9/C10/C12/C13 mechanism results (bertweet-large, identical recipe)
Judged on whether each mechanism moved the class it was *designed* to move (test):
| member | acc | POS R | NEU R | NEG R | uniquely right vs plain |
|---|---|---|---|---|---|
| plain btwL | **79.75** | 74.13 | 86.16 | 61.06 | — |
| **C9 balanced CE** | 78.88 | **76.34** | 82.54 | **66.37** | 4.63% |
| **C13 sibling-logit** | 76.57 | 72.24 | 80.56 | **67.26** | **5.11%** |
| C12 minority-margin | 77.82 | **70.66 ↓** | 85.17 | **58.41 ↓** | 3.95% |
| C10 sibling-marking | 76.95 | **61.83 ↓↓** | 88.47 | 57.52 ↓ | 4.15% |

* **C9 works** — the designed trade, +2.2 POS / +5.3 NEG recall for −0.87 accuracy.
* **C13 works partially** — +6.2 NEG recall and the highest unique-right share of any
  variant (5.11%), confirming that constraining the *decision* beats B9's SupCon on
  representations. Costs POS/NEU.
* **C12 BACKFIRED** — minority recall went *down* (POS −3.5, NEG −2.7). A hard margin of
  0.5 on the minority-vs-NEU logit gap apparently pushes the model to satisfy the margin on
  already-easy minority cases and abandon the hard ones. Rolled back.
* **C10 BACKFIRED BADLY** — POS recall −12.3. Inserting `< >` around sibling aspects
  disrupts the pretrained encoder's reading of the tweet far more than it helps
  aspect-conditioning. The *idea* may still be right; this *encoding* of it is not.
  Rolled back.

### C.21 ★★ THE MEMBER ROUTE IS CLOSED — mechanisms don't help the ensemble either
Members selected **on dev** (keep a mechanism only if `POS_R + NEG_R` on dev beats the plain
member — 7 of 8 pass), then assembled selection-free:
| assembly | n | dev | **test F1** | `a` |
|---|---|---|---|---|
| core 9 | 9 | 68.62 | 70.29 | 80.40 |
| **core 9 + 3 gated (C7)** | 12 | 68.27 | **70.39** | 80.51 |
| core 9 + dev-kept mechanisms | 16 | 68.09 | 69.91 | 79.96 |
| core 9 + gated + dev-kept | 19 | 68.39 | 70.05 | 80.88 |

Individually valid, dev-validated, mechanistically-confirmed members **still do not move the
ensemble**. Together with §C.7 (OOF stacker loses to log-average) and §C.17 (dev
anti-correlates with test over inclusion thresholds), the entire *polarity-member* axis is
now exhausted at **`a` ≈ 80.4–80.9, joint ≈ 70.3–70.4**. The best configuration remains
**core 9 + the three C7 gated members = 70.39**, and the C7 calibrated gate is the only new
mechanism that added anything at all (+0.10).

**Where that leaves the campaign.** With `a` capped near 80.5 and MATE@τ at 87.43, joint is
pinned at ~70.4. Clearing 72.9 from here requires MATE@τ ≈ 88.5–89 *without* losing `a` —
i.e. **precision on the extractor**, which is what C4 (§C.19) targets and the only untried
axis left. §C.18 says the same thing from the other direction: the spans MATE misses are the
ones MASC handles best, so extraction — not polarity — is where the coupling is broken.

### C.22 ★ C4 SPAN RERANKER WORKS — and dev/test AGREE for the first time
MATE 2-fold OOF (fold0 test 83.90, fold1 similar — deliberately weaker half-data models, used
only to generate honest candidates), then a bertweet-large binary judge over candidate spans:
**dev AUC 0.8102**. Assembly on the high-recall candidate pool (`cand_thr` 0.12), 8 members:

| candidates | mode | rerank-mix | dev | MATE@τ | `a` | **test joint** |
|---|---|---|---|---|---|---|
| argmax, no rerank | span | — | 68.68 | 87.46 | 80.15 | 70.10 |
| hr 0.12 | span | 0.0 (tagger conf) | 68.53 | 87.75 | 80.15 | 70.33 |
| hr 0.12 | span | 0.5 | 69.31 | 87.44 | 80.44 | 70.34 |
| hr 0.12 | span | 1.0 (judge only) | 68.95 | 87.37 | 80.44 | 70.28 |
| hr 0.12 | joint | 0.0 | 68.50 | 87.46 | 80.46 | 70.37 |
| **hr 0.12** | **joint** | **0.5** | **69.48** | **87.54** | **80.55** | **70.51** |
| hr 0.12 | joint | 1.0 | 68.86 | 87.27 | 80.40 | 70.16 |

**Dev selects the test-best cell.** mix 0.5 / joint-mode has both the highest dev (69.48) and
the highest test (70.51), **+0.41** over the identical-member argmax reference — the first
lever in this chapter where dev and test move together instead of inverting. That is exactly
what one expects if the reranker is adding *real* signal rather than fitting noise: the judge
answers a different question (annotation policy) from the tagger (boundary confidence), and
the geometric mix of the two beats either alone (mix 0.0 → 70.37, mix 1.0 → 70.16).

Note also that C3 (joint-product τ) is **positive at every mix** here, unlike on the argmax
pool where its sign was unstable — confirming the §C.19 prediction that the product threshold
only pays when there is a genuinely larger candidate pool to select from.

### C.23 ★★ PDQ-MATE: HURTS as an ensemble member, HELPS as an evidence source
`experts/pdq_mate.py` (built earlier, first run now): BLIP-2 Q-Former + EPE GlobalPointer as
the **extractor**. bertweet-large tower, dev MATE **86.33** (P 87.40 / R 85.29, thr 0.6),
test **85.46** — competitive with the BIO members from a completely different mechanism.

**As a 6th marginal-averaging MATE member it produces the best dev of any combination and a
worse test:**
| MATE members | dev | test P | test R | test F1 |
|---|---|---|---|---|
| PDQ-MATE alone | 86.33 | 86.48 | 84.47 | 85.46 |
| 5 BIO members | 86.49 | 85.00 | 89.10 | **87.01** |
| 5 BIO + PDQ-MATE | **87.31** | 84.99 | 88.43 | 86.67 |

Mechanical cause, not just noise: `pdq_mate` exports *decoded* spans as hard pseudo-marginals
(`m[a,B]=p`, `O=1.0` everywhere else), so a span it misses casts a **hard `O`=1.0 veto**
against five soft CRF votes. Recall duly falls (89.10 → 88.43) while precision does not rise.

**Used instead as one term in a geometric span-evidence product it is clearly positive.**
Combining tagger confidence, the C4 judge, PDQ-MATE's span probability, and the MASC
confidence — **equal weights, no per-signal tuning, only τ dev-tuned**:
| signals | dev | test P | test R | **test F1** |
|---|---|---|---|---|
| tagger only | 68.53 | 69.80 | 70.88 | 70.33 |
| tagger + judge | 69.31 | 70.47 | 70.20 | 70.34 |
| tagger + judge + MASC (C3) | 69.39 | 70.68 | 70.68 | 70.68 |
| **tagger + judge + PDQ-MATE** | 69.39 | 70.78 | 70.78 | **70.78** |
| **all four, equal weights** | 69.36 | 70.82 | 70.68 | **70.75** |
| all four, span-heavy (2,2,1,1) | **69.46** | 70.98 | 70.30 | 70.64 |

Dev spans only 69.31–69.46 and cannot discriminate, so the **reported number uses the
selection-free rule — every available signal, equal weights**: **joint 70.75**
(P 70.82 / R 70.68, `a` 80.55). Running best, up from 70.51 → **gap to MADSC 2.15**.

**Generalisable lesson:** a model whose output shape differs from the ensemble's (hard
decoded spans vs soft marginals) should enter as *evidence in the decision rule*, not as a
member of the average. Averaging silently converts its abstentions into vetoes.

### C.24 THE SELECTION AXIS IS ALSO SATURATED — three negatives in a row
With the C.23 evidence product at **70.75**, every attempt to improve the *selection* fails:

**(a) τ is already near-optimal.** dev-τ 0.330 → test 70.75; oracle-τ 0.295 → 70.94.
τ-selection costs **0.19**. Not the bottleneck.

**(b) 2D acceptance surface (span-evidence × MASC-confidence, 5×5, dev-fit) — WORSE.**
test **69.07** vs 70.75, and dev rejects it too (69.03 vs 69.36). The dev cell precisions are
broadly monotonic in both axes (0.94 bottom-right → 0.13–0.22 top-left), so both signals do
carry information — but discretising into 25 dev-estimated cells overfits more than the
smooth 1-D product gains.

**(c) Judge ensemble — WORSE, despite better individual judges.**
| judges (dev AUC) | test F1 |
|---|---|
| bertweet-large 0.8102 | **70.75** |
| + deberta-v3-large 0.8340 + bertweet-large/s102 0.8246 | 70.54 |
| 3 judges + 2 PDQ-MATE sources | 70.66 |

**(d) More MASC members in the product — no change.** core8 70.75 · +gate 70.66 ·
+pdq2 70.46 · +mech 70.59 · +ALL(15) 70.75.

**Why the ceiling is where it is (measured, test):** the candidate pool holds **760 of 1037**
gold pairs with the polarity already correct, and the current operating point already
captures **733 of those 760 (96%)**. The loss is the ~302 wrong candidates kept alongside,
which split into **≈177 gold-span-but-wrong-polarity** and **≈125 non-gold spans**. A perfect
selector over this pool would score **84.59**, so the information exists — but every
realizable selector (τ, 2-D surface, judge ensemble, learned stacker) lands within ±0.2 of
the simple geometric product. Improving the judge's AUC by +0.02 did not convert into F1.

### ★★ CHAPTER C FINAL STANDING (t2015 test, selection-free rules throughout)
```
MATE (5-member marginal avg, no selection)     87.01     beats MADSC 86.60 / CORSA 86.30 / AoM 86.20
MATE@tau in the evidence product               87.83
a  (polarity on correctly-extracted)           80.55
JOINT   P 70.82   R 70.68   F1 70.75           bar 72.8 / 73.1 / 72.9  ->  margin -2.15
```
Trajectory: 66.41 → 66.95 → 69.73 → 70.29 → 70.39 → 70.51 → **70.75**.
**Beats 15 of 19 t2015 baselines** (incl. CORSA 69.9, TCMT/Vanesa 69.8, SMCAF 69.5, AoM 68.6).
Below: SGBIS 71.1 · DQPSA 71.9 · VLHA 72.5 · MADSC 72.9.

**What it would take, arithmetically:** `joint = MATE@τ × a`, so 72.9 needs
**MATE@τ ≈ 89.6 AND `a` ≈ 81.4** simultaneously (or 87.8 × 83.0, where 83.0 exceeds every
published gold-span MASC number on t2015). Our gold-span MASC accuracy of **81.20 is already
3rd-best published**. Both required gains sit in axes with three independent saturation
measurements each. No single remaining patch in this ledger closes 2.15.

### C.25 ★★ TWO NEW PRODUCTIVE FAMILIES — 70.75 → 71.37 (gap 2.15 → 1.53)

**C2c — PDQ with the full DQPSA objective (ITC + ITM + EPE).** We had been running their
`no_its_and_itm` branch. Adding the two pretrained BLIP-2 stage-1 heads makes members that
are **weaker standalone but better ensemble partners** — the same pattern C9 showed:
| member | standalone test | unique-right vs plain |
|---|---|---|
| pdq_btwL (epe only) | **79.94** | — |
| pdq_btwL + ITC/ITM | 77.53 | 5.21% |
| pdq_twrob + ITC/ITM | 76.57 | **7.33%** (campaign high) |
| pdq_deb + ITC/ITM | 76.86 | 6.17% |
Deviation: DQPSA mines hard negatives from the similarity matrix; at batch 8 that is
pointless, so one shuffled in-batch negative is used instead.

**C15 — PDS (Polarity-Direction Supervision), TARKAN-native, nothing imported.**
Every teacher signal TARKAN has ever used is *relevance* ("is this image relevant?"), which
says nothing about what the evidence **does**. Chapter B measured relevance streams at ≈0 and
Chapter C measured the calibrated gate at +0.10 — the signature of a real but *directionless*
signal. PDS asks the teacher, once offline, whether the evidence shifts sentiment toward the
aspect **more positive / more negative / not at all**.

*Teacher* (`experts/pds_teacher.py`, Llama-3.1-8B 4-bit — 3.2-3B is gated and not granted):
**scored, not generated** — one forward pass reading the logits of single-token options
A/B/C. Deterministic, ~3× faster, and yields a **soft** distribution so teacher uncertainty
stays uncertainty. Result: **POS-shift 687 · NEG-shift 101 · no-shift 2391 (75.2%)**, mean
confidence 0.795.

*First attempt FAILED* (test 75.89 vs 79.75) for a diagnosable reason: with 75% "no-shift"
labels and an L2 pull of `z_full` toward `z_text`, the objective is a **visual-suppression
regulariser**, not a direction constraint. Down-weighting alone did not fix it
(`w_none=0.1` → 75.80).

*Three fixes, all measured:*
| fix | why |
|---|---|
| **zero-init RESIDUAL** `lg_full = lg_base + α·g·δ`, `lg_base` supervised too | starts *exactly* at the strong plain classifier, so a 6.5%-unique-right signal can only correct it, never replace it |
| **signed margin on the DECISION** (`δ[POS]−δ[NEU]`), dead-zone for no-shift | evidence may change the representation freely; only the **sign of its effect** is constrained. The L2-on-representation was the original error |
| **POS/NEG inverse-frequency weighting inside the auxiliary loss only** (0.147 / 1.0) | 687:101 would otherwise teach "images make things positive" — a bias, not a mechanism. The MASC classifier itself is never class-weighted |

| PDS variant | standalone test | ensemble effect |
|---|---|---|
| original (replace classifier, L2, w_none=1) | 75.89 | **−0.09** |
| w_none=0.1 only | 75.80 | — |
| **residual + signed margin, w_none=0.05** | 78.40 | **+0.26** |
| **residual + signed margin, w_none=0** | **78.50** | best PDS |

**The `w_none=0` control settles it: no-shift supervision is HARMFUL, not merely
over-weighted.** The teacher's "no shift" is best read as an *abstention*, not a regression
target.

**Combined (selection-free, all current-generation members):**
```
core8 + ITC/ITM(4) + PDS-residual(3)   n=15
MATE@tau x a = 71.47 P / 71.26 R / 71.37 F1     a = 81.21   <- highest a of Chapter C
bar 72.8 / 73.1 / 72.9   ->   margin -1.53
```
Trajectory: 66.41 → 69.73 → 70.29 → 70.51 → 70.75 → 71.08 → **71.37**.
Dev again fails to discriminate (69.16–69.84 across configurations spanning 70.75–71.37).

### C.26 ⚠ MEMBER-SET SELECTION IS UNSTABLE — read the headline number with this caveat
Scaling both productive families to 22 members made things WORSE, and dev cannot find the
good configuration:
| member set | n | dev | **test F1** | `a` |
|---|---|---|---|---|
| core8 | 8 | 69.36 | 70.75 | 80.55 |
| core8 + ITC/ITM(2) | 10 | 69.84 | 71.08 | 80.88 |
| **core8 + ITC/ITM(4) + PDS-res(3)** | 15 | 69.72 | **71.37** | **81.21** |
| core8 + ITC/ITM(5) | 13 | 69.90 | 70.14 | 79.80 |
| core8 + PDS(9) | 17 | 69.92 | 70.46 | 80.22 |
| **all available** | 22 | **70.08** | 70.50 | 80.22 |
| one-per-(family,tower) | 16 | 69.43 | 70.20 | 79.82 |
| one-per-tower ITC/ITM | 12 | 69.99 | 70.46 | 80.13 |

**No a-priori rule reaches 71.37 and dev ranks it 4th**; dev-argmax over admissible rules
selects the 22-member set → **70.50**. Test spans 70.14–71.37 while dev spans 69.36–70.08
in roughly the *opposite* order, so a good part of that 1.2 spread is selection noise.

**Honest reporting position:** ~**70.5 under strict dev-selection**, **71.37 best measured**.
Quoting 71.37 as "the system's score" would be test-set selection and is not claimed here.
This instability is itself the most robust finding of Chapter C: with n_dev = 1122 and
binomial σ ≈ 1.2, dev cannot resolve differences of the size being chased, which is the same
mechanism that killed the OOF stacker (§C.7), the dev-threshold sweep (§C.17), the 2-D
acceptance surface and the judge ensemble (§C.24).

Also measured: **ITC/ITM on bertweet-base collapses** (test 58.82 — below the 58.53 majority
class). The tower must be strong enough to carry the contrastive objective; ITC/ITM is not a
free upgrade for weak towers.

### C.27 MEASURED NEGATIVES — PDS-v2 (continuous target) and the NEU-escape gate
| variant | dev | test | ensemble effect |
|---|---|---|---|
| PDS-res, signed margin, w_none=0 **(keep)** | 77.90 | **78.50** | **+0.26** |
| PDS-v2 continuous target, bertweet-large | 77.45 | 77.34 | — |
| PDS-v2 continuous target, deberta-v3-large | **79.23** | 77.63 | — |
| PDS-v2 + NEU-escape | 76.56 | 75.80 | — |
| PDS-margin + NEU-escape (isolates the gate) | 78.79 | 76.57 | — |
| adding all four to the best set | 70.01 | **70.59** (vs 71.37) | **−0.78** |

Both refinements fail. **Continuous target:** regressing the residual onto
`q·(P(POS)−P(NEG))` uses more of the teacher's output but ties the student to the teacher's
*magnitude*, and that magnitude is not calibrated — the teacher's mean confidence is 0.795
with a 6.8:1 POS bias, so its scale is a bias to be resisted, not a target to be matched. The
hinge form only asks for the correct **sign**, which is the part of the teacher that is
trustworthy. **NEU-escape:** amplifying the residual when the base leans NEU makes the
correction largest precisely where the base is most often *right* (NEU recall is 89.5), so it
injects errors faster than it fixes them — the gate isolates cleanly (76.57 with the margin
loss alone), so this is the gate's own failure, not an interaction.

**Standing conclusion: the PDS formulation to keep is zero-init residual + signed-margin
hinge + POS/NEG inverse-frequency weighting + no-shift ignored (w_none = 0).**

### ★★★ CHAPTER C FINAL RESULT (t2015 test)
```
MATE  (5-member marginal average, selection-free)      87.01
MATE@tau inside the evidence product                   ~87.9
a     (polarity on correctly-extracted spans)          81.21
JOINT   P 71.47   R 71.26   F1 71.37     best measured (15 members)
JOINT                        F1 ~70.5    strict dev-selection
bar (MADSC)  P 72.8 / R 73.1 / F1 72.9   ->  margin -1.53 (best) / -2.4 (dev-selected)
```
Trajectory: 66.41 → 66.95 → 69.73 → 70.29 → 70.51 → 70.75 → 71.08 → **71.37**.

**Baselines cleared at 71.37 (16 of 19):** JML 64.1 · GMP 49.3 · CMMT 66.5 · VLP-MABSA 66.6 ·
Atlantis 67.3 · M2DF/MultiPoint 67.6 · MCPL-VLP 68.2 · AoM 68.6 · RNG 68.6 · SMCAF 69.5 ·
TCMT 69.8 · Vanesa 69.8 · CORSA 69.9 · **SGBIS 71.1** · (all pre-2023).
**Not cleared:** DQPSA 71.9 · VLHA 72.5 · **MADSC 72.9**.
**The goal — beating every baseline — is NOT met.**

**What would be required, and why no measured lever supplies it:** `joint = MATE@τ × a`, so
72.9 needs **MATE@τ ≈ 89.6 with `a` ≈ 81.4** together. Our gold-span MASC accuracy (81.20) is
already 3rd-best published on t2015, and MATE 87.01 already beats MADSC's own extractor
(86.60). Every axis now carries multiple independent saturation measurements: combiners (4),
member count/selection (5), span selection (4), polarity-member families (3). The two
mechanisms that did work this chapter — ITC/ITM and PDS — each bought ≈ +0.3 and then stopped
scaling.

**Honest remaining options, both untested here:** (a) a Qwen2.5-VL-7B member (~4 h on the T4)
to restore the strong VL member Chapter B relied on; (b) **VLP-MABSA pretrained weights** —
the MABSA-specific vision-language pretraining that every 68–72 baseline inherits and this
pipeline lacks, which Chapter B §7c identified as the root cause of the visual stream
contributing ≈0.

**Environment notes (cost real time, worth recording):**
* Lightning Studio **blocks `venv`** ("max 1 environment") — install into the default conda env.
* transformers ≥4.56 **honours the checkpoint's stored dtype**: `deberta-v3-large` loads as
  **fp16**, so AMP then raises `ValueError: Attempting to unscale FP16 gradients`. Fix:
  `from_pretrained(..., dtype=torch.float32)` and let autocast handle the forward.
* `pgrep -f "<script>"` inside a waiting shell **matches the waiting shell itself** → the
  chain never fires. Wait on an explicit PID (`while kill -0 $PID`) instead.
* Marker (`marker_single`) failed in this Studio (stalls fetching layout models, exit 1);
  `pypdfium2` text extraction was used for the paper instead.
* **PDQ porting bug (found the hard way):** DQPSA deliberately carries **two** tokenizers
  (`IE_tokenizer`, `PQ_former_tokenizer`). Collapsing them works only while the text
  encoder is BERT-vocab; with bertweet (64k vocab) the query ids index the Q-Former's 30k
  BERT embedding table and CUDA dies with a device-side gather assert inside
  `Blip2TextEmbeddings.forward`. Query ids must always come from the BLIP-2 tokenizer.

**C1 seed spread (t2015 test MATE)** — seeds 43/44 came in *below* seed 42, so 3-seed
marginal averaging gained on dev (86.86 → 87.01) and **nothing on test**:
| member | P | R | F1 |
|---|---|---|---|
| seed 42 | 84.14 | 87.46 | **85.77** |
| seed 43 | 83.27 | 86.40 | 84.81 |
| seed 44 | 82.41 | 87.66 | 84.95 |
| 3-seed marginal ensemble | 83.67 | 87.95 | 85.75 |

This is the "seed lottery" Chapter B flagged (§7d: s47 = 84.57 hole), and it means more
seeds is the wrong lever — the recipe is. Hence **C4b** before any new module.

**Assembly, 3 MATE × 3 MASC:** span-τ **joint 66.95** (τ 0.65, MATE@τ 86.18, `a` 77.69,
dev 64.83) vs C3 joint-τ 66.70 (τ 0.54, `a` 77.85, dev 65.30). **C3 wins dev but loses
test here**, the reverse of the 1-MATE case — so C3's sign is not stable at this member
count. Recorded as inconclusive rather than a win.

**Target arithmetic (updated for the MADSC bar):** need MATE ≈ 88.5 **and** `a` ≈ 82.4
(or MATE 89.0 × `a` 81.5). Chapter B reached MATE 88.0 × `a` 81.3 = 71.26.
C2 targets `a`; C1+C4 target MATE precision (the binding constraint, §7d); C3 targets the
product directly. **Honest prior: rebuilding alone caps at ~71.3. Clearing 72.5 requires C2 to
deliver a genuinely decorrelated ≥79 member AND C3/C4 to convert it — uncertain, but these are
the levers Chapter B never pulled.**

---

# ===== CHAPTER D — THE FROZEN POOL (2026-08-13) =====

Chapter C ended with "every axis is saturated", which was true of everything it measured
but rested on a diagnosis that turns out to be **wrong in its most important claim**.
Chapter D re-derives the target from a pool that is built once and never rebuilt, so that a
negative result can finally be attributed to the *selector* rather than to a pool that
silently changed underneath it.

New code: `experts/pool.py` (materialise the pool + every decision signal + the gold label
into one flat table) and `experts/anatomy.py` (read that table; measure what is reachable).
This also discharges the standing "fold the evidence product into a single command" debt —
the §C.23 product is now `anatomy.py`'s printed baseline, not a transcript script.

## D.1 ★★★ THE LOSS DECOMPOSITION — polarity loses TWICE what extraction loses

Frozen pool = 5 MATE members, `cand_thr` 0.12, 15 MASC members log-averaged, C4 judge,
PDQ-MATE span evidence. Of the **1037 t2015 test gold pairs**:

| | n | |
|---|---|---|
| span never entered the candidate pool | **92** | extraction loss |
| span in the pool, **polarity wrong** | **186** | polarity loss |
| recoverable (span present, polarity right) | **759** | pool ceiling R 73.19, perfect-selector F1 **84.52** |

**This reverses §C.19 and §C.21**, which concluded "the remaining route runs through MATE
precision, not polarity" and drove eight subsequent experiments. That conclusion came from
the identity `joint = MATE@τ × a` plus the observation that our gold-span `a` (81.20) is
already 3rd-best published — true, but it prices the two losses as if they were comparable.
Counted directly they are not: **polarity throws away 2.02× more gold pairs than extraction
does.** Every §C.24 selection experiment was therefore aimed at the smaller of the two pots.

Confusion on the 945 in-pool gold spans is not diffuse either — it is one direction:

| gold | n | →NEG | →NEU | →POS | recall |
|---|---|---|---|---|---|
| NEG | 101 | **58** | 37 | 6 | **57.4** |
| NEU | 552 | 19 | **487** | 46 | 88.2 |
| POS | 292 | 3 | 75 | **214** | 73.3 |

**158 of the 186 polarity errors are minority↔NEU.** NEG recall 57.4 is the single worst
number in the campaign.

**Baseline reconciliation (important for reading every D number).** The frozen pool
reproduces §C.23's core-8 configuration on **dev to within 0.01** (69.37 vs 69.36) and lands
0.57 lower on test (70.18 vs 70.75); the 15-member set gives dev 70.07 / test 70.37 against
§C.25's 71.37. The whole difference is τ tie-breaking — a 0.005 grid on F1 here versus a
0.01 grid on the per-cell margin there — which is precisely the §C.26 instability
(test spans 1.2 with dev flat). Nothing is broken and nothing is being chased: **§C.26
already ruled that ~70.5 is the defensible dev-selected figure and 71.37 is not
dev-selectable, and 70.37 is that same number computed without a tie-break lottery.**
Chapter D therefore measures everything against **70.37**, so the honest gap to the bar is
**+2.53**, not +1.53.

## D.2 ★★ THE FEATURES ARE EXHAUSTED — proved by letting a selector CHEAT

§C.24 called selection saturated from four dev-fitted negatives, leaving open the reading
that dev noise (σ ≈ 1.2 at n=1122) was the culprit and more fitting data would fix it. It
is not. A logistic regression over 11 features (tagger conf, judge, PDQ-MATE, MASC max /
margin / entropy / argmax, sentence-relative score, candidate count, span length, overlap),
**fitted directly on the test set**:

| selector | dev | **test F1** |
|---|---|---|
| §C.23 equal-weight geometric product | 70.07 | **70.37** |
| logistic fitted on DEV (honest) | 70.21 | 70.83 |
| **logistic fitted on TEST (cheating — an upper bound)** | 69.65 | **70.77** |

**Cheating does not beat the honest fit.** The ceiling of these features is ~70.8 against a
perfect-selector ceiling of 84.52, so 13.7 points of the gap are invisible to every signal
the system currently produces. No amount of extra fitting data, calibration, or combiner
sophistication recovers them. **Selection is closed for a reason stronger than dev noise:
the information is absent, not merely hard to estimate.** Signal AUCs on test for
"is this pair correct" — tagger .689, judge .671, PDQ-MATE .681, MASC max .659 — and note
MASC max is AUC **.513** for "is this a gold span", i.e. polarity confidence carries almost
no validity information, which is exactly why the geometric product plateaus.

**Also measured and dead (free, CPU-only):**
* **Decode-time class bias.** Multiplying the NEU column by β before argmax, dev-selected:
  dev is flat at 69.8–70.15 across β ∈ [0.2, 1.0] while test wanders 70.1–71.0. The 2-D
  version (separate NEG/POS boosts) is the same story. Dev cannot select it; the honest
  reading is ±0. This closes the last cheap route at the minority↔NEU boundary that §C.12
  and §C.20 attacked from the training side.
* **Overlap filtering.** Zero overlapping candidates exist — BIO decoding cannot produce
  them. The check is void, not negative.
* **Longer-span filtering.** Dropping len≥4 candidates removes 22 spans of which 15 are
  correct. Strictly harmful.

## D.3 ★ POOL RECALL — 0.12 is already the optimum, but a SECOND MECHANISM adds real spans

Lowering `cand_thr` below 0.12 does **not** buy recall — it merges adjacent spans into
wrong longer ones, so recall *falls* while the pool inflates:

| cand_thr | test cand | pool P | pool R |
|---|---|---|---|
| 0.00 (argmax) | 1087 | 85.00 | 89.10 |
| **0.12** | 1178 | 80.22 | **91.13** |
| 0.05 | 1218 | 77.09 | 90.55 |
| 0.01 | 1362 | 68.87 | 90.45 |

What *does* work is a **different extraction mechanism** proposing candidates the BIO
marginals cannot express. §C.23 established PDQ-MATE must not enter as an averaging member
(its hard `O`=1.0 abstentions become vetoes) and works as evidence instead; this is its
**third role — a candidate source**:

| pool | test cand | pool P | pool R |
|---|---|---|---|
| BIO 0.12 | 1178 | 80.22 | 91.13 |
| **BIO 0.12 ∪ PDQ-MATE (2 towers)** | **1276** | 75.94 | **93.44** |

**+24 gold spans reachable for 98 junk candidates.** Precision of the pool is irrelevant —
selection is downstream — so this strictly raises the ceiling. `experts/emit_spans.py`
gained `--union`.

## D.4 THE ARITHMETIC, RESTATED HONESTLY

At the frozen operating point: 1032 kept = 728 correct + 179 gold-span/wrong-polarity + 125
non-gold. To reach **72.9** from there, exactly one of:

* **drop 72 of the 304 false positives** at zero cost to true positives (a 24% FP cut — but
  D.2 says no available signal ranks them), **or**
* **flip 27 of the 186 wrong polarities** (a 15% error cut on that subpopulation).

Flipping is worth ~2.7× dropping, because a flip fixes precision *and* recall. And the
polarity pot is the larger one (D.1). So **the entire remaining campaign should be aimed at
`a`, which is where Chapter C stopped aiming after §C.19.**

Required `a` ≈ **83.5**. Published t2015 gold-span MASC: MADSC 82.34 · DEQA 82.10 · VLHA
81.50 · DQPSA/CORSA 81.10 · **ours 81.20**. So the bar is above every published number, and
Chapter C already measured that no rearrangement of ~200M-parameter encoders reaches it:
class-balanced CE (§C.20), minority margins (§C.12), sibling conditioning (§C.10/C13), PDS
(§C.25/C.27), ITC/ITM (§C.25), 15-member ensembling, gating (§C.7), kNN memory (§C.15).
Every one of those is a re-arrangement of the same model class — and every baseline in the
table additionally carries MABSA-specific vision-language pretraining this pipeline lacks.

**Conclusion: the only honest lever left is a stronger model class, disclosed as such.**

## D.5 PATCH D1 — MASC as a QLoRA'd 8B decoder (`experts/masc_llm.py`)

*Status: running.* PAPER-DISOBEYING and explicitly labelled so. Not copied from DQPSA,
CORSA or MADSC — none of them use a decoder LLM for polarity.

Llama-3.1-8B-Instruct, 4-bit NF4, LoRA r=16 on all seven projections (42M trainable,
0.52%). Two choices make it a classifier rather than a generator:

* **Restricted-vocab scoring in TRAINING as well as inference.** One forward pass; read the
  logits of the single-token options A/B/C at the final position and softmax over just
  those three. The training objective is *literally the inference computation* — no
  generation, no format drift — and the output is a calibrated 3-way distribution that
  drops into the existing log-average ensemble unchanged (`probs_*.npz`, same keys/shape).
  This is the same scored-not-generated trick §C.25 used for the PDS teacher, promoted from
  offline labelling to the student itself.
* **Aspect marked in place** with `[ ]` (Chapter B, B2) — t2015 has tweets where one
  surface form appears twice with different gold polarity, so "Target: X" is ambiguous.

The image enters as its BLIP caption, exactly as the teacher sees it, so no new vision
weights enter the system. Measured cost on the T4: **11.2 s/opt-step at batch 2 × accum 8,
597 steps → 1.86 h** for 3 epochs.

**MEASURED.** dev gold-span 76.83 → 78.79 → **78.88** (converged; train loss reached 0.018,
i.e. the 8B memorised all 3179 training aspects, so the binding constraint is data, not
epochs — `--init-adapter` was built to extend the run and is not worth using).

| | gold-span test acc |
|---|---|
| best encoder member (pdq_btwL_s47) | 79.94 |
| **8B decoder member** | **80.04** |
| 15-encoder log-average ensemble | 80.91 |

**An 8B decoder fine-tuned on this task ties a 355M encoder.** That is the first honest
surprise of Chapter D and it caps what any "bigger model" argument can buy here.

## D.6 ★★★ THE 8B IS GENUINELY DECORRELATED AND STILL CONVERTS TO NOTHING

This is the sharpest version of the wall the campaign keeps hitting, because for once the
diversity is unambiguous and its *shape* matches the diagnosis:

| | |
|---|---|
| 8B uniquely RIGHT (ensemble wrong) | **49 aspects (4.73%)** |
| 8B uniquely WRONG (ensemble right) | 58 (5.59%) |
| **oracle of the two** | **85.63** |
| best fixed log-space mixing weight | 81.00 (vs 80.91 — noise) |

Per class, the 8B does exactly what §D.1 asked for: **NEG recall 69.0 vs the ensemble's
61.1 (+7.9)**, POS 75.1 vs 73.8, paid for with NEU 84.7 vs 88.3.

So the information is there, it is the *right* information, and **no realizable rule
extracts it.** All three combination modes, selected on dev and reported on test:

| rule | dev | **test** |
|---|---|---|
| ensemble alone (w=0) | 80.12 | **80.91** |
| selective NEU-boundary gate (best dev: w 0.7, δ 0.7) | **81.64** | 80.62 |
| per-class weights (NEU 0.2 / minority 0.3) | 81.19 | 80.52 |
| NEU-escape, 8B only, conf > 0.9 | 80.48 | 81.00 |

**The gate wins dev by +1.52 and loses test by −0.29.** Dev and test invert for the fifth
independent time (§C.7 OOF stacker, §C.17 threshold sweep, §C.24(b) 2-D surface, §C.24(c)
judge ensemble, now this). The selective gate was specifically argued to differ from the
failed §C.27 NEU-escape because it hands the decision to a *different model class* rather
than amplifying a residual built from the same features. That argument is sound and the
result is still negative — which localises the cause precisely:

> **The bottleneck is not member quality, member diversity, or combiner form. It is that
> n_dev = 1122 with binomial σ ≈ 1.2 cannot resolve the ~0.5 differences being chased, so
> every combiner that is fitted is fitted to noise.** Adding a better member does not fix a
> selection problem, and Chapter C's §C.7 already showed that moving the fit to OOF train
> data does not fix it either.

Recorded as the strongest form of the §B.8 error-correlation ceiling.

## D.7 MEASURED NEGATIVE — option-order TTA

Averaging the two option orderings (`--tta`) to cancel the decoder's position bias:
dev **78.88 → 78.43**, test **80.04 → 79.56**. Consistently −0.5.

Diagnosable, and worth keeping as a general lesson: option-order averaging is a
**zero-/few-shot** debiasing technique. Once the adapter is fine-tuned on a fixed
ordering, "A = negative" is part of what it learned, so the reversed rendering is
*off-distribution for the adapter* and averaging it in injects noise rather than
cancelling bias. Rolled back.

## D.8 ★★ MEASURED NEGATIVE — the 8B in the JOINT metric, and the union pool

§D.6 measured the 8B on gold-span accuracy. The joint metric is the real objective, and
the member's confidence also enters the selection score there, so it could in principle
pay even while flat on accuracy. It does not.

**At constant pool (`cand_thr` 0.12), sweeping the group-mixing weight `w`:**
| w on the 8B | dev | **test** |
|---|---|---|
| **0.00 (encoders only)** | 70.07 | **70.37** |
| 0.20 | 70.64 | 70.52 |
| **0.30 — dev-best** | **70.82** | **69.63** |
| 1.00 (8B only) | 68.78 | 69.58 |

Dev says include it (+0.75); test says it costs **−0.74**. Seventh independent inversion.

**The union pool (§D.3) is also negative**, despite raising the reachable ceiling:
| pool | dev | **test** | MATE@τ |
|---|---|---|---|
| BIO 0.12, 15 members | 70.07 | **70.37** | 87.68 |
| BIO ∪ PDQ-MATE, 15 + 8B, dev-selected | **70.86** | **68.91** | 85.99 |

MATE@τ falls 87.68 → 85.99: the 98 extra candidates are junk that the selector cannot
filter, which is exactly what §D.2 predicts — **raising pool recall only pays if selection
can exploit it, and selection is dead.** The +24 reachable gold spans never arrive.

**So `--union` is a correct mechanism with a negative measured effect, and is not used.**

*Both negatives re-measured after `queue26b` restored full member coverage* (five members
— 2 PDQ, 3 PDS — had failed a strict `state_dict` load during the re-score because their
checkpoints predate the ITC/ITM heads and the NEU-escape `beta`; see the `--score-only`
guards in `pdq.py` / `masc_pds.py`). With every member scoring every candidate: union pool
best test **69.82**, dev-selected **68.91**; 0.12 pool + 8B best **70.52**, dev-selected
**69.63**; **0.12 pool encoders-only 70.37**. The incomplete coverage was not the cause —
both conclusions stand unchanged.

### ★★ WHERE CHAPTER D LEAVES THE TARGET
`joint = MATE@τ × a`. At our MATE@τ = 87.68, clearing 72.9 requires **`a` = 83.1**.

| | gold-span MASC acc |
|---|---|
| our 15-member ensemble | 80.91 |
| our best single member (8B decoder, 355M encoder) | 80.04 / 79.94 |
| **best published on t2015 (MADSC)** | **82.34** |
| **required** | **83.1** |

Clearing the bar therefore requires **beating the published SOTA polarity model by ~0.8
and converting all of it**, on a T4, without the MABSA-specific vision-language
pretraining every baseline in the table inherits. Chapter D adds five independent
saturation measurements to Chapter C's, and every one of them concerns *rearranging
signals already in the system*.

## D.9 ★★ THE IMAGE CARRIES 3× THE SIGNAL AND MAKES THE MEMBER WORSE

`experts/masc_qwenvl.py --counterfactual`. Qwen2.5-VL-7B (4-bit, T4) sees the **original
image**, and the teacher performs the intervention itself rather than being asked to judge
it: `delta = P(y | tweet, aspect, IMAGE) - P(y | tweet, aspect)`, sign only (§C.27).

**The pixels do carry far more signal than the caption.** Same 3179 training aspects:

| teacher | POS-shift | NEG-shift | no-shift | POS:NEG skew | aspects moved |
|---|---|---|---|---|---|
| Llama-3.1-8B on BLIP captions (§C.25) | 687 | 101 | **2391 (75%)** | **6.8 : 1** | 25% |
| **Qwen2.5-VL on pixels** | 1394 | 644 | 1141 (36%) | **2.16 : 1** | **76%** |

Both numbers move the right way. The near-3× increase in evidence-bearing aspects confirms
the caption was the bottleneck, and the collapse of the POS:NEG skew from 6.8:1 to 2.16:1
answers §C.25's own objection to the old teacher (a 6.8:1 skew "would teach *images make
things positive* — a bias, not a mechanism").

**And every member trained on it is worse.** Identical recipe, identical loss, identical
hyperparameters — only the teacher's direction labels differ:

| PDS member (test gold-span) | caption teacher | **Qwen-VL, shift-floor 0.05** | Δ |
|---|---|---|---|
| bertweet-large | 78.50 | 77.24 | **−1.26** |
| deberta-v3-large | 76.76 | 75.60 | **−1.16** |
| roberta-large | 79.27 | 78.21 | **−1.06** |

A uniform **≈ −1.1 across three architectures** is a property of the labels, not of any
tower. Reading: **76% of aspects "moving" is too many.** Much of `delta` is the model
reacting to an image being present at all rather than to aspect-specific sentiment
evidence, so a 0.05 floor promotes noise to direction — and §C.27 already measured that bad
direction supervision actively hurts (that is exactly why `w_none = 0` won).

**Generalisable lesson, and the sharpest one in Chapter D:** *more teacher signal is not
better teacher signal.* The caption teacher's 75% abstention rate looked like a defect and
was partly a virtue — abstaining is what kept its 25% of direction labels clean. An
evidence teacher's value is set by the precision of its non-abstentions, not by its
coverage.

`--shift-floor` / `--shift-temp` were first guesses and the floor is the obvious suspect;
a sweep at 0.20 / 0.35 is queued (`runs/queue30.sh`). `masc_qwenvl.py` now caches the raw
arms `p_img` / `p_txt` so `experts/remap_direction.py` can re-threshold on CPU — the first
run saved only the mapped labels, which cost one extra teacher pass.

Two sources of genuinely new information remain untested:
1. **Qwen2.5-VL on the original pixels** (§D.5 `masc_qwenvl.py`) — every multimodal signal
   so far has passed through a BLIP caption.
2. **VLP-MABSA pretraining**, reachable via AoM's official checkpoint already working in
   `graft/` (t2015 68.42 reproduced) — the one component class carrying information no
   member here has. Needs the legacy env (py3.8 / torch1.13 / transformers3.4).

---

# ===== CHAPTER A — T4 ERA (kept verbatim; original title below) =====

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

---

# ===== CHAPTER B — A100 DECOMPOSITION SESSION (2026-07-11 → 07-14) =====

Recovered into this file on 2026-08-12 (it had never been committed). **Best: joint F1 = 71.26**
on t2015 (from 66.4 on arrival), MATE 88.0 × `a` 81.3. Gap to bar 1.24. R6-clean (dev-selected),
R9 canonical splits. All numbers measured.

## B.1 THE REFRAME — the whole session's key idea
`joint_F1 ≈ MATE_F1 × MASC_acc_on_extracted` holds to **±0.1 on every row of the paper** and on
every one of our runs (verified exact to the decimal: `87.05 × 0.7565 = 65.85`). So the Table-1 bar
decomposes into two independently measurable, independently attackable sub-bars. A MATE run costs
minutes; chasing *joint* directly mixes the signals and makes every run expensive and ambiguous.

**Binding-constraint corollary (§7d).** The bar is per-cell, so with `joint_P ≈ MATE_P × a`:
| `a` | MATE_P needed (>72.3) | MATE_R needed (>72.7) |
|---|---|---|
| 0.83 | **87.1** | 87.6 |
| 0.85 | 85.1 | 85.6 |
| 0.87 | 83.1 | 83.6 |
After the B1 recall fix, **MATE precision became the short leg** — and every +1 of `a` cuts ~2
points off the MATE_P requirement, so MASC does double duty.

## B.2 THE MATE BUG (B1) — found, fixed, confirmed by measurement ★
`evaluate.py:decode_word_tags` took `argmax` over all **7** joint tags (B/I × POS/NEU/NEG + O) and
only *then* collapsed polarity. When a word is clearly an aspect but its polarity is uncertain the
mass splits across its three variants and **O wins**:

    P(O)=0.35  vs  P(B-POS)=0.25, P(B-NEU)=0.22, P(B-NEG)=0.18   -> argmax = O
    but  P(B-*) = 0.65.   The word is dropped and the WHOLE SPAN is lost.

Verified on exactly that vector: old decode → `[]`, marginalized decode → `[(0,1,'POS')]`.
**Fix (obeying — the paper never specifies a decode rule):** marginalize polarity out, decide
O/B/I, then read polarity off the winning group. B1 also removes the leak at *training* level via a
dedicated O/B/I head.

**Measured MATE ladder (t2015 test):**
| run | head | backbone | P | R | **F1** |
|---|---|---|---|---|---|
| old champion | 7-tag joint | DeBERTa-v3-**large** | — | — | 85.40 |
| B1 | O/B/I + CRF | DeBERTa-v3-**base**, on CPU | 83.65 | 87.85 | 85.70 |
| **B1** | **O/B/I + CRF** | **DeBERTa-v3-large** | 85.34 | **88.72** | **87.00** |
| B1 + 4-member ens, τ=0 | | | 85.08 | 89.10 | 87.05 |
| **B1 + ens + dev-tuned τ=0.70** | | | **86.84** | 87.85 | **87.34** |

A **base** model on **CPU** beat the old **large** model → the gain is the *head*, not capacity, and
it lands exactly where predicted (recall). This retired the "two unrelated models both cap at 85 ⇒
capacity ceiling" reading: it was a shared **decode** bottleneck. 87.00 already beats M2DF 86.3,
AoM 86.2, Atlantis 86.1, CMMT 85.9, VLP-MABSA 85.7 — only DQPS (87.7) remained above.

**Pre-cleared suspects (measured, all retired):** word-level O/B/I oracle = **100.00** on every
split; truncation loss @max_len 160 = **0/1037** spans; tokenizer round-trip **100% lossless**
(0/11,308 spans; longest tweet 69 subtokens). Data/tokenization was never the bottleneck.

## B.3 THE ENCODER PATH IS DEAD (§7c) — the session's most important negative result
| run | arch | MASC Acc | macF1 |
|---|---|---|---|
| old student (A7-rich head) | DeBERTa-lg + dedicated ASC head | 78.5 | 73.4 |
| **TARKAN stage-2 as specified (relevance gate + KAN)** | DeBERTa-v3-lg + CLIP, pooled | **75.41** | 71.44 |
| + frozen CLIP | | 76.76 | — |
| Qwen2.5-VL-7B (joint generation, by-product) | 7B VLM | **80.76** | — |

**The paper's own stage-2 (relevance gate + KAN fusion) LOST to a plain head (75.41 vs 78.5)**, and
75–78 is what a *text-only* model scores ⇒ **our visual stream contributes ≈ 0**. Diagnosis: AoM
(80.2) / M2DF (78.9) / VLHA (81.5) all inherit **VLP-MABSA's MABSA-specific VL pretraining**; that
pretraining, not the fusion module, buys the visual stream its 3–4 points. No fusion-head tuning on
a from-scratch CLIP+DeBERTa trunk recovers it. **This retires the entire "tune the fusion" family.**

## B.4 MASC ladder on the 7B (§7g, §7o) — and where it caps
| MASC (t2015) | dev Acc | test Acc |
|---|---|---|
| stage-2 encoder (relevance+KAN) | 74.15 | 75.41 |
| Qwen2.5-VL-7B ep1 | 77.54 | 77.53 |
| **Qwen2.5-VL-7B ep2** | **78.97** | **79.17** |
| + hi-res (max_pixels 100k→200k) | **80.04** | ~79 |
| verbalizer / vision-tune / EMA / rationale variants | 79–80 | 78.2–79.2 |
| 8-member ensemble (`a`, on extracted spans) | — | **81.19–81.30** |
| published SOTA | — | VLHA 81.7 · TCMT 81.4 · DQPSA 81.1 · AoM 80.2 |
| **needed** | — | **~82.9** |

Three bugs cost a GPU slot each to find: (1) must call the **outer**
`Qwen2_5_VLForConditionalGeneration.forward` — the inner `Qwen2_5VLModel` is the text decoder and
rejects `pixel_values`; (2) bf16 head vs **fp32** pooled state (`prepare_model_for_kbit_training`
upcasts the final norm); (3) the `requires_grad` checkpoint warning is the frozen vision tower —
harmless, grads do flow.

## B.5 END-TO-END LADDER (all dev-selected, R6-clean)
| system | MATE | `a` | **JOINT** |
|---|---|---|---|
| single 7B MASC | 87.48 | 78.01 | 68.25 |
| + 5-member MASC ens | 87.49 | 79.65 | 69.68 |
| + 8-member curated ens (4 Qwen + 4 text), uniform mean | 87.66 | 80.7 | 70.4 |
| **+ log-avg combiner + dev-tuned τ** | **87.66–88.0** | **81.19–81.30** | **71.26** |

**Honest correction logged at the time:** an earlier 70.9 was quoted off Qwen's *mid-training* test
peaks (79.4); the dev-selected checkpoint (correct per R6) tests at 78.2 → real `a` = 78, joint
68.25. Also: a 3-member MATE ensemble scoring 87.48 was **test-set peeking**; dev-greedy selection
keeps only DeBERTa-large and its honest test score is 87.00. `--select-on-dev` added to enforce it.

**Ensembling rules that survived measurement:** uniform-mean-ALL dilutes (70.4); greedy dev-subset
overfits (dev 82 → test 69); the robust rule is **ALL strong Qwen + fixed diverse text, curated
(not dev-selected), combined by log-avg**, with a dev-tuned τ on the bar-margin objective (F1 alone
is blind to a P/R imbalance that fails a cell).

## B.6 METRIC RECONCILIATION (§5) — resolved, in the safe direction
The old claim *"official AoM ckpt on our pipeline: 68.42"* was **wrong** — that run used AoM's own
harness and metric. Scoring AoM's **own dumped predictions** with our `metrics.py`: **joint 68.19**
(P 67.45 / R 68.95) on exactly **1037** gold pairs, vs AoM's harness 68.42 and published 68.6.
⇒ **our metric is ~0.2–0.4 STRICTER than the baselines' own, never more lenient**, so any margin we
claim over 72.5 cannot be a lenient-scorer artifact. Our gold is canonical: 100% polarity agreement
with AoM's gold on all 1037 test / 1122 dev aspects, span mapping a bijection. Residual gap is BART
beam-search nondeterminism under different batch padding. **These dumps survive in `graft/` and are
the regression gate for the Chapter-C rebuild (C0).**

Also logged: t2017 has 1 exact train/test duplicate tweet (+3 near-dups) **in the original
release** — every Table-1 baseline carries it; document, don't "fix" (R9). t2017 train is 3560
unique aspects, not 3562 (`train.tsv` duplicates one tweet). t2015 train∩test = 0; cross-dataset
overlap = 0. `data.py` image fallback returns raw `torch.zeros` (not a CLIP-normalized black image)
behind a bare `except` — latent silent-failure landmine, not firing (0 missing images).

## B.7 KG EVIDENCE (R3) — built, and it validates the paper's filter by being noisy raw
`scripts/build_kg_evidence.py`: t2015 **83.8% hit, avg 8.0 triples** (paper 68.4% / 8.7 — right
regime). Rendered raw the triples are mostly noise: *Oklahoma* → "follow Causes follow";
*Lydia* → "crazy Synonym brainsick [positive]" (about the opinion word, wrong polarity);
*Barbara Hepworth* → "barbara_hepworth Synonym barbara_hepworth". So injecting *unfiltered* KG would
HURT — which is **exactly the case for the teacher-guided filter (R3)**. KG only pays off *with* the
LLM teacher.

## B.8 THE DEFINITIVE NEGATIVE (§7v / §7x) — the ceiling is error-correlation
| ensemble | JOINT |
|---|---|
| **curated 8 (4 Qwen + 4 text)** | **71.26** |
| + 2 rationale-distilled members (**best singles**: test 79.2, macF1 74.6) | 71.17 |
| + caption-augmented member | 71.17 |
| + LLaVA-1.5-7B member (single test 78.6) | 71.04 |
| top-6 MATE (indiv 88.0) | 71.11 |

The strongest single members of the session **hurt** the ensemble. Correlation measured directly:
LLaVA vs the Qwen block **both_wrong = 0.149 vs independence-expectation 0.043 → ratio 3.43**.
Different VL *family* ≠ different errors: the hard t2015 cases are **data-intrinsic shared blind
spots**. Oracle over 8 members = **92.7–93.4** (and LLaVA is uniquely-right on **2.70%** where all
Qwen fail), so complementary signal genuinely exists — but every realizable extractor overfits dev:
confidence-gate dev 82.17 → test 80.42; reweight-all dev 82.17 → test 80.52; max-conf route dev
81.37 → test 79.75; per-class log-bias dev 80.84→81.73 but test 81.20→**80.62**. Root cause:
**member confidence ≠ correctness on the disagreement cases.**

**Conclusion carried into Chapter C:** every dev-tuned lever gains ~+0.9 dev and does not transfer
(dev/test gap ~1.2 < dev's binomial σ ~1.2). Combiners are DONE at test `a` ≈ 81.2 / joint ≈ 71.2.
Only **new information** or a **mechanism-decorrelated** member can move it — which is precisely
what Chapter C's C2 (PDQ/BLIP-2) tests, with C3/C4 as the conversion levers.

## B.9 FILE REGISTRY — the rebuild spec (all of this is GONE; recreate under `experts/`)
| id | file | what it must do |
|---|---|---|
| B1 | `experts/mate_expert.py` | O/B/I + word-level CRF extractor; marginalized decode; multi-seed; emits word-level tag marginals |
| B2 | `experts/masc_expert.py` | Qwen2.5-VL MASC; aspect marked **in place**; verbalizer or MLP head; `--tune-vision`, `--rationale`, `--ema`, `--gpu-gib` |
| B3 | `experts/assemble.py` | 2-stage assembly; MATE ensembling by averaging **word-level tag marginals** (not voting on decoded spans); **log-avg** MASC combiner; dev-tuned τ on the margin objective; `--select-on-dev`; `--hier-combine` |
| — | `experts/masc_enc.py` | encoder MASC (relevance-gate + KAN, R2/R4); `--no-img`, `--captions`, `--supcon`, `--ema` |
| — | `experts/common.py` | shared canonical data layer (MATE/MASC examples, scorers) |
| B10 | `experts/gen_rationales.py` | teacher writes a 1-sentence rationale per aspect |
| — | `experts/vl_masc.py` | generic cross-family VL MASC (`AutoModelForImageTextToText`) |
| infra | `scripts/gpu_holder.py` etc. | shared-GPU claim/hold (live-tensor ballast — `empty_cache()` leaks a reserved-but-cached claim; a live tensor cannot be reclaimed). **Not needed on the single dedicated T4.** |
| — | `RULES.md` | paper-essence invariants R1–R10 + the bar |

**Assets built in Chapter B that were also lost:** 3179 teacher rationales (`data/rationales/`),
3502 BLIP captions (`data/captions/`), KG 5.11M triples (`data/kg_index/kg.sqlite`), KG evidence
cache (`data/kg_evidence/`). Regenerate only if a lever needs them (all measured ≤ 0 for the
ensemble; KG/rationales are paper-narrative assets, not bar-breakers).
