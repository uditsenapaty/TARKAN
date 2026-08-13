# Session Handoff — TARKAN Chapter D: freeze the pool, re-derive the target, exhaust the model-class levers

## Where it started
Chapter C ended at "every axis is saturated" with a best-measured 71.37 (~70.5 under strict
dev-selection) against the MADSC bar of 72.9, and a standing goal to beat every t2015
baseline. Two deferred items were named: fold the §C.23 evidence product into one command,
and choose between a Qwen-7B member and VLP-MABSA weights for the next GPU spend. Hardware
unchanged: one Tesla T4 16 GB, fp16 only.

## Decisions locked + what shipped
- **Everything is committed and pushed** to `https://github.com/uditsenapaty/TARKAN.git`
  main. Chapter D is commits `bc92e56 … 3447bdf` plus the final standing.
- **The pool is now frozen** (`experts/pool.py` → `pools/`, read by `experts/anatomy.py`
  and `experts/decide.py`). Every §C.24 selection experiment had silently rebuilt the pool
  underneath itself; now a negative is attributable to the selector. This also discharges
  the "one command" debt — `decide.py` applies the whole rule with two fitted scalars.
- **The baseline was reconciled downward and this matters.** The frozen pool reproduces
  §C.23's core-8 on **dev to 0.01** (69.37 vs 69.36) and 0.57 lower on test, entirely from
  τ tie-breaking. §C.26 had already ruled 71.37 non-selectable. **Chapter D measures
  against 70.37, so the honest gap was +2.53, not +1.53.**
- **FINAL RESULT: joint F1 70.43** (P 70.46 / R 70.40, MATE@τ 87.70, `a` 80.31), from the
  only rule with no fitted weights — all 19 members equal-weight, τ tuned on dev. It is
  simultaneously the **dev-best and test-best** configuration of the chapter.
  **Clears 15 of 19 baselines. SGBIS 71.1 / DQPSA 71.9 / VLHA 72.5 / MADSC 72.9 remain
  above. The goal is NOT met.**

## The four findings (this is the session's actual output)
1. **§D.1 — polarity loses 2.02× what extraction loses.** Of 1037 test gold pairs: 92 lost
   to extraction, **186 to polarity**, 759 recoverable. This *reverses* §C.19, which had
   directed the eight experiments after it toward MATE precision. NEG recall is 57.4 and
   158 of the 186 errors are minority↔NEU.
2. **§D.2 — a selector fitted directly ON TEST cannot beat the honest one** (70.77 vs
   70.83; geometric product 70.37), against a perfect-selector ceiling of 84.52. Selection
   is closed because **the information is absent, not because dev is noisy.**
3. **§D.6/D.8/D.10 — the ensemble is saturated, three independent ways.** A member with 49
   uniquely-right aspects and an 85.63 two-model oracle adds nothing; a 1.1-point swing in
   member quality moves the joint by 0.03. **Seven dev/test inversions** localise the
   cause: n_dev = 1122 with σ ≈ 1.2 cannot resolve the differences being chased, so every
   fitted combiner fits noise. **Better members do not fix a selection problem.**
4. **§D.9/D.10 — more teacher signal is not better teacher signal.** Qwen2.5-VL on the
   *original pixels* carries ~3× the evidence of a BLIP caption (76% vs 25% of aspects)
   with a far better POS:NEG balance (2.16:1 vs 6.8:1) — and every member trained on it is
   ~1.1 worse at **every** threshold tested (floors 0.05/0.20/0.35, non-monotonic). A
   counterfactual Δ is the difference of two noisy estimates; asking a weak teacher
   outright beat differencing it against itself.

## Measured negatives (do not re-run these)
| lever | test | vs 70.37 |
|---|---|---|
| 8B decoder at a dev-tuned mixing weight | 69.63 | −0.74 |
| union pool (BIO ∪ PDQ-MATE, +24 reachable gold spans) | 68.91 | −1.46 |
| option-order TTA on the 8B member | 79.56 (member) | −0.48 |
| selective NEU-boundary gate / per-class weights / NEU-escape | all | invert dev↔test |
| Qwen-VL image teacher → PDS members | 77.24 / 75.60 / 78.21 | −1.1 each |

## Key files
- `possible-patches.md` — **read first.** Chapter D is §D.0–D.10 plus the final standing.
- `experts/pool.py` · `anatomy.py` · `decide.py` — freeze the pool, diagnose it, decide.
- `experts/masc_llm.py` — QLoRA 8B MASC member, restricted-vocab A/B/C scoring in training
  as well as inference. dev 78.88 / test 80.04.
- `experts/masc_qwenvl.py` · `remap_direction.py` — Qwen2.5-VL on the original image, as a
  member (built, **not run** — 4.52 h) and as a counterfactual teacher (run). Caches
  `p_img`/`p_txt` so re-thresholding is free CPU.
- `runs/queue26.sh` … `queue30.sh` — every Chapter D GPU job, in order.

## Running state
- Background processes: **none.** All queues completed; GPU free.
- The 4.52 h Qwen-VL MASC member was deliberately **not** run: §D.8 measured a decorrelated
  member at −0.74, so another one is poor value for that much GPU.

## Verification
- `git status -sb` → clean, `## main...origin/main`.
- Scorer gate: AoM's dumped predictions must re-score to **68.19** (P 67.45 / R 68.95).
  Note this is ~0.4 *stricter* than AoM's published 68.6 — §B.6 resolved the metric
  deliberately in the conservative direction, so all our numbers carry that penalty.
- `python3 experts/decide.py --pool pools/final19 --w-grid 0.0` → dev 70.16 / test 70.43.
- `python3 experts/anatomy.py --pool pools/best15` → reproduces §D.1 and §D.2.
- Run multi-arg member lists under **bash**, not zsh (zsh does not word-split unquoted vars).

## Deferred + open questions
- **The one untested component class: VLP-MABSA pretraining**, reachable via AoM's official
  checkpoint already reproduced in `graft/` at 68.42; needs the legacy py3.8 / torch1.13 /
  transformers3.4 env. **But §D.6/D.8/D.10 say a new *member* will not convert**, so it
  would have to enter as a *backbone* — a different and much larger project.
- Open: **how to report.** 70.43 is defensible and selection-free. Chapter C's 71.37 is not
  dev-selectable and should not be quoted as the system score.
- Open: t2017 was not touched this session. Also untried as a *labelled variant*: t2017
  train as auxiliary MASC supervision (§B4 rejected pooling for MATE on a prior-shift
  argument that does not apply to MASC; the fairness objection does stand).

## Pick up here
The honest position is that the bar needs `a` = 83.1 against our 80.91 and the best
**published** 82.34 — i.e. beating SOTA polarity by ~0.8 *and* converting all of it, on a
T4, without the MABSA pretraining every baseline inherits. §A15 reached the same verdict
from the other direction: the gap is **model-class, not effort or architecture**. Either
commit to VLP-MABSA-as-backbone (large), or write up Chapter D's four findings as the
contribution and report 70.43.
