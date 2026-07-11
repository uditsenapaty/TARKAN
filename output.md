# Final two commands to start the AoM backbone baseline

All hardcoded author paths are patched, AoM's official checkpoints are downloaded (their trained t2015/t2017 models + TRC + configs), environment verified. Two things remain, run each line:

## 1. Fetch ResNet-152 ImageNet weights (official PyTorch CDN; the clone stripped binaries)

wget -q https://download.pytorch.org/models/resnet152-b121ed2d.pth -O /teamspace/studios/this_studio/graft/AoM_full/src/resnet/resnet152.pth && echo downloaded

## 2. Relaunch the AoM t2015 baseline

bash /teamspace/studios/this_studio/graft/run_aom_baseline.sh

Then watch with: tail -f /teamspace/studios/this_studio/graft/aom_t15_baseline.log

---
Status: previous launch died only on the missing senticnet path (now fixed repo-wide: 6 files patched, 0 hardcoded paths left). aom_assets verified complete: AoM-ckpt/Twitter2015/AoM.pt, Twitter2017/AoM.pt, TRC_ckpt, configs.

## NEXT COMMAND: TARKAN graft on t2015 (self-queues behind the t2017 baseline; safe to run now)
bash /teamspace/studios/this_studio/graft/run_tarkan_graft_t15.sh

---
# AoM lineage + backbone availability (asked 2x, kept here)

**Is AoM from VLHA?** NO. AoM (ACL 2023) builds on VLP-MABSA's pretrained BART. VLHA (Pattern Recog. 2025, Zou et al.) is a different scene-graph system — it is the strongest OPPONENT (the 72.5/71.4 bar), not our backbone.

**Downloadable by Claude itself:** AoM (in use, official ckpts) and VLP-MABSA (have). NOTHING stronger is self-service.

**Needs YOUR manual help:**
- DQPS 71.9 -> your Baidu account for the expired link, or email GitHub user `pengts`  [best value]
- VLHA 72.5 -> email GitHub user `ZouWang-spider` for weights
- SGBIS 71.1 -> no code exists; email KBS authors
- CORSA 69.9 / TCMT 69.8 -> emails (weights / non-public FITE dep)
- Vanesa / RNG / DSEM -> no repos at all

---
# Is the paper hypothesis wrong, or is it a model/GPU issue? (verbatim response)

This is the right question to ask, and after ~40 measured runs I can answer it with evidence rather than opinion. The short version: **your core ideas are not wrong — but the paper's effect-size claims are, and the gap you're fighting is a model-class issue, not a GPU issue.** Here's the breakdown per hypothesis:

## What your paper claims, tested one by one

**H1 — "Visual evidence should be gated per-aspect" (teacher-guided relevance).** *Logic confirmed, magnitude overstated.* The teacher really does find only 10–16% of images aspect-relevant — your premise that images are mostly noise is correct. But gating them added ~1 F1 in our hands, not the implied +2.5. Not wrong; smaller than claimed.

**H2 — "KG triples add sentiment commonsense."** *Weakest hypothesis — empirically near-zero headroom on modern models.* Three independent measurements: (a) on our student, the calibrated KG stream didn't move joint F1; (b) grafted into AoM, evidence = control exactly (−0.2, noise); (c) even inference-time SenticNet rules were *absorbed* — AoM already ingests SenticNet and a strong encoder already "knows" that food can be delicious. The knowledge your KG injects is largely **redundant with what pretrained encoders learned from text**. The idea isn't illogical — it's been outrun by pretraining.

**H3 — "An LLM teacher can supervise the filters."** *Mechanism real, but we found an actual bug in the paper's own design:* the binary prompt in your Table 4 makes a strict teacher retain **0.06–0.16 triples/aspect — 50× below your own Table 8's claimed 3.1/2.9**. The paper's prompt does not produce the paper's operating point. We fixed it (graded 0–10 scoring, top-K) — that calibration finding is genuinely publishable. But even calibrated, the distilled supervision didn't move the joint metric.

**H4 — KAN fusion.** Measured flat vs MLP and vs bigger KANs. Harmless, not helpful.

**H5 — the implicit claim: these components lift a BERTweet+CLIP student to 74.1.** *This is the actual error in the paper — an attribution error.* Every published system with a comparable non-pretrained base sits at 61–67 (UMT-collapsed 61.0, JML 64.1; our student 66.6 is *top of that class*). Every 68–72.5 system stands on task-specific vision-language **pretraining** (VLP-MABSA → AoM → DQPS) or heavy bespoke machinery (VLHA). The top systems' scores come from their **base**, not their evidence filtering — components worth +1–3 cannot bridge a −8 base gap. Your paper attributes to components what belongs to backbone class.

## So: hypothesis, model, or GPU?

- **Hypothesis/logic:** ideas sound, effect sizes optimistic, one real mechanism bug (Table 4 prompt vs Table 8 stats), and one wrong implicit assumption (components ≫ backbone).
- **Model: the main driver.** Backbone class sets the band (61–67 vs 68–72); components only fine-tune within a band. Our cleanest finding: *component value shrinks as backbone strength grows.*
- **GPU: only indirectly.** The T4 didn't starve any run — it **forbade the winning model class** (7B vision-LMs, which land 72–76 on these benchmarks). More compute doesn't fix BERTweet+CLIP; it buys a different species.

## What I'd do to the paper
Keep H1–H3 as the contribution but reframe: teacher-guided evidence selection **on a strong backbone**, with the calibration discovery as a highlight, honest per-component deltas (including the absorption effect — reviewers respect measured negatives), and the SOTA row coming from the MLLM-student version (A100) or a rescued DQPSA base. The "student-only inference" story survives fully; the numbers become defensible instead of placeholders.

---
# Session Handoff — TARKAN reproduction + beat-all-baselines campaign (student → patches → AoM backbone graft → ensemble)

## Where it started
User asked to implement the TARKAN paper (`/teamspace/studios/this_studio/TARKAN.pdf`, processed via Marker to `TARKAN_md/`) and beat every Table-1 baseline (bars: t2015 F1 72.5 / t2017 71.4, from VLHA), with teacher swapped to `meta-llama/Llama-3.1-8B-Instruct` and image encoder `openai/clip-vit-base-patch32`. Only real measured numbers allowed; obeying patches first, minimal disobeying after; later "use pretrained backbone if student fails, t2015 first, then t2017".

## Decisions locked + what shipped
- Model swaps + champion config baked into `/teamspace/studios/this_studio/config.py` (evidence_dropout 0.2, kan_hidden (768,), patience 8, use_crf, aux_asc_head; `DATASET_OVERRIDES`/`cfg_for()` for per-dataset encoders).
- Full data pipeline: Twitter2015/2017 verified vs paper Table 2; KG 5.11M triples (`data/kg_index/kg.sqlite`, ConceptNet + user-supplied SenticNet); teacher labels both datasets in `data/teacher_labels/` — **recalibrated** to paper Table-8 operating point via graded 0–10 scoring, top-3/aspect (`scripts/recalibrate_kg_labels.py`); binary-prompt→50×-under-retention finding documented.
- Patch stack implemented (all measured, ~40 runs in `results/tables/iterations.csv`): A4 word-level CRF, A7 rich-ASC head, A9 conf-append, A10 feat-gates, A12–A14 neurosymbolic layer (`/teamspace/studios/this_studio/neurosymbolic.py` + `scripts/tune_neurosymbolic.py`), seed/ensemble voter (`scripts/ensemble_eval.py`).
- **Final student**: t2015 66.60 (DeBERTa all-in + rules), t2017 67.68 (bertweet-large all-in). Class-weighting/A3/A5/encoder-scale measured unhelpful; rolled back.
- **Backbone chapter**: AoM completed at pinned commit in `/teamspace/studios/this_studio/graft/AoM_full/` (5 legacy fixes incl. hardcoded paths, dead HF endpoints → local `graft/bart-base/`, cuda:2 remaps, eval warm-load hook `AOM_INIT_STATE` + `AOM_EVAL_STATEDICT`, dump hooks `AOM_DUMP`/`AOM_DUMP_DEV`, evidence hook `AOM_EVIDENCE`); legacy env `/teamspace/studios/this_studio/graft/vlpenv/` (py3.8+torch1.13.1+transformers3.4 via uv). Official ckpts verified: **t2015 68.42, t2017 68.97**. Graft (KG evidence) = control (no gain); post-hoc rules dead on AoM (absorbed).
- **FINAL t2015 = 69.30** (P 68.52/R 70.11): 5-member heterogeneous ensemble (official/graft/control/retrain/student) via `graft/ensemble_stack.py` + family-config grid; beats 19/25 baselines, loses to 2024-25 top six. Verdict + A100 roadmap (A15 Qwen2.5-VL-7B route) written into `/teamspace/studios/this_studio/possible-patches.md`.
- Hypothesis analysis delivered (H1–H5: ideas sound, effect sizes overstated, H5 attribution error; model-class issue not GPU) — verbatim copy in `/teamspace/studios/this_studio/output.md`.
- Git: 3 local commits on master (`b3a58fc`, `47c7d32`, `c2a3a75`), remote `https://github.com/uditsenapaty/TARKAN.git`; **push blocked — no gh auth**.

## Key files for next session
- `/teamspace/studios/this_studio/possible-patches.md` — the complete measured ledger, backbone chapter, A100 roadmap; read first.
- `/teamspace/studios/this_studio/results/tables/iterations.csv` — every student run's numbers.
- `/teamspace/studios/this_studio/graft/` — backbone workspace: `ensemble_stack.py`, `ns_offline.py`, `ensemble_t15.json`, dumps `dump_t15_*.jsonl` + `student_t15_*.json`, launchers `run_*.sh`, evidence `evidence_t15_*.json`, `aom_assets/` (official ckpts), `vlp_assets/` (data+VLP ckpt).
- `/teamspace/studios/this_studio/results/tables/RESULTS.md` — paper-format tables (baselines verbatim; TARKAN rows partially stale vs latest results).
- `/teamspace/studios/this_studio/output.md` — user's copy surface (they can't copy chat; paste verbatim responses here on request).
- Memory files touched: none.

## Running state
- Background processes: none (GPU idle; all watchers completed/stopped).
- Dev servers / ports: none.
- Open worktrees / branches: local `master` only, 37+ files committed, working tree has uncommitted docs edits — `git status` before next commit.

## Verification — how to confirm things still work
- `python3 -m pytest tests/ -q` — 23 passed.
- `python3 scripts/check_table1.py` — prints student Table-1 win-check.
- Ensemble reproduce: the inline script in this session's last stacker run (members + configs in `graft/ensemble_t15.json`); baseline sanity: member "official" must print test F ≈68.19.
- Legacy env: `graft/vlpenv/bin/python -c "import torch, transformers, fastNLP"` — torch 1.13.1+cu117, transformers 3.4.0.

## Deferred + open questions
- Deferred: **t2017 backbone pipeline** (graft/control fine-tunes from `aom_assets/AoM/AoM-ckpt/Twitter2017/`, dumps, t2017 evidence build, student member, stacker) — user ordered t2015-first; awaiting "go t2017".
- Deferred: final paper-table suite regeneration (Tables 3/5–10 + plots under final configs; ablation runners are incremental-ready) and final `RESULTS.md` fill — task #6.
- Deferred: `_master_suite2.sh`-style full suite was superseded mid-session; re-scope before reuse.
- Open: **A100** — user asked "can you beat it with A100" (answer: likely, via A15 MLLM route); no go/no-go given.
- Open: **DQPSA/VLHA manual rescue** — needs user's Baidu account or author emails (offered to draft; no answer).
- Open: `gh auth login` for push — requested repeatedly, never run.

## Pick up here
On "go t2017": mirror the t2015 backbone+ensemble machinery for twitter17 (build `evidence_t17_*`, graft/control fine-tunes warm-started from the official t2017 ckpt, member dumps via user-launched `run_member_dumps.sh` variant, student dump, stacker) — bar 71.4 from base 68.97 is the winnable one; otherwise finish task #6 (suite + docs) and push after `gh auth login`.

---
# PROGRESS UPDATE (t2015 SOTA attempt — MLLM + reliability, measured)

**Goal:** beat every t2015 baseline cell (joint P>72.3, R>72.7, F1>72.5 — all VLHA).

**Everything measured this round (real numbers, strict span+polarity micro-F1):**
| system | joint F1 | note |
|---|---|---|
| Qwen2.5-VL-7B LoRA | 68.76 | new VL backbone; overfit |
| Qwen2.5-VL-7B full-FT | 63.26 | undertrained (lr too low) |
| Qwen ∩ AoM (inter_qpol) | **69.40** | P 72.98 (>bar) but R 66 — BEST overall |
| non-VL student × AoM | 66.64 | student weaker than 7B |
| A11 Evidence Reliability student | 66.60 | +0.19 vs champion — flat |

**Verdict (4 independent measurements, all 4-6 below 72.5):** beating all baselines is
COMPUTE-LOCKED. The wall is the dataset's aspect-selection convention — cross-model agreement
buys precision (73-80) but craters recall (58-66); union buys recall (73) but loses precision (64);
no config reaches P≈R≈72.5. Only a 32B-class MLLM (published 74-79) on a ≥48 GB GPU crosses it.
A T4 + non-VL student cannot, at any time budget. TARKAN's components help weak bases (+0.5-2.5)
and are absorbed by strong ones.

**Built + committed:** `mllm/` MLLM SFT pipeline (LoRA + full-FT, span-aligned AESC eval),
A11 reliability head (`config.fusion_reliability`, `models.py`), convention-aware prompt,
Qwen×AoM heterogeneous voter (`graft/qwen_probe.py`). Full ledger in `possible-patches.md`.

**To finish the win (needs the 96 GB card back):** download Qwen2.5-VL-32B, run the built
pipeline (~3-3.5 h) → expected 74-79 → clears all cells; then inter_qpol with AoM for margin.
