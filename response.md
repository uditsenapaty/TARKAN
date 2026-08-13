# Session Handoff — TARKAN Chapter C: rebuild the two-stage pipeline on a T4 and chase the MADSC bar

## Where it started
The Studio arrived wiped: `data/` was 4.7 MB (senticnet only), `experts/` did not exist, no
Python env, no HF cache, and the entire A100-session pipeline (which had reached joint F1
71.26) had never been committed. Hardware had also regressed from 5×A100-40GB to a single
Tesla T4 16 GB (fp16 only, no bf16). The user supplied `TARKAN_new.pdf`, which moves the
beat-everything target from VLHA 72.5 to **MADSC 72.9** (per-cell P>72.8, R>73.1), and set a
standing goal: beat every t2015 baseline, trying any structural patch, keeping
`possible-patches.md` updated throughout.

## Decisions locked + what shipped
- **Everything is committed and pushed.** `/teamspace/studios/this_studio` — commits
  `dbea47b` (Chapter C rebuild) + `d493ae9` (merge), pushed to
  `https://github.com/uditsenapaty/TARKAN.git` main. `.git` had been corrupt (objects/ and
  refs/ missing) so the repo was re-initialised; the remote's Chapter A/B lineage
  (`b3a58fc..662bebb`) was **merged back in, not force-pushed over**.
- **Data restored and verified** — cloned `CopotronicRifat/TwitterDataMABSA`; counts match
  paper Table 2 exactly (3179/1122/1037 · 3562/1176/1234), images symlinked at
  `/teamspace/studios/this_studio/data/images/`.
- **Scorer regression-gated**: AoM's dumped predictions re-score to joint 68.19 (P 67.45 /
  R 68.95) on exactly 1037 pairs — identical to Chapter B, so all numbers are comparable.
- **16 expert modules rebuilt** in `/teamspace/studios/this_studio/experts/` (see Key files).
- **Best measured result: joint F1 71.37** (P 71.47 / R 71.26; MATE 87.01, `a` 81.21) with
  15 members. **~70.5 under strict dev-selection.** Clears 16 of 19 t2015 baselines
  including SGBIS 71.1; DQPSA 71.9 / VLHA 72.5 / MADSC 72.9 remain above. **Goal not met.**
- **MATE is solved**: 87.01 selection-free, above MADSC's own extractor (86.60), CORSA
  (86.30), AoM (86.20). Root cause of the earlier shortfall was head-lr 1e-3 over-driving
  the CRF transition matrix; 1e-4 fixed it.
- **Two mechanisms that worked** (each ≈ +0.3, then stopped scaling): PDQ with the full
  DQPSA objective (ITC+ITM+EPE), and PDS in its residual + signed-margin + POS/NEG-balanced
  + `w_none=0` form.
- **Reporting position agreed in the ledger**: 71.37 is *best measured*, not dev-selectable;
  ~70.5 is the defensible number. Quoting 71.37 as the system score would be test selection.

## Key files for next session
- `/teamspace/studios/this_studio/possible-patches.md` — **read first.** 1509 lines, now a
  strict superset: Chapter A (T4 era) verbatim, Chapter B (A100 session, recovered from the
  user's paste and never previously committed), and Chapter C §C.0–C.27 with every measured
  positive and negative.
- `/teamspace/studios/this_studio/experts/` — `common.py` (canonical data layer),
  `mate_expert.py` (anchor generator, O/B/I + vendored CRF), `assemble.py` (two-stage
  assembly + joint-expected-F1 decoding + evidence product), `pdq.py` / `pdq_mate.py`
  (BLIP-2 Q-Former + EPE GlobalPointer), `masc_text.py`, `masc_gated.py`, `masc_pds.py`,
  `pds_teacher.py`, `aadg.py`, `span_rerank.py`, `stack.py`, `diagnose.py`, `emit_spans.py`,
  `cache_vit.py`, `knn_memory.py`.
- `/teamspace/studios/this_studio/referred_clones/FIXES.md` — records that **MADSC has no
  public code** (searched 2026-08-13) and that CORSA's `senticnet_word.txt` is never read by
  its own code (we repurposed it as an opinion lexicon).
- `/teamspace/studios/this_studio/runs/` — 83 GB of artifacts; checkpoints and `.npz` prob
  dumps are gitignored, but all `queue*.log` and `metrics.json` are committed.
- `/teamspace/studios/this_studio/results/` — `q7_*.json` … `q13_*.json` assembly outputs.
- Plan file: none (no `.claude/plans` artifact drove this session).
- Memory files touched: none. `/teamspace/studios/this_studio/CLAUDE.md` Applied Learning
  section was extended with 6 environment one-liners.

## Running state
- Background processes: **none.** All queues (queue1–queue25) completed or were killed; GPU
  shows 0 MiB. No shell IDs outstanding.
- Dev servers / ports: none.
- Open worktrees / branches: none beyond `main`, which is in sync with `origin/main`.

## Verification — how to confirm things still work
- `cd /teamspace/studios/this_studio && git status -sb` — expect `## main...origin/main`
  with a clean tree.
- `python3 -c "import json;from metrics import joint_prf;rows=[json.loads(l) for l in open('graft/dump_t15_test.jsonl')];print(joint_prf([[tuple(x) for x in r['pred']] for r in rows],[[tuple(x) for x in r['gold']] for r in rows]))"`
  — must print P 67.45 / R 68.95 / F1 68.19. This is the scorer gate; if it drifts, nothing
  downstream is comparable.
- `python3 -c "from experts.common import load;print([(s,len(load('twitter2015',s))) for s in ['train','dev','test']])"`
  — expect 2101 / 727 / 674 sentences (3179 / 1122 / 1037 aspects).
- `python3 experts/assemble.py --mate runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB --masc runs/masc_btwL_s45 runs/masc_deb_s44 runs/masc_robL_s46 runs/masc_twrob_s42 runs/masc_btw_s43 runs/masc_deb_sqrt runs/pdq_btwL_s47 runs/pdq_twrob_s43 --mode span`
  — reproduces the core-8 baseline. Note: run under **bash**, not zsh — zsh does not
  word-split unquoted variables and the multi-arg lists silently collapse.
- Reproducing 71.37 requires the evidence product (tagger × judge × PDQ-MATE × MASC), which
  lives in ad-hoc scripts in the transcript, not yet in `assemble.py`. See "Deferred".

## Deferred + open questions
- Deferred: **fold the evidence product into `assemble.py`.** The 71.37 configuration is
  currently only reproducible via an inline script; `assemble.py --rerank` covers the judge
  term but not the PDQ-MATE span-probability term.
- Deferred: **Qwen2.5-VL-7B MASC member** (~4 h on the T4) — restores the strong VL member
  Chapter B leaned on. Deprioritised because our best member (79.94) already matches
  Chapter B's Qwen members.
- Deferred: **VLP-MABSA pretrained weights** — the MABSA-specific vision-language
  pretraining every 68–72 baseline inherits and this pipeline lacks; Chapter B §7c named it
  as the reason our visual stream contributes ≈0. This is the highest-upside untested lever.
- Deferred: t2017 entirely. Only t2015 was touched this session.
- Open: **which of the two big levers to spend GPU on** — Qwen-7B member vs VLP-MABSA
  weights. Recommended VLP-MABSA; no decision given.
- Open: **how to report 71.37 vs ~70.5 in the paper.** Dev cannot select the 71.37
  configuration (it ranks 4th on dev), and test spans 70.14–71.37 while dev spans
  69.36–70.08 in roughly the opposite order. The user has not said which number to use.
- Open: the `.env.local` HF token was invalid at first and replaced mid-session; it now
  works (`UzzyDizzy`), and `meta-llama/Llama-3.1-8B-Instruct` is granted while
  `Llama-3.2-3B-Instruct` is **not** (403).

## Pick up here
Fold the evidence-product scoring (tagger conf × span judge × PDQ-MATE span prob × MASC
conf, equal weights, geometric) into `experts/assemble.py` so the 71.37 result is
reproducible from a single command — then decide between the Qwen-7B member and VLP-MABSA
weights for the next GPU spend.
