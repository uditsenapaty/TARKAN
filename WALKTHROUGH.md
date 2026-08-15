# TARKAN — Reproduction Walkthrough

Everything needed to rebuild, evaluate and ablate the three best **paper-faithful**
configurations on **Twitter-2015** and **Twitter-2017**, from a clean checkout.

**Faithful** here means: no external aspect-sentiment corpora, no t2017 data used for a
t2015 number, no test labels touched, no architectural departure from the paper. The
external-supervision line of work (§D.33, `experts/absa_extra.py`) is **excluded from every
configuration below** — it is recorded in `possible-patches.md` as a finding, not shipped.

Single Tesla T4 16 GB throughout. All timings are measured wall clock on that card.

---

## 0. The three configurations

| rank | config | members | **t2015 F1** | status |
|---|---|---|---|---|
| **#1** | `F19+KG+KAN` | 5 MATE + 21 MASC | **70.62** | base measured · KG/KAN to build |
| **#2** | `F18+KG+KAN` | 5 MATE + 20 MASC | **70.41** | base measured · KG/KAN to build |
| **#3** | `F7+KG+KAN` | 5 MATE + 9 MASC | **70.27** | base measured · KG/KAN to build |

Measured detail on the bases (t2015 test, dev-selected τ, `joint = MATE@τ × a`):

| base | dev | test | P | R | MATE@τ | a |
|---|---|---|---|---|---|---|
| F19 | 70.43 | **70.62** | 70.66 | 70.59 | 87.80 | 80.44 |
| F18 (F19 − QLoRA-8B) | 70.36 | **70.41** | 70.61 | 70.20 | 87.62 | 80.35 |
| F7 | 70.10 | **70.27** | 70.34 | 70.20 | 87.74 | 80.09 |

> ### ⚠ Read before quoting any number
> * **The bases are measured. The `+KG+KAN` additions are NOT.** Neither component exists in
>   the current pipeline (see §2). §B.7 measured raw KG triples as noise, §C.7 measured the
>   relevance gate at **+0.10**, and §D.26/§D.27 measured the whole evidence family at
>   −0.20/−0.23. **Expect these numbers to hold, not rise.**
> * **The single-run detection floor is ±1.31 F1** (§D.20, paired seeds, n_dev = 1122).
>   The 0.35 spread across #1–#3 is *inside* it. Ranking them by test F1 is convention, not
>   evidence.
> * **Member-set choice is a lottery** (§D.14): test spans 69.47–70.62 across sets that dev
>   cannot separate. Report the choice-free rule *and* the fixed structure, as below.
> * KAN as an added *member* is not KAN in its designed position (the fusion head). The only
>   architecture where it is native is Chapter A's single trunk, measured **65.87–69.30**.

**Targets** (t2015): MADSC **P 72.8 / R 73.1 / F1 72.9** · VLHA 72.5 · DQPSA 71.9 ·
SGBIS 71.1 · CORSA 69.9 · AoM 68.6. The configurations above clear **15 of 19**.

---

## 1. Prerequisites

```bash
cd /teamspace/studios/this_studio
conda activate cloudspace          # Lightning Studio blocks venv; use the default env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`data/` is wiped between sessions. Restore it first:

```bash
git clone https://github.com/CopotronicRifat/TwitterDataMABSA data/_raw
ln -s $PWD/data/_raw/images data/images        # per-dataset image dirs underneath
python3 -c "from experts.common import load; \
            print({s: len(load('twitter2015', s)) for s in ('train','dev','test')})"
# expect {'train': 2101, 'dev': 727, 'test': 674}
```

Sanity gate — canonical split must be bijective with the AoM dumps:

```bash
python3 -c "
from experts.common import load, gold_pairs
for ds, n in (('twitter2015',1037), ('twitter2017',1234)):
    g = sum(len(x) for x in gold_pairs(load(ds,'test')))
    print(ds, 'test aspects', g, 'OK' if g==n else 'MISMATCH')"
```

---

## 2. The two components that must be built first

**Both are absent from the current pipeline.** This is the honest state and the reason this
section exists; see the component audit in `possible-patches.md` §D.34.

### 2a. KG filtering

`data/kg_index/` and `data/kg_evidence/` do not exist — they were built in Chapter B and
lost when `data/` was wiped. No member currently reads KG. The builder survives.

```bash
# one-time, dataset-independent  (~45 min: ConceptNet fetch + streaming sqlite build)
mkdir -p data/conceptnet
#   place conceptnet_en.parquet in data/conceptnet/  (head, relation, tail, weight)
#   data/senticnet/senticnet_en.parquet is already force-added to the repo
python3 scripts/build_kg.py                    # -> data/kg_index/kg.sqlite

# per dataset (~20 min): top-M=10 triples per aspect, teacher-ranked
python3 scripts/build_kg_evidence.py --dataset twitter2015   # -> data/kg_evidence/
```

Expected build stats (Chapter B, t2015): **83.8 % aspect hit rate, 8.0 triples/aspect**
against the paper's 68.4 % / 8.7 — the right regime.

**Then wire one member to consume it.** The KG member is a text tower whose second segment
carries the filtered triples, exactly as `--desc` carries the AADG description:

```bash
# TO IMPLEMENT: experts/masc_text.py --kg data/kg_evidence/twitter2015
#   appends the top-M teacher-ranked triples after the aspect-marked tweet.
#   Unfiltered triples are noise (§B.7) — the teacher filter is the load-bearing part.
```

### 2b. KAN

**Never implemented as a layer in this repo.** The only `KAN` strings are two docstring
lines in `experts/masc_gated.py`. It *was* implemented and measured in Chapter A, where the
fusion family KAN-vs-MLP-vs-gated came out **flat** and "KAN capacity" **flat (±0.2)**.

```bash
# TO IMPLEMENT: experts/kan.py — KANLayer(in, out, grid=5, spline_order=3)
#   paper spec: 2 layers x width 256, grid 5, spline order 3
# TO IMPLEMENT: experts/masc_gated.py --head kan
#   replaces the final Linear over the fused z_a with the KAN stack.
#   masc_gated already implements MADSC Eqs. 9/13/14/16/19 (calibrator, gate,
#   convex fusion) and is currently in NO pool — adding it is what makes the
#   KAN + modality-gate path real.
```

---

## 3. Shared artifacts (per dataset)

Run once per dataset before any member. Times are for t2015; t2017 is ~15 % larger.

| # | artifact | command | time |
|---|---|---|---|
| 1 | BLIP captions | `python3 experts/aadg.py --stage captions --dataset twitter2015` | 15 min |
| 2 | CLIP region grid | `python3 experts/aadg.py --stage regions --dataset twitter2015` | 20 min |
| 3 | AADG descriptions | `python3 experts/aadg.py --stage describe --dataset twitter2015` | 90 min |
| 4 | PDS teacher labels | `python3 experts/pds_teacher.py --split train --batch 4` | 60 min |
| 5 | BLIP-2 ViT patch cache † | `python3 experts/cache_vit.py --dataset twitter2015` | 20 min |
| 6 | KG evidence (§2a) | `python3 scripts/build_kg_evidence.py --dataset twitter2015` | 20 min |

† **#5 is only needed for configs #1 and #2** — config #3 has no PDQ members.
Config #1/#2 artifacts = **3.4 h**; config #3 artifacts = **3.1 h**.

---

## 4. Stage 1 — extraction (all three configs share this)

Five MATE members: three deberta seeds plus two recipe probes. §C.6's fixed recipe
(`--head-lr 1e-4`; the original 1e-3 cost ~1.2 MATE F1).

```bash
for s in 42 43 44; do
  python3 experts/mate_expert.py --seed $s --lr 1e-5 --head-lr 1e-4 \
    --epochs 12 --patience 4 --out runs/mate_deb_s$s            # 16 min each
done
python3 experts/mate_expert.py --seed 42 --lr 1e-5 --head-lr 1e-4 --epochs 16 \
  --patience 5 --dropout 0.3 --out runs/mate_probeA             # 21 min
python3 experts/mate_expert.py --seed 42 --lr 2e-5 --head-lr 1e-4 --epochs 16 \
  --patience 5 --out runs/mate_probeB                           # 21 min
```

Freeze the candidate anchor set. **`cand_thr 0.12` is the measured optimum** — lower merges
adjacent spans and recall *falls* (§D.3):

```bash
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
python3 experts/emit_spans.py --mate $MATE --cand-thr 0.12 --out runs/mate_ens5_hr
# t2015: dev 1234 / test 1178 candidates, pool recall 89.75% / 91.13%
```

Two auxiliary scorers used by the decision rule:

```bash
for f in 0 1; do
  python3 experts/mate_expert.py --seed 42 --lr 1e-5 --head-lr 1e-4 --epochs 12 \
    --patience 4 --dropout 0.3 --fold $f --nfolds 2 --out runs/mate_oof_f$f
done
python3 experts/span_rerank.py --oof runs/mate_oof_f0 runs/mate_oof_f1 \
  --cand runs/mate_ens5_hr --mate $MATE --cand-thr 0.12 \
  --model microsoft/deberta-v3-large --seed 101 --epochs 5 --out runs/rerank
python3 experts/pdq_mate.py --text-model vinai/bertweet-large --seed 42 \
  --lr 1e-5 --epochs 10 --patience 3 --out runs/pdqmate_btwL
```

**Stage 1 total: 90 min members + ~70 min auxiliaries.**

---

## 5. Stage 2 — polarity members

`S=runs/mate_ens5_hr` throughout, so every member scores the **same** frozen candidate set.
Omitting `--spans` silently falls back to gold-span probabilities and produces a member that
looks weak rather than broken — always pass it.

```bash
S=runs/mate_ens5_hr
```

### Config #3 — `F7+KG+KAN` (9 MASC members, cheapest)

```bash
# 4 text towers — architecture diversity is what they contribute, not strength
python3 experts/masc_text.py --model vinai/bertweet-large  --seed 45 --batch 8 --lr 1e-5 \
  --epochs 6 --spans $S --out runs/masc_btwL_s45                          # 11 min
python3 experts/masc_text.py --model microsoft/deberta-v3-large --seed 44 --batch 8 \
  --lr 1e-5 --epochs 6 --spans $S --out runs/masc_deb_s44                 # 11 min
python3 experts/masc_text.py --model roberta-large --seed 46 --batch 8 --lr 1e-5 \
  --epochs 6 --spans $S --out runs/masc_robL_s46                          # 11 min
python3 experts/masc_text.py --model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --seed 42 --batch 16 --lr 2e-5 --epochs 6 --spans $S --out runs/masc_twrob_s42   # 11 min

# 3 PDS members — teacher-guided evidence, direction supervision (§C.25)
for m in "vinai/bertweet-large btwL 82" "microsoft/deberta-v3-large deb 83" \
         "roberta-large robL 85"; do
  set -- $m
  python3 experts/masc_pds.py --residual --model "$1" --seed "$3" \
    --lambda-pds 0.5 --w-none 0.0 --spans $S --out runs/pds_res_${2}_wn0   # 13 min each
done

# KG member (§2a) and KAN member (§2b)
python3 experts/masc_text.py --model vinai/bertweet-large --kg data/kg_evidence/twitter2015 \
  --seed 47 --batch 8 --lr 1e-5 --epochs 6 --spans $S --out runs/masc_kg_btwL   # 11 min
python3 experts/masc_gated.py --model vinai/bertweet-large --head kan --fuse convex \
  --desc data/aadg/twitter2015 --seed 50 --spans $S --out runs/masc_kan_btwL    # 11 min
```

**Config #3 stage 2: 105 min. Full build (with stage 1): 3.3 h/seed.**

### Config #2 — `F18+KG+KAN` (20 MASC members)

Everything in #3, plus:

```bash
# 2 more text towers
python3 experts/masc_text.py --model vinai/bertweet-base --seed 43 --spans $S \
  --out runs/masc_btw_s43                                                  # 2.5 min
python3 experts/masc_text.py --model microsoft/deberta-v3-large --class-weight sqrt \
  --seed 71 --batch 8 --lr 1e-5 --epochs 8 --spans $S --out runs/masc_deb_sqrt   # 11 min

# 3 counterfactual-teacher PDS members
for m in "vinai/bertweet-large btwL" "microsoft/deberta-v3-large deb" "roberta-large robL"; do
  set -- $m
  python3 experts/masc_pds.py --residual --w-none 0 --pds data/pds_qwen/twitter2015 \
    --model "$1" --spans $S --out runs/qpds_${2}_bal                       # 13 min each
done

# 6 PDQ members — BLIP-2 Q-Former, the mechanism-decorrelated block (§C.2)
python3 experts/pdq.py --text-model vinai/bertweet-large --seed 47 --batch 8 --lr 1e-5 \
  --epochs 8 --spans $S --out runs/pdq_btwL_s47                            # 14 min
python3 experts/pdq.py --text-model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --seed 43 --spans $S --out runs/pdq_twrob_s43
for m in "vinai/bertweet-large btwL 60 1e-5" \
         "cardiffnlp/twitter-roberta-base-sentiment-latest twrob 61 2e-5" \
         "microsoft/deberta-v3-large deb 62 1e-5" "roberta-large robL 64 1e-5"; do
  set -- $m
  python3 experts/pdq.py --text-model "$1" --itc 1.0 --itm 1.0 --seed "$3" \
    --lr "$4" --epochs 8 --spans $S --out runs/pdq_${2}_itcitm
done
```

**Config #2 full build: 5.4 h/seed.**

### Config #1 — `F19+KG+KAN` (21 MASC members)

Everything in #2, plus the QLoRA Llama-3.1-8B decoder member:

```bash
python3 experts/masc_llm.py --seed 42 --spans $S --out runs/masc_llm_s42   # 60 min
```

**Config #1 full build: 6.4 h/seed.**
This member costs 60 min/seed/dataset and buys **+0.21 F1**; §D.6/§D.8 measured it as
converting to nothing in the joint metric. Drop it (→ config #2) if time is tight.

---

## 6. Stage 3 — assemble and evaluate

```bash
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
F7="runs/masc_btwL_s45 runs/masc_deb_s44 runs/masc_robL_s46 runs/masc_twrob_s42 \
runs/pds_res_btwL_wn0 runs/pds_res_deb_wn0 runs/pds_res_robL_wn0 \
runs/masc_kg_btwL runs/masc_kan_btwL"
F18="$F7 runs/masc_btw_s43 runs/masc_deb_sqrt \
runs/qpds_btwL_bal runs/qpds_deb_bal runs/qpds_robL_bal \
runs/pdq_btwL_s47 runs/pdq_twrob_s43 runs/pdq_btwL_itcitm runs/pdq_twrob_itcitm \
runs/pdq_deb_itcitm runs/pdq_robL_itcitm"
F19="$F18 runs/masc_llm_s42"

for name in F7 F18 F19; do
  eval M=\$$name
  python3 experts/pool.py --mate $MATE --masc $M --rerank runs/rerank \
    --pdqmate runs/pdqmate_btwL --cand-thr 0.12 --out pools/$name
  printf "%-6s " $name
  python3 experts/decide.py --pool pools/$name --w-grid 0.0
done
```

`decide.py` fits **two scalars on dev only** (`w`, `τ`) and prints
`P R F1 MATE@τ a` on test. CPU, seconds.

**Report both rules** — the project convention since §D.14:
* **choice-free**: every available correct member, equal weight, τ the one fitted scalar
* **fixed structure**: the named member set above, τ fitted on dev

Two τ alternatives were tested and are **both dead** (§D.33): count-matching
(τ s.t. |pred| = 1.513 × n_sent) loses 0.18–0.50, and the isotonic-calibrated §D.22 cut
(accept q > F1/2) reproduces dev-argmax to within 0.23. Do not re-try them.

---

## 7. Ablations

Single seed each, t2015. Every row rebuilds only the members it touches.

| row | what changes | command delta | rebuild | time |
|---|---|---|---|---|
| **−KG** | drop the KG member | remove `runs/masc_kg_btwL` from the pool | none | **0** |
| **−KAN** | KAN → plain Linear head | `masc_gated ... --head mlp` | 1 member | 11 min |
| **−modality gate** | convex gate → concat | `masc_gated ... --fuse concat` | 1 member | 11 min |
| **−teacher evidence** | drop PDS direction supervision | `masc_pds --lambda-pds 0` | 3–6 members | 39–78 min |
| **−AADG description** | evidence text removed | drop `--desc` | 3–6 members | 39–78 min |
| **−aspect anchors** | argmax decode, no candidate set | `emit_spans --cand-thr 0.0` | 0 (re-pool) | 1 min |
| **−CRF** | token softmax instead of linear-chain | `mate_expert --no-crf` | 5 MATE | 90 min |
| **−visual entirely** | text towers only | pool = 4 text members | none | **0** |
| **sub-task rows** | MATE-only / MASC-only | already in `metrics.json` | none | **0** |

**Config #1/#2 ablations ≈ 5.5 h · config #3 ≈ 3.5 h.**

Statistical note: with a **±1.31** single-run floor, expect most of these rows to be
directional only. Deltas worth defending need paired seeds — the protocol is in §D.20.

---

## 8. Twitter-2017 (separate, clean)

t2017 is a **separate experiment**. Never mix it into a t2015 number and never select on it.
§B.4 measured pooling the two train sets at **MATE −3.2** (t2017 has 2.04 aspects/sentence
vs t2015's 1.51 → prior shift), and it is unfair besides, since the baselines are
single-dataset.

Repeat §3–§6 with `--dataset twitter2017` and `--split` paths changed. Known quirk to
document rather than "fix": t2017 ships **1 exact train/test duplicate tweet plus 3
near-duplicates in the original release** — every Table-1 baseline carries it. t2017 train
is 3560 unique aspects, not 3562 (`train.tsv` duplicates one tweet).

---

## 9. Time budget

Measured unit costs on one T4: MATE tagger 16 min · MATE probe 21 min · text tower
11 min (large) / 2.5 min (base) · PDQ 14 min · PDS 13 min · QLoRA-8B 60 min ·
KAN/KG member 11 min · `pool`+`decide` < 1 min (CPU).

| | **#1 F19+KG+KAN** | **#2 F18+KG+KAN** | **#3 F7+KG+KAN** |
|---|---|---|---|
| t2015 F1 (measured base) | **70.62** | **70.41** | **70.27** |
| build / seed | 6.4 h | 5.4 h | 3.3 h |
| artifacts / dataset | 3.4 h | 3.4 h | 3.1 h |
| **1 seed, t15 + t17** | **20.4 h** | **18.4 h** | **13.6 h** |
| **+ ablations (t15)** | **25.9 h** | **23.4 h** | **17.1 h** |
| **3 seeds, t15 + t17 + ablations** | **51.5 h** | **45 h** | **30.3 h** |

Plus **45 min one-time** for the ConceptNet fetch + KG sqlite build (dataset-independent).

**Under 24 h with everything included: #2 (23.4 h, no margin) and #3 (17.1 h, comfortable).**
#3 costs 0.35 F1 against #1 — inside the noise floor — while using 12 fewer members.

---

## 10. Operational notes

* **Run the deterministic battery before any GPU spend.** A multi-hour failure becomes
  seconds: `.claude/skills/custom-skills/ml-deterministic-checks`.
* `transformers ≥ 4.56` loads checkpoint dtype, so **deberta-v3-large arrives fp16** and
  breaks AMP. Pass `dtype=torch.float32` and let autocast handle the forward.
* Chain queue scripts serially (`bash a.sh; bash b.sh`). A waiting shell running
  `pgrep -f "<script>"` **matches itself** and deadlocks with the GPU idle.
* `pool.py` reads only `probs_span_*.npz`, `marginals_*.npz`, `spanscore_*.npz` — never
  `best.pt`. Checkpoints from measured runs are safe to delete when disk is tight.
* Commit `experts/` early. Two sessions of pipeline code have already been lost.

---

## 11. What is deliberately not here

* **§D.33 external aspect-sentiment supervision** (Dong-2014 + SemEval-14 + MAMS, 23,370
  aspects). Measured at **+1.41 tower-level polarity** (t = 3.90) and **+0.64 extraction**
  (t = 7.60), both 3/3 seeds, leak gate clean at max Jaccard 0.000. Ensemble conversion was
  **+0.01 under the strictest rule / +0.70 under the 19-member convention.** Excluded here
  because it adds data the baselines do not use. Full record: `possible-patches.md` §D.33.
* Everything in the **DID NOT HELP** tables of `possible-patches.md` — advanced combiners,
  rationale distillation, cross-family VL members, per-class log-bias, union pools, PACS,
  ASOE, CET, TORF, TBRF. All measured, all ≤ 0. Do not re-try blindly.
