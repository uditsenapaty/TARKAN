# PACS — Polarity-Aware Candidate State: full observed results and findings

**Verdict: negative. The coupling route is closed, and the reason generalises.**
Everything below is measured on t2015 only. No t2017 was used anywhere in this work.

---

## 1. The hypothesis and why it was worth testing

§C.18 is the sharpest structural finding in the project:

| MASC accuracy measured on | value |
|---|---|
| all gold spans (`a_gold`) | **81.20** |
| spans MATE actually extracts (`a_selected`) | **80.52** |
| spans MATE **misses** | **86.73** |

The extractor systematically selects the spans its own classifier is **worst** at. The
anti-correlation is `a_selected − a_gold = −0.68`. MADSC's published numbers imply the
opposite sign (`a` 84.18 against 82.34 gold-span, i.e. **+1.84**) — a ~2.5-point swing in
`a`, almost exactly our joint gap.

The cause is structural and self-inflicted: Chapters C/D deliberately built MATE and MASC as
*decorrelated* model families, which is ideal for ensembling and exactly wrong for joint
MABSA.

**Why this reframed the target.** The bar had been read as "get `a` to 83.1 on all gold
spans", which is above every published number. But spans at **86.73** accuracy already exist
in this system — they are simply not selected. Holding MATE@τ at 87.8, moving `a_selected`
from 80.4 to ~83.5 clears 72.9. That is a selection problem inside a population we already
have, not a demand for SOTA polarity.

---

## 2. The arithmetic — including a threshold error worth recording

Micro-F1 is `2·TP/(|pred| + |gold|)`. Dropping a prediction whose probability of being
correct is `q` therefore improves F1 exactly when

```
q < F1/2          (derivation: 2(TP−q)/(K−1+G) > 2TP/(K+G)  ⇔  q < TP/(K+G) = F1/2)
```

which is **0.353** at our operating point (TP 728, kept 1032, gold 1037, F1 70.6).

**Both I and the ChatGPT proposal initially used `q > 0.5`.** That rule minimises *expected
error count* (`2(1−q)` if extracted vs `1` if skipped) — a different objective — and it
over-drops. With **oracle** knowledge of `q`:

| cut | predictions dropped | TP lost | projected F1 |
|---|---|---|---|
| **q < 0.353** (correct for F1) | 187 | 25.2 | **74.68** |
| q < 0.40 | 205 | 32.1 | 74.66 |
| q < 0.50 (the intuitive rule) | 253 | 53.6 | 74.28 |

**74.68 clears the 72.9 bar.** Everything that follows is about whether `q` is obtainable
without the answers.

---

## 3. The determinability signal `q`, and its validation

`q` = probability that a candidate's polarity will be predicted correctly, estimated from
four architecturally distinct out-of-fold towers (`oof_btwL`, `oof_btw`, `oof_deb`,
`oof_twrob`).

**Estimator choice, measured not assumed.** Vote-counting (`k/4`) throws away most of the
signal — within the k=2 bucket the true `q` spans 0.29–0.71:

| k of 4 towers correct | n | `q` range | mean `q` |
|---|---|---|---|
| 0 | 356 | 0.00–0.35 | 0.055 |
| 1 | 278 | 0.13–0.50 | 0.297 |
| 2 | 303 | 0.29–0.71 | 0.508 |
| 3 | 462 | 0.43–0.86 | 0.717 |
| 4 | 1780 | 0.68–1.00 | 0.961 |

**Cross-family transfer test.** Estimate `q` from 2 towers, then predict whether the *other*
2 architectures get the same aspect right:

| q from | predicts | AUC (vote count) | AUC (mean softmax) |
|---|---|---|---|
| btwL+btw | deb+twrob | 0.8744 | **0.9217** |
| btwL+deb | btw+twrob | 0.8439 | 0.9022 |
| btwL+twrob | btw+deb | 0.8533 | 0.9028 |
| btw+deb | btwL+twrob | 0.8554 | 0.9049 |
| btw+twrob | btwL+deb | 0.8454 | 0.9023 |
| deb+twrob | btwL+btw | 0.8547 | 0.9046 |
| **mean** | | **0.8545** | **0.9064** |

So per-example polarity difficulty is a largely **model-independent** property of examples —
**not** merely the shared blind spot §D.11 warned about — and the softmax estimator is worth
**+0.05 AUC** over vote-counting. Reference: §D.2 measured MASC max-probability at AUC
**0.659** for the same target.

`q` is saved in `data/qdet_train.npz` (3179 aspects). 779 (24.5%) fall below 0.5; 576
(18.1%) below the correct 0.353 cut.

---

## 4. PACS itself — architecture and results

`experts/pacs.py`. One encoder, two heads:

```
deberta-v3-large ─► word representations hw
                      ├── proj → BIO emissions → linear-chain CRF   (anchor / MATE)
                      └── span pooling [mean, first, last] → MLP    (polarity / MASC)

  U(s) = A(s) · P(ŷ|s)^λ          A(s) = mean(1 − P(O)) over the span
```

Both heads read the **same** `hw`, so a joint loss backprops through both at once — which is
what a post-hoc product over frozen scores structurally cannot do. TARKAN's identity is
preserved: BIO anchor generation with a CRF, aspect-conditioned polarity, student-only
inference.

### Results (seed 42, 16 epochs, identical recipe throughout)

| arm | MATE@τ dev | MATE@τ test | `a_selected` dev | **`a_selected` test** | joint dev | **joint test** |
|---|---|---|---|---|---|---|
| **control** (`lam_joint` 0, `lam_det` 0) | 84.70 | 84.83 | 78.88 | **76.91** | 66.82 | **65.24** |
| **determinability-ranked** (`lam_det` 1.0) | 84.96 | 85.17 | 77.60 | **74.00** | 65.93 | **63.02** |
| Δ | +0.26 | +0.34 | −1.28 | **−2.91** | −0.89 | **−2.22** |

**The result is negative and of the opposite sign to the hypothesis.** Training the extractor
to prefer polarity-determinable spans made its selected spans **harder** for its own
classifier (`a_selected` −2.91) and cost 2.22 joint F1. At −2.22 this is well clear of the
§D.20 ±1.31 single-pair detection floor, so it is a real degradation rather than noise.

Reading: forcing the anchor score to encode difficulty **fights the tagging objective** and
damages both. MATE@τ rose slightly (+0.34) while the polarity of what it selected collapsed.

### A structural limit found while designing the control
Hard parameter sharing costs extraction before PACS even starts: the shared-encoder control
gets MATE@τ **84.83** against the dedicated tagger's **85.77** and the 5-member ensemble's
**87.01**. PACS therefore begins at joint 65.24 and **cannot be a standalone system** — no
coupling gain closes 5+ points to the 70.4 ensemble. Its only plausible use was as a
better-ranked *candidate source*, which §5–6 then tested directly and more cheaply.

---

## 5. Two bugs found before they could fake a result

Both would have produced a *plausible null* attributable to the hypothesis rather than to the
code. Both were caught by printing intermediate quantities, not the final metric.

**(a) Double zero-init dead branch.** The first version had a polarity→emissions feedback
gate with **both** the gate scalar `alpha` and its projection `fb` zero-initialised:

```
∂L/∂alpha = fb    = 0        ∂L/∂fb = alpha = 0
```

Both parameters had **identically zero gradient forever**. The branch was dead code that
looked principled and printed a plausible `alpha +0.000`. It was also applied only to gold
spans during training and **never at inference** — train/test inconsistent. Removed; the
joint margin is the actual coupling.

**(b) Vacuous batch-mean margin.** The hard-negative loss compared batch **means**:

```
lj = relu(margin + Un.mean() − Ug.mean())
```

That is one scalar constraint per batch, which ordinary training already satisfies. The
control arm logs the term even though `lam_joint = 0` there, and it read **0.0013 by epoch 8
and 0.0000 by epoch 12 without ever being optimised.** Running the coupled arm would have
returned "coupling does nothing" as an artifact of the loss. Replaced with a true pairwise
margin — every gold span against **its own** boundary errors,
`relu(m + U_neg − U_gold[owner]).mean()` — verified on a live batch (15 gold spans → 45
negatives, every negative owned by a gold span in the same sentence, spot-checks are genuine
boundary shifts).

**(c) Conceptual, also corrected.** The first formulation treated "0 of 4 towers correct" as
a ground-truth *indeterminability label*, which contradicts §D.11's own finding that residual
errors are **consensus** errors. `q` was made a Laplace-smoothed estimate,
`q̂ = (k+α)/(4+2α)`, and the loss given an absolute threshold rather than a pure ranking —
noting that mapping `d = k/4` to `u = 2q−1` changes **nothing** in a pairwise ranking loss,
since it is a monotone affine transform. The real content of the `q > 0.5` insight is a
threshold, which the pairwise form never had.

---

## 6. Follow-up probes: why PACS failed

### 6a. Predict `q` from the text instead (`experts/qpredict.py`)
Rather than force the extractor to internalise `q`, estimate it in a separate model and apply
it at decision time. Trained on **3635 OOF MATE candidates** (gold span → `q`, non-gold span
→ 0; 815 non-gold, mean target 0.563, 36.2% below the 0.353 cut), 4 epochs, loss 0.686 →
0.465. Consumed by `decide.py --qhat`.

| acceptance score | dev | **test** |
|---|---|---|
| geometric product (standing) | **70.16** | **70.43** |
| q̂ alone (`--qhat-mix 1.0`) | 69.52 | 69.81 |
| q̂ × product (`--qhat-mix 0.5`) | 70.03 | 69.96 |

Below baseline. The reason is directly measurable:

| score | AUC for "this (span, polarity) pair is correct" |
|---|---|
| **oracle `q`** — other classifiers **+ gold labels** | **0.906** |
| geometric product (standing) | 0.7334 |
| **learned q̂ from the text** | **0.7049** |
| span evidence S | 0.6938 |
| MASC max probability | 0.6605 |

The learned predictor is **worse than the product it replaces**.

### 6b. Use member disagreement — a feature §D.2 never tested
§D.2's eleven features used the entropy of the **averaged** distribution, never inter-member
disagreement. At test time we do have 19 members' behaviour, so this is the computable proxy
for whatever scores 0.906.

| score | AUC | dev | **test F1** |
|---|---|---|---|
| geometric product | 0.7334 | 70.16 | **70.43** |
| member agreement alone | 0.6502 | — | — |
| member variance | 0.6609 | — | — |
| **product × agreement** | **0.7670** | 70.19 | 69.70 |
| product² × agreement | — | 70.18 | 70.06 |
| product³ × agreement | — | 70.19 | 70.64 |

**+0.034 AUC converts to nothing.** Dev is flat at 70.16–70.19 while test scatters
69.70–70.64, entirely inside the ±1.31 floor. Third independent confirmation of §C.24's
"improving the judge's AUC by +0.02 did not convert into F1".

---

## 7. ★ The mechanism — why the 74.68 oracle is unreachable

| source of `q` | AUC |
|---|---|
| other classifiers **compared to the gold labels** | **0.906** |
| the geometric product | 0.733 |
| a model reading the **text** | 0.705 |
| the same ensemble's **disagreement, without labels** | 0.650 |

**`q` is a property of model competence, not a readable property of the input.** It is
recoverable at 0.906 only by running independent classifiers *and comparing them to the
answer*. Strip the labels and the identical ensemble's disagreement carries 0.650 — barely
above MASC confidence.

So **74.68 was never a target**: the information that would fix the selection exists only
once you already know the answer. This is the deepest statement of the §D.2 ceiling (a
selector fitted directly on test cannot beat the honest one), and it reconciles with §D.11
rather than contradicting it — the residual errors are consensus errors **and** the consensus
cannot identify which of its own answers are the wrong ones.

---

## 8. What this closes, and what it does not

**Closed.** The extraction↔polarity coupling route, by three independent attacks: internalise
`q` in the extractor (−2.22), predict `q` from text (AUC 0.705 < 0.733), estimate it from
member disagreement (+0.034 AUC → 0.0 F1).

**Not overturned.** §C.18 remains a true and interesting observation about the system — the
extractor really does select the spans its classifier handles worst. What PACS establishes is
that the anti-correlation is **not actionable**, because acting on it requires a competence
estimate that only the labels provide.

**Standing result unchanged: 70.32** (choice-free, all correct members equal-weight) /
**70.62** (fixed 19-member structure). 15 of 19 t2015 baselines cleared; SGBIS 71.1,
DQPSA 71.9, VLHA 72.5, MADSC 72.9 remain above.

---

## 9. Artifacts

| path | contents |
|---|---|
| `experts/pacs.py` | PACS: shared encoder, BIO-CRF anchor head + span-polarity head, joint margin, determinability ranking (`--lam-det`, `--det-mode utility\|rank`) |
| `experts/qpredict.py` | text model predicting P(pair correct) on OOF MATE candidates |
| `experts/decide.py --qhat` | consumes q̂ as the acceptance score (`--qhat-mix`) |
| `data/qdet_train.npz` | `q` per train aspect (mean OOF P(gold) over 4 towers) + vote counts |
| `runs/d22_pacs_nojoint/` | the control arm (MATE@τ 84.83, `a_selected` 76.91, joint 65.24) |
| `runs/d22c_det/` | determinability-ranked arm (MATE@τ 85.17, `a_selected` 74.00, joint 63.02) |
| `runs/qpred_btwL/` | q̂ scores for dev/test |
| `runs/queue42.sh` · `queue43.sh` · `queue44.sh` · `queue45.sh` | the four PACS/q̂ job scripts, in order |

**Reproduce the arithmetic without a GPU:**
`python3 experts/decide.py --pool pools/final19 --w-grid 0.0` → dev 70.16 / test 70.43.
