# TARKAN — Reproduced Results (paper format)

> Baseline rows are **copied verbatim** from the paper's LaTeX (exact published numbers).
> The **TARKAN** row is filled **only from real runs** on this machine (T4) — never fabricated.
> `PENDING` = run not yet complete. Teacher = `meta-llama/Llama-3.1-8B-Instruct` (4-bit, training-time only);
> image encoder = `openai/clip-vit-base-patch32`; text encoder = `vinai/bertweet-base`.
> KG: ConceptNet 5.7 EN (3.42M triples). SenticNet 7 pending its canonical source file.
>
> **Win criterion (user):** ≥90% of TARKAN's metric cells beat *every* referred baseline in that column.

---

## Table 1 — Main comparison results (%) on Twitter-2015 and Twitter-2017

| Modality | Method | Venue | T15 P | T15 R | T15 F1 | T17 P | T17 R | T17 F1 |
|---|---|---|---|---|---|---|---|---|
| Text | SPAN | ACL 2020 | 53.7 | 53.9 | 53.8 | 59.6 | 61.7 | 60.6 |
| Text | D-GCN | COLING 2020 | 58.3 | 58.8 | 59.4 | 64.2 | 64.1 | 64.1 |
| Text | RoBERTa | — | 61.8 | 65.3 | 63.5 | 65.5 | 66.9 | 66.2 |
| Text | BART | ACL 2021 | 62.9 | 65.0 | 63.9 | 65.2 | 65.6 | 65.4 |
| Text&Image | UMT+TomBERT | ACL 2021 | 58.4 | 61.3 | 59.8 | 62.3 | 62.4 | 62.4 |
| Text&Image | OSCGA+TomBERT | ACM MM 2020 | 61.7 | 63.4 | 62.5 | 63.4 | 64.0 | 63.7 |
| Text&Image | UMT-collapsed | ACL 2020 | 60.4 | 61.6 | 61.0 | 60.0 | 61.7 | 60.8 |
| Text&Image | OSCGA-collapsed | ACM MM 2020 | 63.1 | 63.7 | 63.2 | 63.5 | 63.5 | 63.5 |
| Text&Image | RpBERT-collapsed | AAAI 2021 | 49.3 | 46.9 | 48.0 | 57.0 | 55.4 | 56.2 |
| Text&Image | CLIP | ICML 2021 | 44.9 | 47.1 | 45.9 | 51.8 | 54.2 | 53.0 |
| Text&Image | JML | EMNLP 2021 | 65.0 | 63.2 | 64.1 | 66.5 | 65.5 | 66.0 |
| Text&Image | VLP-MABSA | ACL 2022 | 65.1 | 68.3 | 66.6 | 66.9 | 69.2 | 68.0 |
| Text&Image | CMMT | IPM 2022 | 64.6 | 68.7 | 66.5 | 67.6 | 69.4 | 68.5 |
| Text&Image | MultiPoint | ACM MM 2023 | — | — | 67.6 | — | — | 63.8 |
| Text&Image | M2DF | EMNLP 2023 | 67.0 | 68.3 | 67.6 | 67.9 | 68.8 | 68.3 |
| Text&Image | AoM | ACL 2023 | 67.9 | 69.3 | 68.6 | 68.4 | 71.0 | 69.7 |
| Text&Image | Atlantis | Inf. Fusion 2024 | 65.6 | 69.2 | 67.3 | 68.6 | 70.3 | 69.4 |
| Text&Image | MCPL-VLP | KBS 2024 | 67.2 | 69.2 | 68.2 | 69.0 | 69.4 | 69.2 |
| Text&Image | DQPS | AAAI 2024 | 71.7 | 72.0 | 71.9 | 71.1 | 70.2 | 70.6 |
| Text&Image | RNG | IEEE ICME 2024 | 67.8 | 69.5 | 68.6 | 69.5 | 71.0 | 70.2 |
| Text&Image | Vanesa | EMNLP 2024 | 68.6 | 71.1 | 69.8 | 69.2 | 72.1 | 70.6 |
| Text&Image | TCMT | ESWA 2025 | 69.3 | 70.4 | 69.8 | 70.2 | 71.5 | 70.8 |
| Text&Image | CORSA | COLING 2025 | 69.0 | 70.8 | 69.9 | 70.1 | 71.0 | 70.6 |
| Text&Image | SGBIS | KBS 2026 | 70.0 | 71.7 | 71.1 | 69.8 | 72.1 | 71.3 |
| Text&Image | VLHA | Pattern Recog. 2025 | 72.3 | 72.7 | 72.5 | 69.9 | 71.8 | 71.4 |
| **Ours** | **TARKAN** | — | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| _best baseline (to beat)_ | _per-column max_ | | _72.3_ | _72.7_ | _72.5_ | _71.1_ | _72.1_ | _71.4_ |

## Table 3 — Subtask & model-type comparison (%)

### MATE
| Method | T15 P | T15 R | T15 F1 | T17 P | T17 R | T17 F1 |
|---|---|---|---|---|---|---|
| JML | 83.6 | 81.2 | 82.4 | 92.0 | 90.7 | 91.4 |
| VLP-MABSA | 83.6 | 87.9 | 85.7 | 90.8 | 92.6 | 91.7 |
| CMMT | 83.9 | 88.1 | 85.9 | 92.2 | 93.9 | 93.1 |
| M2DF | 85.2 | 87.4 | 86.3 | 91.5 | 93.2 | 92.4 |
| AoM | 84.6 | 87.9 | 86.2 | 91.8 | 92.8 | 92.3 |
| Atlantis | 84.2 | 87.7 | 86.1 | 91.8 | 93.2 | 92.7 |
| DQPS | 88.3 | 87.1 | 87.7 | 95.1 | 93.5 | 94.3 |
| **TARKAN** | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| _best baseline_ | _88.3_ | _88.1_ | _87.7_ | _95.1_ | _93.9_ | _94.3_ |

### MASC (P/Acc, F1)
| Method | T15 Acc | T15 F1 | T17 Acc | T17 F1 |
|---|---|---|---|---|
| TomBERT | 77.2 | 71.8 | 70.5 | 68.0 |
| CapTrBERT | 78.0 | 73.2 | 72.3 | 70.2 |
| VLP-MABSA | 78.6 | 73.8 | 73.8 | 71.8 |
| M2DF | 78.9 | 74.8 | 74.3 | 73.0 |
| AoM | 80.2 | 75.9 | 76.4 | 75.0 |
| MCPL-VLP | 79.3 | 74.9 | 75.1 | 74.0 |
| TCMT | 81.4 | 76.7 | 77.3 | 75.8 |
| VLHA | 81.5 | 81.7 | 77.2 | 75.8 |
| **TARKAN** | PENDING | PENDING | PENDING | PENDING |
| _best baseline_ | _81.5_ | _81.7_ | _77.3_ | _75.8_ |

### LLM/MLLM
| Method | T15 P | T15 R | T15 F1 | T17 P | T17 R | T17 F1 |
|---|---|---|---|---|---|---|
| VisualGLM-6B | 69.2 | 64.6 | 66.8 | 57.2 | 52.0 | 54.5 |
| ChatGPT-3.5 | 66.3 | 66.3 | 66.3 | 58.9 | 58.9 | 58.9 |
| LLaMA+DPCI | 76.4 | 76.4 | 76.4 | 74.7 | 74.7 | 74.7 |
| ChatGPT-4V | 74.2 | 74.2 | 74.2 | 75.5 | 75.5 | 75.5 |
| SGBIS | 79.4 | 79.4 | 79.4 | 76.0 | 76.0 | 76.0 |
| **TARKAN** | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| _best baseline_ | _79.4_ | _79.4_ | _79.4_ | _76.0_ | _76.0_ | _76.0_ |

## Table 6 — Component ablation (F1)
| Variant | T15 F1 | T17 F1 |
|---|---|---|
| TARKAN (full) | PENDING | PENDING |
| w/o LLM teacher guidance | PENDING | PENDING |
| w/o aspect–visual relevance | PENDING | PENDING |
| w/o KG evidence filtering | PENDING | PENDING |
| w/o KG stream | PENDING | PENDING |
| w/o KAN fusion (MLP fusion) | PENDING | PENDING |
| w/o visual stream | PENDING | PENDING |
| w/o KAN-enhanced tag representation | PENDING | PENDING |
| _paper reference values_ | _74.1 / 72.8 / 72.5 / 72.9 / 72.2 / 73.0 / 71.6 / 73.4_ | _72.9 / 71.6 / 71.4 / 71.8 / 71.1 / 71.9 / 70.8 / 72.1_ |

## Table 10 — Fusion-strategy ablation (joint F1) — TARKAN's own variants (no external baselines)
| Fusion | T15 F1 | T17 F1 |
|---|---|---|
| Concatenation+Linear | PENDING | PENDING |
| Concatenation+MLP | PENDING | PENDING |
| Gated | PENDING | PENDING |
| Cross-modal attention | PENDING | PENDING |
| Bilinear | PENDING | PENDING |
| Tensor | PENDING | PENDING |
| **KAN (ours)** | PENDING | PENDING |

## Table 5 / 7 / 8 / 9 — diagnostics (TARKAN-only, no external baselines to beat)
- Table 5 (error distribution), Table 7 (teacher supervision quality), Table 8 (KG retrieval/filtering stats),
  Table 9 (perf by visual-relevance bucket): produced by `analysis/*.py` after training. PENDING.

---

### Status log
- 2026-06-30: data prepared (Table-2 verified), ConceptNet KG built (3.42M triples), teacher swapped to Llama-3.1-8B, deterministic battery green. Teacher labeling + training pending.
- 2026-06-30 late: SenticNet 7 added (user-provided zip) → full KG 5.11M triples (paper-faithful §3.2). Full teacher labeling done both datasets (rel pos-rate 9.9%/16.5%, KG pos-rate 0.6%/1.6%).
- 2026-07-01: faithful baseline measured — t2015 joint 61.19 / t2017 67.06 (0/6 Table-1 cells beat best baselines). Patch chase: 15 real runs on t2015; champion so far D2 (CRF + rich-ASC) joint 64.98 / MATE 83.99 / MASC 77.2. A1/A3/A5 measured and rolled back (hurt or flat on micro-joint). bertweet-large composition runs (E8/E9/T17) in flight. All numbers from real runs; full ledger in `possible-patches.md` + `iterations.csv`.
