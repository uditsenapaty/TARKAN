# Issue-folder audit

Reference: the five supplied screenshots issues/s1.jpg through issues/s5.jpg.
The paper and screenshots are local review inputs, not republished here.

| Finding | Resolution |
|---|---|
| Teacher Qwen / Llama-3.1 mismatch | Meta-Llama-3-8B-Instruct checkpoint in defaults, profile, teacher and README |
| Teacher 4-bit vs 8-bit | Explicit 8-bit BitsAndBytes loading, with no silent precision fallback |
| BLIP-large vs base | BLIP-base throughout; checkpoint-specific caption cache |
| T4 vs A100 | A100 default and runtime hardware check; A100 setup commands |
| Three losses vs six | Eq. 24 documented and implemented, with tests for all six weighted components |
| Missing auxiliary ASC description | Linear ASC on fused aspects, training-only and masked to matched anchors |
| Two-stage extraction/re-tagging description | One final unified BIO sequence after predicted-anchor evidence qualification |
| 30 vs 10 epochs | Default/profile maximum 10; patience 3 on dev joint F1 |
| Unfrozen CLIP | CLIP and BERTweet backbones frozen; projection/head parameters remain trainable |
| Evidence dropout 0.2 undocumented | Default/profile zero; nonzero values labeled experimental |
| Optional CRF / alternate sentiment heads | Dead or unsupported paths removed; unknown switches fail validation |
| Outdated equation/table comments | Updated against final PDF; equation map in reproducibility guide |
| No loadable paper configuration | Strict JSON/YAML loader shared by CLI; profile saved in checkpoints |
| Training/eval/ablation reproducibility | CLI paths, checkpoint config, manifests, loss logs, explicit ablation names and regression tests |

Additional correctness fixes found while applying the audit:

- Training now uses predicted anchors with one-to-one descending IoU matching,
  rather than directly feeding gold aspect spans to the student.
- ACEQ respects missing-label masks, unavailable negatives, stream ablations,
  current-aspect re-scoring of fallback KG negatives, and singleton image batches.
- KAN configuration reaches the instantiated backend; missing B-splines fail.
- Evaluation uses the saved model config and visual concepts.
- Retrieval caches distinguish KG objects and ranking settings.
- Model/config mismatches and old teacher cache provenance fail explicitly.
- Broken references to unpublished setup scripts were replaced with prepare.py.

Historical settings and numerical reproduction remain unverified. Code edits
cannot prove what generated the paper's tables. See reproducibility.md for the
missing artifacts and explicitly recorded implementation choices.
