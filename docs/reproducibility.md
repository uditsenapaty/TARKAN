# Reproduction protocol

The supplied final TARKAN.pdf is the reference. The LaTeX bundle contains many
commented-out drafts, so only active manuscript statements were used. This
profile aligns code and documented settings; it cannot establish which settings
produced historical scores without the original experiment artifacts.

## Equation map

| Paper reference | Implementation |
|---|---|
| Section 3.2, Eqs. 1-2 | Frozen text/image backbones in encoders.py |
| Section 3.3, Eqs. 3-5 | Anchor head, pooling, anchor CE; anchors.py handles IoU 0.5 matching |
| Section 3.4, Eqs. 6-8 | Visual qualification and teacher BCE |
| Section 3.5, Eqs. 9-12 | Shared text encoding, linear triple projection, KG scores/BCE/pooling |
| Section 3.6, Eqs. 13-15 | ACEQ visual and KG ranking |
| Section 3.7, Eqs. 16-18 | B-spline KAN fusion |
| Section 3.8, Eqs. 19-23 | Fused token features, unified BIO, auxiliary ASC |
| Section 3.9, Eq. 24 | Six-term total in losses.py |
| Section 3.10 | Student-only final BIO inference |
| Table 2 | configs/information_fusion.yaml |
| Table 4 | Binary teacher prompts |
| Table 5 / Table 15 | Component ablations / fusion alternatives |

The supervised losses sum valid terms, matching the equations. The previous
implementation averaged terms and contained additional objectives/unused flags;
old checkpoints should be treated as a different experimental setup.

## Required local inputs

Supply the original dataset distribution and preserve train/dev/test splits:

    data/twitter2015/{train,dev,test}.tsv
    data/twitter2017/{train,dev,test}.tsv
    data/images/twitter2015/<image_id>
    data/images/twitter2017/<image_id>

Alternatively use four-line .txt files. TSV columns are ID, sentiment (0 NEG,
1 NEU, 2 POS), image ID, tweet with $T$ placeholder, and aspect text. The .txt
format is masked tweet, aspect, sentiment (-1 NEG, 0 NEU, 1 POS), image ID.
Records are grouped by reconstructed tweet/image to recover joint annotations.
Check resulting counts against Table 1 before comparing scores. Do not mix
dataset variants or move records across splits. Missing images fail explicitly.

Supply normalized English ConceptNet 5.7 and SenticNet 7 triples as JSONL:

    {"head":"phone","relation":"IsA","tail":"device","weight":1.0,"source":"conceptnet"}
    {"head":"excellent","relation":"HasPolarity","tail":"positive","weight":1.0,"source":"senticnet","polarity":0.9}

The optional polarity field belongs to the head concept. Resource downloads and
normalization must follow the source distributions; the repository does not
redistribute their snapshots or claim these examples form a usable KG.

    python prepare.py kg --triples data/kg_triples.jsonl
    python prepare.py concepts --dataset twitter2015
    python prepare.py teacher --dataset twitter2015
    python prepare.py concepts --dataset twitter2017
    python prepare.py teacher --dataset twitter2017

The KG builder refuses to overwrite an existing index. CLIP ranks KG vocabulary
concepts for all splits without using sentiment labels; BLIP captions and teacher
labels are generated only for the training split. The teacher is 8-bit and uses
deterministic decoding; malformed binary answers fail instead of becoming zero.
Teacher metadata identifies checkpoint, precision, prompts, retrieval settings,
and hashes of the training split, concept cache, and KG index. Old or mismatched
caches must be regenerated. Predicted anchors only inherit cached KG labels for
triples actually labeled for their IoU-matched gold aspect.

## Training, evaluation, and ablations

Run the README commands for each dataset and each seed in [42, 43, 44]. Select the
best checkpoint using development joint F1, then evaluate test once. Preserve all
three test JSON files and report mean and standard deviation. Do not select a
seed using test scores.

    python train.py --dataset twitter2017 --seed 43 --ablation no_aceq
    python evaluate.py --checkpoint results/twitter2017/no_aceq/seed_43/best.pt --output results/twitter2017/no_aceq/seed_43/test.json

Supported component ablations: full, no_teacher, no_aceq, no_relevance,
no_kg_filter, no_kg, no_visual, mlp_fusion, no_kan_tag.
The no_teacher variant disables visual/KG teacher BCE and both ACEQ terms.
The no_aceq variant retains the teacher BCE terms. Removed streams also remove
their qualification losses and counterfactuals. Ablations have separate run
directories, so they do not overwrite full-model checkpoints.

Every run saves its resolved configuration, dependency versions, observed GPU,
epoch losses, development metrics, and selected checkpoint. Evaluation restores
the checkpoint configuration and reloads CLIP concept caches. The A100 check is
enabled by default; CPU tests explicitly disable it without changing gpu=A100.

JMASA uses only predicted anchors and final seven-class BIO labels. Incompatible
I-tag polarity transitions open a new span, consistently with the shared BIO
decoder. MASC is a separate gold-anchor diagnostic using a deterministic majority
of word-level BIO polarity suffixes (POS, NEU, NEG tie order; no suffix => NEU).
Gold aspects completely beyond text truncation are omitted from MASC, and remain
false negatives in joint evaluation. This decoding policy is an implementation
choice, not a documented historical setting.

## Choices the manuscript does not resolve

The paper specifies three independent runs, but not the seeds or exact dependency
versions. Seeds 42/43/44 and requirements.txt are proposed reproducible choices.
Weight decay 0.01, linear warmup 0.1, clipping 1.0, FP16 student autocast, mean
aspect pooling, a 24-token triple limit, relation priors, and lexical/weight-based
retrieval ranking are explicit implementation choices inherited or clarified in
this repair. Retrieval ranking is a proxy, not a verified original semantic
retrieval implementation. Preserve the normalized resource snapshots and compare
these choices with original run manifests before claiming exact reproduction.

Evidence dropout is zero because it is not specified by the final manuscript.
RBF/rational/interaction KAN, Numberbatch, alternative fusion, and nonzero evidence
dropout are experimental configurations. CRF and alternate polarity-head switches
are unsupported by this manuscript implementation and rejected by its loader.
No unverified accuracy improvements or historical run measurements are asserted.

To verify Tables 3-15, the authors still need to supply original seeds, checkpoints,
resource snapshots, cached teacher labels, and logs, or rerun the specified
protocol and its diagnostics on A100. Several tables additionally require human
annotations, baseline outputs, and perturbation protocols absent from the repo.
