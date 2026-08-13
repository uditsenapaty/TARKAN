#!/bin/bash
# Queue 17 — PDQ as the EXTRACTOR (C2b), the one built-but-never-run piece.
# Two reasons this is the top remaining lever:
#  (1) Chapter B: "MATE needs ARCH-DIVERSE strong members, not more DeBERTa seeds". Every
#      MATE member we have is a BIO/CRF tagger; this scores (start,end) cells directly with
#      the EPE bi-affine head, so BIO decode pathologies cannot occur. DQPSA reports 87.7.
#  (2) C18: our extractor selects spans that are HARDER for our classifier than the ones it
#      misses (80.52 vs 86.73). PDQ-MATE shares its mechanism and visual bridge with the
#      PDQ MASC members, so the spans it finds should be spans that stack understands --
#      attacking the coupling rather than either subtask in isolation.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/2] PDQ-MATE + bertweet-large ==="
python3 experts/pdq_mate.py --text-model vinai/bertweet-large --seed 42 \
  --lr 1e-5 --epochs 10 --patience 3 --out runs/pdqmate_btwL

echo "=== [2/2] PDQ-MATE + twitter-roberta ==="
python3 experts/pdq_mate.py --text-model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --seed 43 --lr 2e-5 --epochs 10 --patience 3 --out runs/pdqmate_twrob
echo "=== QUEUE17 DONE ==="
