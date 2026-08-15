#!/bin/bash
# Queue 63 — D.33d: convert the tower-level gain into JOINT F1.
#
# A tower-level delta is not the deliverable; §C.24 and §D.24 both measured levers that
# improved a member and converted to nothing. This runs the conversion directly:
#   A. one intermediate encoder per BACKBONE (4 runs, not 19), all four corpora
#   B. new towers initialised from them, scored on the canonical candidate set
#      (runs/mate_ens5_hr: dev 1234 / test 1178, exactly the frozen pool's candidates)
#   C. rebuild the pool with the standing 19 + the new towers, then decide.py
#
# The added members are ADDITIONS, not replacements, so this measures the new information
# on top of everything already standing. §D.32 warns that adding members can cost MATE@tau
# by shifting the log-average; that is the risk this arm prices.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr
PD=twitter,rest,laptop,mams

pre () {   # $1=model $2=tag $3=lr $4=batch
  if [ -f "runs/ext_enc_$2/enc.pt" ]; then echo "-- ext_enc_$2 exists, skipping"; return; fi
  python3 experts/masc_text.py --model "$1" --batch "$4" --lr "$3" --epochs 6 \
    --pre-epochs 2 --pre-data $PD --pre-only --pre-save "runs/ext_enc_$2" \
    --out "runs/ext_enc_$2"
}

echo "=== [A] intermediate encoders, one per backbone ==="
pre vinai/bertweet-large                                    btwL  1e-5 8
pre microsoft/deberta-v3-large                              deb   1e-5 8
pre roberta-large                                           robL  1e-5 8
pre cardiffnlp/twitter-roberta-base-sentiment-latest        twrob 2e-5 16

echo "=== [B] towers initialised from them ==="
tower () {  # $1=model $2=tag $3=lr $4=batch $5=seed
  python3 experts/masc_text.py --model "$1" --pre-init "runs/ext_enc_$2" \
    --seed "$5" --batch "$4" --lr "$3" --epochs 6 --spans $S \
    --out "runs/d33_x_$2"
}
tower vinai/bertweet-large                             btwL  1e-5 8  90
tower microsoft/deberta-v3-large                       deb   1e-5 8  91
tower roberta-large                                    robL  1e-5 8  92
tower cardiffnlp/twitter-roberta-base-sentiment-latest twrob 2e-5 16 93

echo "=== [C] pool + decide ==="
# Guard: load_masc silently falls back from probs_span_* to probs_* (gold spans). A member
# that missed --spans would then cover only the matched candidates and look like a weak
# member rather than a broken one. Fail loudly instead.
for d in runs/d33_x_btwL runs/d33_x_deb runs/d33_x_robL runs/d33_x_twrob; do
  for s in dev test; do
    [ -f "$d/probs_span_$s.npz" ] || { echo "MISSING $d/probs_span_$s.npz"; exit 1; }
  done
done
echo "  all 4 new towers have span-scored probabilities"

STANDING="runs/masc_btwL_s45 runs/masc_deb_s44 runs/masc_robL_s46 runs/masc_twrob_s42 \
runs/masc_btw_s43 runs/masc_deb_sqrt runs/pdq_btwL_s47 runs/pdq_twrob_s43 \
runs/pdq_btwL_itcitm runs/pdq_twrob_itcitm runs/pdq_deb_itcitm runs/pdq_robL_itcitm \
runs/pds_res_btwL_wn0 runs/pds_res_deb_wn0 runs/pds_res_robL_wn0 \
runs/qpds_btwL_bal runs/qpds_deb_bal runs/qpds_robL_bal runs/masc_llm_s42"
NEW="runs/d33_x_btwL runs/d33_x_deb runs/d33_x_robL runs/d33_x_twrob"
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"

for name in "f_standing-19......" "f_+ext-23.........." "f_ext-only-4......."; do
  case "$name" in
    *standing*)  MASC="$STANDING" ;;
    *ext-23*)    MASC="$STANDING $NEW" ;;
    *ext-only*)  MASC="$NEW" ;;
  esac
  python3 experts/pool.py --mate $MATE --masc $MASC \
    --rerank runs/rerank --pdqmate runs/pdqmate_btwL --cand-thr 0.12 \
    --out "pools/$name" > /dev/null
  printf "%-22s " "$name"
  python3 experts/decide.py --pool "pools/$name" --w-grid 0.0 2>&1 | grep "dev-best"
done
