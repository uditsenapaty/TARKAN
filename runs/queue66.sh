#!/bin/bash
# Queue 66 — D.33g: recover the NEG that D.33 pays away.
#
# D.33's per-class table is a TRADE, not a lift: mean NEG -8.26 against NEU +2.85 and
# POS +2.10. The ensemble's NEG recall is 59.41 on 101 matched test aspects (41 errors),
# already its worst class. Fixing NEG to ~70 is worth ~+1.16 on `a` -- larger than the
# entire remaining gap to the bar.
#
# HYPOTHESIS, and it is falsifiable: the damage comes from the REVIEW corpora, whose NEG is
# lexically explicit ("the food was terrible"), not from Dong-2014, whose aspects are named
# entities annotated exactly like t2015's. If so, a Twitter-only stage 1 keeps the transfer
# and drops the NEG cost.
#
# Two arms, because they are not exclusive:
#   A. twitter-only towers, measured per class against the all-four towers
#   B. BOTH kinds in one pool -- supervision source is a decorrelation axis, and if the
#      twitter towers hold NEG while the all-four towers hold NEU/POS, the log-average
#      gets both. That is the axis D.32 tried with evidence source and failed; here the
#      members are individually BETTER, not worse, which is the difference.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr

echo "=== [A] twitter-only intermediate encoders ==="
pre () {   # $1=model $2=tag $3=lr $4=batch
  [ -f "runs/twx_enc_$2/enc.pt" ] && { echo "-- twx_enc_$2 exists"; return; }
  python3 experts/masc_text.py --model "$1" --batch "$4" --lr "$3" --epochs 6 \
    --pre-epochs 2 --pre-data twitter --pre-only --pre-save "runs/twx_enc_$2" \
    --out "runs/twx_enc_$2"
}
pre vinai/bertweet-large                             btwL 1e-5 8
pre microsoft/deberta-v3-large                       deb  1e-5 8
pre roberta-large                                    robL 1e-5 8
pre cardiffnlp/twitter-roberta-base-sentiment-latest twrob 2e-5 16

echo "=== [B] twitter-only towers ==="
tower () { python3 experts/masc_text.py --model "$1" --pre-init "runs/twx_enc_$2" \
             --seed "$5" --batch "$4" --lr "$3" --epochs 6 --spans $S \
             --out "runs/d33_tw_$2"; }
tower vinai/bertweet-large                             btwL  1e-5 8  90
tower microsoft/deberta-v3-large                       deb   1e-5 8  91
tower roberta-large                                    robL  1e-5 8  92
tower cardiffnlp/twitter-roberta-base-sentiment-latest twrob 2e-5 16 93

echo "=== [C] does twitter-only keep NEG? (the whole point) ==="
for b in btwL deb robL twrob; do
  echo "--- $b : all-four vs twitter-only"
  python3 experts/perclass.py --arms "runs/d33_x_$b" "runs/d33_tw_$b" --splits test 2>&1 | tail -4
done

echo "=== [D] pools ==="
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
KEEP="runs/masc_btw_s43 runs/masc_deb_sqrt \
runs/pdq_btwL_s47 runs/pdq_twrob_s43 runs/pdq_btwL_itcitm runs/pdq_twrob_itcitm \
runs/pdq_deb_itcitm runs/pdq_robL_itcitm \
runs/pds_res_btwL_wn0 runs/pds_res_deb_wn0 runs/pds_res_robL_wn0 \
runs/qpds_btwL_bal runs/qpds_deb_bal runs/qpds_robL_bal runs/masc_llm_s42"
X4="runs/d33_x_btwL runs/d33_x_deb runs/d33_x_robL runs/d33_x_twrob"
TW4="runs/d33_tw_btwL runs/d33_tw_deb runs/d33_tw_robL runs/d33_tw_twrob"

run () { python3 experts/pool.py --mate $MATE --masc $2 --rerank runs/rerank \
           --pdqmate runs/pdqmate_btwL --cand-thr 0.12 --out "pools/$1" > /dev/null
         printf "%-26s " "$1"
         python3 experts/decide.py --pool "pools/$1" --w-grid 0.0 2>&1 | grep "dev-best"; }

echo "    pool                          w    tau  dev F1 |  TEST P       R      F1  MATE@t      a"
run "i_swap-all4-19........" "$KEEP $X4"
run "i_swap-tw4-19........." "$KEEP $TW4"
run "i_BOTH-sources-23....." "$KEEP $X4 $TW4"
run "i_sources-only-8......" "$X4 $TW4"
