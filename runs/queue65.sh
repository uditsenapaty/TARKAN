#!/bin/bash
# Queue 65 — D.33f: rebuild the REMAINING members on the same lever.
#
# §D.33d measured the absorption: four members each +0.63 better moved the ensemble's `a`
# by only +0.24, because a 19-member log-average absorbs ~80% of any member-level gain.
# The corollary is that the lever has to be applied to the whole ensemble, not a corner of
# it — 15 of the 19 members have still never seen an external aspect.
#
# 14 of those 15 are rebuildable. `masc_llm_s42` is a QLoRA'd Llama-8B decoder whose
# encoder has no counterpart here, so it is carried over unchanged.
#
# Every member keeps its ORIGINAL seed and recipe; the only change is --pre-init. That
# makes each one a paired comparison against the standing member of the same name.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr
PD=twitter,rest,laptop,mams

echo "=== [0] the one missing backbone encoder: bertweet-base ==="
if [ ! -f runs/ext_enc_btw/enc.pt ]; then
  python3 experts/masc_text.py --model vinai/bertweet-base --batch 16 --lr 2e-5 --epochs 6 \
    --pre-epochs 2 --pre-data $PD --pre-only --pre-save runs/ext_enc_btw \
    --out runs/ext_enc_btw
fi

echo "=== [1/3] text members (2) ==="
python3 experts/masc_text.py --model vinai/bertweet-base --pre-init runs/ext_enc_btw \
  --seed 43 --spans $S --out runs/e33_masc_btw_s43
python3 experts/masc_text.py --model microsoft/deberta-v3-large --pre-init runs/ext_enc_deb \
  --class-weight sqrt --seed 71 --batch 8 --lr 1e-5 --epochs 8 --spans $S \
  --out runs/e33_masc_deb_sqrt

echo "=== [2/3] PDS members (6) — cheaper, run first ==="
pds () {  # $1=out  $2=model  $3=encdir  $4=seed
  python3 experts/masc_pds.py --residual --model "$2" --pre-init "runs/ext_enc_$3" \
    --seed "$4" --lambda-pds 0.5 --w-none 0.0 --spans $S --out "runs/e33_$1"
}
qpds () { # $1=out  $2=model  $3=encdir
  python3 -u experts/masc_pds.py --residual --w-none 0 --pds data/pds_qwen/twitter2015 \
    --model "$2" --pre-init "runs/ext_enc_$3" --spans $S --out "runs/e33_$1" \
    2>&1 | grep -v "it/s\]" | tail -6
}
pds  pds_res_btwL_wn0 vinai/bertweet-large      btwL 82
pds  pds_res_deb_wn0  microsoft/deberta-v3-large deb  83
pds  pds_res_robL_wn0 roberta-large             robL 85
qpds qpds_btwL_bal    vinai/bertweet-large      btwL
qpds qpds_deb_bal     microsoft/deberta-v3-large deb
qpds qpds_robL_bal    roberta-large             robL

echo "=== [3/3] PDQ members (6) ==="
pdq () {  # $1=out $2=text-model $3=encdir $4=seed $5=lr $6=extra
  python3 experts/pdq.py --text-model "$2" --pre-init "runs/ext_enc_$3" --seed "$4" \
    --batch 8 --lr "$5" --epochs 8 --spans $S $6 --out "runs/e33_$1"
}
pdq pdq_btwL_s47      vinai/bertweet-large                             btwL  47 1e-5 ""
pdq pdq_twrob_s43     cardiffnlp/twitter-roberta-base-sentiment-latest twrob 43 2e-5 ""
pdq pdq_btwL_itcitm   vinai/bertweet-large                             btwL  60 1e-5 "--itc 1.0 --itm 1.0"
pdq pdq_twrob_itcitm  cardiffnlp/twitter-roberta-base-sentiment-latest twrob 61 2e-5 "--itc 1.0 --itm 1.0"
pdq pdq_deb_itcitm    microsoft/deberta-v3-large                       deb   62 1e-5 "--itc 1.0 --itm 1.0"
pdq pdq_robL_itcitm   roberta-large                                    robL  64 1e-5 "--itc 1.0 --itm 1.0"

echo "=== [4] FULL-REBUILD POOL ==="
for d in runs/e33_*; do
  for s in dev test; do
    [ -f "$d/probs_span_$s.npz" ] || { echo "MISSING $d/probs_span_$s.npz"; exit 1; }
  done
done
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
STANDING="runs/masc_btwL_s45 runs/masc_deb_s44 runs/masc_robL_s46 runs/masc_twrob_s42 \
runs/masc_btw_s43 runs/masc_deb_sqrt runs/pdq_btwL_s47 runs/pdq_twrob_s43 \
runs/pdq_btwL_itcitm runs/pdq_twrob_itcitm runs/pdq_deb_itcitm runs/pdq_robL_itcitm \
runs/pds_res_btwL_wn0 runs/pds_res_deb_wn0 runs/pds_res_robL_wn0 \
runs/qpds_btwL_bal runs/qpds_deb_bal runs/qpds_robL_bal runs/masc_llm_s42"
REBUILT="runs/d33_x_btwL runs/d33_x_deb runs/d33_x_robL runs/d33_x_twrob \
runs/e33_masc_btw_s43 runs/e33_masc_deb_sqrt \
runs/e33_pdq_btwL_s47 runs/e33_pdq_twrob_s43 runs/e33_pdq_btwL_itcitm \
runs/e33_pdq_twrob_itcitm runs/e33_pdq_deb_itcitm runs/e33_pdq_robL_itcitm \
runs/e33_pds_res_btwL_wn0 runs/e33_pds_res_deb_wn0 runs/e33_pds_res_robL_wn0 \
runs/e33_qpds_btwL_bal runs/e33_qpds_deb_bal runs/e33_qpds_robL_bal runs/masc_llm_s42"

run () { python3 experts/pool.py --mate $MATE --masc $2 --rerank runs/rerank \
           --pdqmate runs/pdqmate_btwL --cand-thr 0.12 --out "pools/$1" > /dev/null
         printf "%-26s " "$1"
         python3 experts/decide.py --pool "pools/$1" --w-grid 0.0 2>&1 | grep "dev-best"; }

echo "    pool                          w    tau  dev F1 |  TEST P       R      F1  MATE@t      a"
run "h_standing-19........." "$STANDING"
run "h_FULL-REBUILD-19....." "$REBUILT"
run "h_both-38............." "$STANDING $REBUILT"
