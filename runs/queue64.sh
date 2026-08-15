#!/bin/bash
# Queue 64 — D.33e: the SWAP arm, which queue63 deliberately does not run.
#
# §D.32 measured the failure mode of the obvious design: adding three members to nineteen
# moved MATE@tau DOWN (87.80 -> 86.96) because their disagreement shifts the log-average
# enough to change which candidates clear tau. So "+ext-23" prices addition, and addition
# carries a dilution cost that has nothing to do with whether the new members are better.
#
# The clean comparison is like-for-like: drop the four standing text towers whose backbones
# the new ones reproduce (btwL, deb, robL, twrob) and put the externally-pretrained versions
# in their place. Member COUNT is unchanged at 19, so any delta is the supervision, not the
# ensemble size.
#
# CPU only -- every member score is already cached.
set -e
cd /teamspace/studios/this_studio

MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
KEEP="runs/masc_btw_s43 runs/masc_deb_sqrt \
runs/pdq_btwL_s47 runs/pdq_twrob_s43 runs/pdq_btwL_itcitm runs/pdq_twrob_itcitm \
runs/pdq_deb_itcitm runs/pdq_robL_itcitm \
runs/pds_res_btwL_wn0 runs/pds_res_deb_wn0 runs/pds_res_robL_wn0 \
runs/qpds_btwL_bal runs/qpds_deb_bal runs/qpds_robL_bal runs/masc_llm_s42"
OLD4="runs/masc_btwL_s45 runs/masc_deb_s44 runs/masc_robL_s46 runs/masc_twrob_s42"
NEW4="runs/d33_x_btwL runs/d33_x_deb runs/d33_x_robL runs/d33_x_twrob"

run () {   # $1=pool name  $2=masc list
  python3 experts/pool.py --mate $MATE --masc $2 \
    --rerank runs/rerank --pdqmate runs/pdqmate_btwL --cand-thr 0.12 \
    --out "pools/$1" > /dev/null
  printf "%-24s " "$1"
  python3 experts/decide.py --pool "pools/$1" --w-grid 0.0 2>&1 | grep "dev-best"
}

echo "    pool                        w    tau  dev F1 |  TEST P       R      F1  MATE@t      a"
run "g_standing-19......." "$KEEP $OLD4"
run "g_SWAP-4-ext-19....." "$KEEP $NEW4"
run "g_both-23..........." "$KEEP $OLD4 $NEW4"
run "g_ext-only-4........" "$NEW4"
