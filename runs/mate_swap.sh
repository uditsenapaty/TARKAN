#!/bin/bash
# D.33h — does the extraction lever reach MATE@tau? CPU only, no new training.
#
# queue61 produces three intermediate-trained taggers (seeds 42/43/44). The standing MATE
# block is five deberta runs (3 seeds + 2 recipe probes). This substitutes the three seed
# members and leaves the two probes, so member count stays at five and only the supervision
# changes -- the same like-for-like discipline queue64 used on the polarity side.
#
# Run only after queue61 finishes. Every score is cached; this is seconds.
set -e
cd /teamspace/studios/this_studio

MASC="runs/masc_btw_s43 runs/masc_deb_sqrt \
runs/pdq_btwL_s47 runs/pdq_twrob_s43 runs/pdq_btwL_itcitm runs/pdq_twrob_itcitm \
runs/pdq_deb_itcitm runs/pdq_robL_itcitm \
runs/pds_res_btwL_wn0 runs/pds_res_deb_wn0 runs/pds_res_robL_wn0 \
runs/qpds_btwL_bal runs/qpds_deb_bal runs/qpds_robL_bal runs/masc_llm_s42 \
runs/d33_x_btwL runs/d33_x_deb runs/d33_x_robL runs/d33_x_twrob"

STD_MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
CTL_MATE="runs/d33_ctl_mate_s42 runs/d33_ctl_mate_s43 runs/d33_ctl_mate_s44 runs/mate_probeA runs/mate_probeB"
PRE_MATE="runs/d33_pre_mate_s42 runs/d33_pre_mate_s43 runs/d33_pre_mate_s44 runs/mate_probeA runs/mate_probeB"

run () { python3 experts/pool.py --mate $2 --masc $MASC --rerank runs/rerank \
           --pdqmate runs/pdqmate_btwL --cand-thr 0.12 --out "pools/$1" > /dev/null
         printf "%-24s " "$1"
         python3 experts/decide.py --pool "pools/$1" --w-grid 0.0 2>&1 | grep "dev-best"; }

echo "    pool                        w    tau  dev F1 |  TEST P       R      F1  MATE@t      a"
run "j_MATE-standing5......" "$STD_MATE"
run "j_MATE-d33ctl5........" "$CTL_MATE"
run "j_MATE-d33pre5........" "$PRE_MATE"

echo
echo "pool recall (the ceiling the 92 unreachable gold spans sit against):"
python3 - <<'PY'
import json
for n in ("j_MATE-standing5......", "j_MATE-d33ctl5........", "j_MATE-d33pre5........"):
    try:
        m = json.load(open(f"pools/{n}/meta.json"))
        for s in ("dev", "test"):
            d = m[s]
            print(f"  {n:24s} {s:4s} recall {100*d['n_gold_span_hits']/d['n_gold']:.2f}% "
                  f"({d['n_gold_span_hits']}/{d['n_gold']}, {d['n_cand']} candidates)")
    except FileNotFoundError:
        print(f"  {n}: not built")
PY
