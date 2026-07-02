#!/bin/bash
set -e
cd /teamspace/studios/this_studio
echo "[launcher] waiting for KG build to finish..."
until ! pgrep -f build_kg >/dev/null 2>&1; do sleep 5; done
echo "[launcher] KG composition:"
python3 -c "import sqlite3;c=sqlite3.connect('data/kg_index/kg.sqlite');[print(' ',r) for r in c.execute('select source,count(*) from triples group by source')];print('  total:',c.execute('select count(*) from triples').fetchone()[0])"
for ds in twitter2015 twitter2017; do
  echo "[launcher] === full train-split teacher labeling: $ds ==="
  python3 scripts/run_teacher_labeling.py --dataset $ds --splits train --device cuda --batch-size 32 --save-every 4000
done
echo "[launcher] ALL LABELING DONE"
