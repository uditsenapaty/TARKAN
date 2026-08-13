#!/bin/bash
# Dump dev+test predictions for the graft/control/retrain checkpoints (ensemble members).
cd /teamspace/studios/this_studio/graft/AoM_full
PY=../vlpenv/bin/python
BASE="--dataset twitter15 ./src/data/jsons/twitter15_info.json \
  --bart_model /teamspace/studios/this_studio/graft/bart-base \
  --model_config config/pretrain_base.json \
  --num_beams 4 --eval_every 1 --lr 7.5e-5 --batch_size 16 --epochs 35 \
  --grad_clip 5 --warmup 0.1 --seed 57 --rank 0 \
  --checkpoint ../aom_assets/AoM/checkpoint/pytorch_model.bin \
  --nn_attention_on --nn_attention_mode 0 --gcn_on --dep_mode 2 --sentinet_on \
  --trc_on --trc_pretrain_file TRC_ckpt/pytorch_model.bin --no_train"
dump() { # name ckpt [evidence]
  local name=$1 ckpt=$2 ev=$3
  echo "############ DUMP $name ############"
  rm -f ../dump_t15_${name}_dev.jsonl ../dump_t15_${name}_test.jsonl
  local EV=""
  if [ -n "$ev" ]; then EV="/teamspace/studios/this_studio/graft/evidence_t15_SPLIT.json"; fi
  AOM_EVAL_STATEDICT=1 AOM_INIT_STATE=$ckpt AOM_EVIDENCE=$EV \
  AOM_DUMP_DEV=../dump_t15_${name}_dev.jsonl AOM_DUMP=../dump_t15_${name}_test.jsonl \
  $PY MAESC_training.py $BASE --checkpoint_dir ./evald --log_dir 15_dump
}
{
  CK_G=$(ls -t graft15/*/pytorch_model.bin 2>/dev/null | head -1); [ -f "$CK_G" ] || { echo "NO graft ckpt"; exit 1; }
  dump graft   "$CK_G" yes
  CK_C=$(ls -t ctrl15/*/pytorch_model.bin 2>/dev/null | head -1); [ -f "$CK_C" ] || { echo "NO control ckpt"; exit 1; }
  dump control "$CK_C"
  CK_R=$(ls -t train15/*/pytorch_model.bin 2>/dev/null | head -1); [ -f "$CK_R" ] || { echo "NO retrain ckpt"; exit 1; }
  dump retrain "$CK_R"
  echo "############ MEMBER_DUMPS_DONE ############"
  wc -l ../dump_t15_*_{dev,test}.jsonl
} > ../member_dumps.log 2>&1 &
echo "member dumps launched PID $! (~1h) — log: graft/member_dumps.log"
