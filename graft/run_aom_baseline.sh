#!/bin/bash
# AoM backbone: (1) verify their OFFICIAL t2015 checkpoint by pure eval, then
# (2) train our own baseline from the VLP base + TRC ckpt (full AoM recipe, rank 0 = single T4).
# Log: graft/aom_t15_baseline.log
cd /teamspace/studios/this_studio/graft/AoM_full
PY=../vlpenv/bin/python
COMMON="--dataset twitter15 ./src/data/jsons/twitter15_info.json \
  --bart_model /teamspace/studios/this_studio/graft/bart-base \
  --model_config config/pretrain_base.json \
  --num_beams 4 --eval_every 1 --lr 7.5e-5 --batch_size 16 --epochs 35 \
  --grad_clip 5 --warmup 0.1 --seed 57 --rank 0 \
  --checkpoint ../aom_assets/AoM/checkpoint/pytorch_model.bin \
  --nn_attention_on --nn_attention_mode 0 --gcn_on --dep_mode 2 --sentinet_on \
  --trc_on --trc_pretrain_file TRC_ckpt/pytorch_model.bin"

{
  echo "############ PHASE 1: eval OFFICIAL AoM.pt (published 68.6) ############"
  $PY MAESC_training.py $COMMON --checkpoint_dir ./eval15 --log_dir 15_eval --no_train
  echo "############ PHASE 2: train our baseline from VLP base + TRC ############"
  $PY MAESC_training.py $COMMON --checkpoint_dir ./train15 --log_dir 15_aesc
  echo "############ AOM_T15_BASELINE_DONE ############"
} > ../aom_t15_baseline.log 2>&1 &
echo "AoM t2015 phase1(eval official)+phase2(train) launched, PID $!"
echo "watch: tail -f /teamspace/studios/this_studio/graft/aom_t15_baseline.log"
