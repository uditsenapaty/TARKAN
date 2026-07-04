#!/bin/bash
# Eval-only: score AoM's OFFICIAL published checkpoint (AoM.pt) on t2015 test. ~10 min.
cd /teamspace/studios/this_studio/graft/AoM_full
../vlpenv/bin/python MAESC_training.py \
  --dataset twitter15 ./src/data/jsons/twitter15_info.json \
  --bart_model /teamspace/studios/this_studio/graft/bart-base \
  --model_config config/pretrain_base.json \
  --checkpoint_dir ./eval15 --log_dir 15_eval \
  --num_beams 4 --eval_every 1 --lr 7.5e-5 --batch_size 16 --epochs 35 \
  --grad_clip 5 --warmup 0.1 --seed 57 --rank 0 \
  --checkpoint ../aom_assets/AoM/checkpoint/pytorch_model.bin \
  --nn_attention_on --nn_attention_mode 0 --gcn_on --dep_mode 2 --sentinet_on \
  --trc_on --trc_pretrain_file TRC_ckpt/pytorch_model.bin \
  --no_train > ../aom_t15_official_eval.log 2>&1 &
echo "official-ckpt eval launched PID $! — result in ~10min:"
echo "grep -E 'aesc|ae_|sc_' /teamspace/studios/this_studio/graft/aom_t15_official_eval.log"
