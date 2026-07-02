#!/bin/bash
# AoM t2015 baseline fine-tune from the VLP-MABSA pretrained base (canonical hyperparams).
# Runs in the legacy vlpenv (py3.8/torch1.13/transformers3.4). Log: graft/aom_t15_baseline.log
cd /teamspace/studios/this_studio/graft/AoM_full
nohup ../vlpenv/bin/python MAESC_training.py \
  --dataset twitter15 ./src/data/jsons/twitter15_info.json \
  --checkpoint_dir ./train15 \
  --model_config config/pretrain_base.json \
  --log_dir 15_aesc \
  --num_beams 4 --eval_every 1 --lr 7.5e-5 --batch_size 16 --epochs 35 \
  --grad_clip 5 --warmup 0.1 --seed 57 --rank 2 \
  --checkpoint ../vlp_assets/VLP-MABSA/pytorch_model.bin \
  --nn_attention_on --nn_attention_mode 0 --gcn_on --dep_mode 2 --sentinet_on \
  > ../aom_t15_baseline.log 2>&1 &
echo "AoM t2015 baseline launched, PID $!"
echo "watch: tail -f /teamspace/studios/this_studio/graft/aom_t15_baseline.log"
