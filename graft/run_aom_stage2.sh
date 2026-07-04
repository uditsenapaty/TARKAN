#!/bin/bash
# Stage 2: official-ckpt evals (both datasets) + t2017 baseline training. One launch.
cd /teamspace/studios/this_studio/graft/AoM_full
PY=../vlpenv/bin/python
FLAGS="--bart_model /teamspace/studios/this_studio/graft/bart-base \
  --model_config config/pretrain_base.json \
  --num_beams 4 --eval_every 1 --lr 7.5e-5 --batch_size 16 --epochs 35 \
  --grad_clip 5 --warmup 0.1 --seed 57 --rank 0 \
  --checkpoint ../aom_assets/AoM/checkpoint/pytorch_model.bin \
  --nn_attention_on --nn_attention_mode 0 --gcn_on --dep_mode 2 --sentinet_on \
  --trc_on --trc_pretrain_file TRC_ckpt/pytorch_model.bin"
{
  echo "############ EVAL OFFICIAL t2015 (expect ~68.6) ############"
  $PY MAESC_training.py --dataset twitter15 ./src/data/jsons/twitter15_info.json \
    --checkpoint_dir ./eval15 --log_dir 15_eval $FLAGS --no_train
  echo "############ EVAL OFFICIAL t2017 (expect ~69.7) ############"
  $PY MAESC_training.py --dataset twitter17 ./src/data/jsons/twitter17_info.json \
    --checkpoint_dir ./eval17 --log_dir 17_eval $FLAGS --no_train
  echo "############ TRAIN t2017 BASELINE (~10h) ############"
  $PY MAESC_training.py --dataset twitter17 ./src/data/jsons/twitter17_info.json \
    --checkpoint_dir ./train17 --log_dir 17_aesc $FLAGS
  echo "############ AOM_STAGE2_DONE ############"
} > ../aom_stage2.log 2>&1 &
echo "stage2 launched PID $! — log: graft/aom_stage2.log"
