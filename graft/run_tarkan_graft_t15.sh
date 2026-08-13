#!/bin/bash
# TARKAN graft on AoM (t2015): warm-start from their official 68.42 model, fine-tune with
# teacher-prior-ranked KG evidence appended to inputs. Waits for the t2017 baseline to
# free the GPU. Two runs: graft fine-tune + no-evidence control at the same lr/epochs.
cd /teamspace/studios/this_studio/graft/AoM_full
PY=../vlpenv/bin/python
while pgrep -f "dataset twitter17" >/dev/null 2>&1; do sleep 300; done
BASE="--dataset twitter15 ./src/data/jsons/twitter15_info.json \
  --bart_model /teamspace/studios/this_studio/graft/bart-base \
  --model_config config/pretrain_base.json \
  --num_beams 4 --eval_every 1 --batch_size 16 \
  --grad_clip 5 --warmup 0.1 --seed 57 --rank 0 \
  --checkpoint ../aom_assets/AoM/checkpoint/pytorch_model.bin \
  --nn_attention_on --nn_attention_mode 0 --gcn_on --dep_mode 2 --sentinet_on \
  --trc_on --trc_pretrain_file TRC_ckpt/pytorch_model.bin"
{
  echo "############ GRAFT: warm-start official + KG evidence, lr 1e-5, 10 ep ############"
  AOM_INIT_STATE=../aom_assets/AoM/AoM-ckpt/Twitter2015/pytorch_model.bin \
  AOM_EVIDENCE=/teamspace/studios/this_studio/graft/evidence_t15_SPLIT.json \
  $PY MAESC_training.py $BASE --checkpoint_dir ./graft15 --log_dir 15_graft --lr 1e-5 --epochs 10
  echo "############ CONTROL: warm-start official, NO evidence, lr 1e-5, 10 ep ############"
  AOM_INIT_STATE=../aom_assets/AoM/AoM-ckpt/Twitter2015/pytorch_model.bin \
  $PY MAESC_training.py $BASE --checkpoint_dir ./ctrl15 --log_dir 15_ctrl --lr 1e-5 --epochs 10
  echo "############ TARKAN_GRAFT_T15_DONE ############"
} > ../tarkan_graft_t15.log 2>&1 &
echo "graft queued (waits for t2017 baseline) PID $!"
