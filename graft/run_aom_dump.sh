#!/bin/bash
# Dump official-ckpt predictions (dev+test) for offline neurosymbolic tuning. ~15 min.
cd /teamspace/studios/this_studio/graft/AoM_full
rm -f ../dump_t15_dev.jsonl ../dump_t15_test.jsonl
AOM_DUMP_DEV=../dump_t15_dev.jsonl AOM_DUMP=../dump_t15_test.jsonl \
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
  --no_train > ../aom_t15_dump.log 2>&1
echo "DUMP_DONE"; wc -l ../dump_t15_dev.jsonl ../dump_t15_test.jsonl
