# Final two commands to start the AoM backbone baseline

All hardcoded author paths are patched, AoM's official checkpoints are downloaded (their trained t2015/t2017 models + TRC + configs), environment verified. Two things remain, run each line:

## 1. Fetch ResNet-152 ImageNet weights (official PyTorch CDN; the clone stripped binaries)

wget -q https://download.pytorch.org/models/resnet152-b121ed2d.pth -O /teamspace/studios/this_studio/graft/AoM_full/src/resnet/resnet152.pth && echo downloaded

## 2. Relaunch the AoM t2015 baseline

bash /teamspace/studios/this_studio/graft/run_aom_baseline.sh

Then watch with: tail -f /teamspace/studios/this_studio/graft/aom_t15_baseline.log

---
Status: previous launch died only on the missing senticnet path (now fixed repo-wide: 6 files patched, 0 hardcoded paths left). aom_assets verified complete: AoM-ckpt/Twitter2015/AoM.pt, Twitter2017/AoM.pt, TRC_ckpt, configs.

## NEXT COMMAND: TARKAN graft on t2015 (self-queues behind the t2017 baseline; safe to run now)
bash /teamspace/studios/this_studio/graft/run_tarkan_graft_t15.sh
