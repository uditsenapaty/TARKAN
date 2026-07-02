# Two commands for you to run (plain text, copy each line)

## 1. Download AoM's official checkpoints (their trained 68.6/69.7 models + TRC ckpt + configs)

cd /teamspace/studios/this_studio/graft && mkdir -p aom_assets && cd aom_assets && python3 -c "import gdown; gdown.download_folder('https://drive.google.com/drive/folders/13YovBuSu6jo9ctp7rJQm95aXsEeOHspV', quiet=False, use_cookies=False)"

## 2. Launch the AoM t2015 baseline training (permission system requires you to start it)

bash /teamspace/studios/this_studio/graft/run_aom_baseline.sh

---

### Status when you run these
- Legacy environment READY: python3.8 + torch1.13.1(CUDA) + transformers3.4 + fastNLP + spacy2.3.9 — all AoM imports verified passing
- AoM code complete (pinned commit), hardcoded author paths patched to our data
- VLP base checkpoint + 8288 t2015 images in place; GPU idle, waiting
- After baseline verifies: I graft TARKAN components (teacher-guided KG evidence, neurosymbolic rules, relevance supervision) fine-tuning FROM the strongest checkpoint, t2015 first, then t2017.

### t2015 no-backbone final verdict (for the record)
Best = DeBERTa all-in + rules: joint 66.60 / MATE 85.71 / MASC 77.05 — vs bar 72.5. Not beatable without the backbone; hence this track.
