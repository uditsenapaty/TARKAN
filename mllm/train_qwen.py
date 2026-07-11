"""LoRA SFT of Qwen2.5-VL (7B or 32B) for MABSA (patch A15). bs=1 + grad-accum (robust, no vision-batch
collation). Dev-selects the best adapter by joint F1. Works for 7B and 32B (add --grad-ckpt for 32B)."""
import argparse, json, os, random, sys, time, torch
sys.path.insert(0, "/teamspace/studios/this_studio")
import mllm.common as C
from qwen_vl_utils import process_vision_info
from mllm.eval_qwen import evaluate


def make_labeled(rec, processor):
    tgt = C.target_json(rec["words"], rec["gold"])
    user_msg = [{"role": "user", "content": [
        {"type": "image", "image": rec["image"]},
        {"type": "text", "text": C.INSTRUCTION + "\n\nTweet: " + rec["text"]}]}]
    full_msgs = user_msg + [{"role": "assistant", "content": [{"type": "text", "text": tgt}]}]
    full_text = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(full_msgs)
    full = processor(text=[full_text], images=imgs, videos=vids, return_tensors="pt")
    L_p = processor(text=[prompt_text], images=imgs, videos=vids, return_tensors="pt").input_ids.shape[1]
    labels = full["input_ids"].clone()
    labels[:, :L_p] = -100
    full["labels"] = labels
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--full-ft", action="store_true")     # full fine-tune LLM (freeze vision), 8-bit Adam
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--dev-limit", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)      # smoke
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)

    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    processor = AutoProcessor.from_pretrained(a.model, min_pixels=256 * 28 * 28, max_pixels=a.max_pixels)
    mkw = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        mkw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, **mkw)
    if a.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=a.grad_ckpt)
    model.config.use_cache = False
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if a.full_ft:
        for n, p in model.named_parameters():
            p.requires_grad = ("visual" not in n)      # fine-tune the LLM, freeze vision tower
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"full-FT: trainable {n_tr/1e9:.2f}B params (vision frozen)")
    else:
        lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    train = C.load_split(a.dataset, "train")
    dev = C.load_split(a.dataset, "dev")
    if a.limit:
        train, dev = train[:a.limit], dev[:max(4, a.limit)]
    if a.dev_limit:
        dev = dev[:a.dev_limit]

    params = [p for p in model.parameters() if p.requires_grad]
    if a.full_ft:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(params, lr=a.lr, weight_decay=a.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = (len(train) + a.grad_accum - 1) // a.grad_accum
    total_steps = steps_per_epoch * a.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)

    best_f, best_dir, log = -1.0, os.path.join(a.out, "best"), []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        random.shuffle(train)
        run_loss, seen = 0.0, 0
        opt.zero_grad()
        for i, rec in enumerate(train):
            try:
                b = make_labeled(rec, processor)
            except Exception as e:
                print("skip sample (build err):", repr(e)[:120]); continue
            b = {k: v.to(model.device) for k, v in b.items()}
            out = model(**b)
            loss = out.loss / a.grad_accum
            loss.backward()
            run_loss += out.loss.item(); seen += 1
            if (i + 1) % a.grad_accum == 0 or (i + 1) == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if seen % 100 == 0:
                print(f"ep{ep} it{i+1}/{len(train)} loss{run_loss/seen:.4f} "
                      f"lr{sched.get_last_lr()[0]:.2e} {(time.time()-t0)/60:.1f}m", flush=True)
        if (ep + 1) % a.eval_every == 0 or (ep + 1) == a.epochs:
            m, _, _ = evaluate(model, processor, dev)
            print(f"== ep{ep} DEV joint_F {m['joint_F']:.2f} mate_F {m['mate_F']:.2f} "
                  f"masc_acc {m['masc_acc']:.2f} ({(time.time()-t0)/60:.1f}m) ==", flush=True)
            log.append({"epoch": ep, "dev": m})
            if m["joint_F"] > best_f:
                best_f = m["joint_F"]
                model.save_pretrained(best_dir)
                if a.full_ft:
                    processor.save_pretrained(best_dir)   # so eval can load OUT/best directly
                json.dump({"epoch": ep, "dev": m}, open(os.path.join(best_dir, "select.json"), "w"))
                print(f"   -> new best dev joint_F {best_f:.2f} saved", flush=True)
    model.save_pretrained(os.path.join(a.out, "final"))
    json.dump(log, open(os.path.join(a.out, "trainlog.json"), "w"), indent=2)
    print(f"DONE best_dev_joint_F {best_f:.2f} adapter={best_dir} total {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
