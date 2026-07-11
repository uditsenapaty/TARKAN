"""Generate + score a Qwen2.5-VL adapter on a split. evaluate() is reused by train_qwen for dev-select."""
import argparse, json, os, sys, torch
sys.path.insert(0, "/teamspace/studios/this_studio")
import mllm.common as C
from qwen_vl_utils import process_vision_info


def build_prompt_messages(rec):
    return [{"role": "user", "content": [
        {"type": "image", "image": rec["image"]},
        {"type": "text", "text": C.INSTRUCTION + "\n\nTweet: " + rec["text"]}]}]


@torch.no_grad()
def evaluate(model, processor, records, max_new_tokens=256, dump=None):
    model.eval()
    gen_pairs, raw = [], []
    for rec in records:
        msgs = build_prompt_messages(rec)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = processor(text=[text], images=imgs, videos=vids, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, temperature=None, top_p=None, top_k=None)
        gen = out[:, inputs.input_ids.shape[1]:]
        txt = processor.batch_decode(gen, skip_special_tokens=True)[0]
        gen_pairs.append(C.parse_pairs(txt)); raw.append(txt)
    m = C.score(records, gen_pairs)
    if dump:
        with open(dump, "w") as f:
            for rec, p, t in zip(records, gen_pairs, raw):
                f.write(json.dumps({"gold": rec["gold"], "pairs": p, "raw": t}, ensure_ascii=False) + "\n")
    return m, gen_pairs, raw


def load_model(model_path, adapter=None, max_pixels=512 * 28 * 28):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, processor


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    recs = C.load_split(a.dataset, a.split)
    if a.limit:
        recs = recs[:a.limit]
    model, processor = load_model(a.model, a.adapter)
    m, _, _ = evaluate(model, processor, recs, dump=a.dump)
    print(f"[{a.dataset}/{a.split} n={len(recs)}] "
          f"JOINT P{m['joint_P']:.2f} R{m['joint_R']:.2f} F{m['joint_F']:.2f} | "
          f"MATE F{m['mate_F']:.2f} | MASC acc{m['masc_acc']:.2f} macF1{m['masc_macF1']:.2f}")
    print(json.dumps(m))
