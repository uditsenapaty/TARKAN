"""Wait for the 32B download, smoke-test, then run the FULL t2015 32B fine-tune + eval.
One completion notification when the whole thing finishes (~3-3.5h) or on failure."""
import json, os, subprocess, sys, time

ROOT = "/teamspace/studios/this_studio"
MODEL = f"{ROOT}/mllm/qwen2_5_vl_32b"
LOG = f"{ROOT}/mllm/logs/t15_qwen32b.log"
OUT = f"{ROOT}/mllm/runs/t15_qwen32b"


def complete():
    idx = os.path.join(MODEL, "model.safetensors.index.json")
    cfg = os.path.join(MODEL, "config.json")
    if not (os.path.exists(idx) and os.path.exists(cfg)):
        return False, "no index/config"
    try:
        shards = set(json.load(open(idx))["weight_map"].values())
    except Exception as e:
        return False, f"index unreadable {e}"
    for s in shards:
        p = os.path.join(MODEL, s)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            return False, f"missing {s}"
    cache = os.path.join(MODEL, ".cache")
    if os.path.isdir(cache):
        for _, _, files in os.walk(cache):
            if any(f.endswith((".incomplete", ".lock")) for f in files):
                return False, "in-flight"
    return True, f"{len(shards)} shards"


def sh(cmd, logf):
    logf.write(f"\n$ {' '.join(cmd)}\n"); logf.flush()
    return subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT).returncode


def main():
    os.makedirs(f"{ROOT}/mllm/logs", exist_ok=True)
    print("watcher32b: waiting for", MODEL, flush=True)
    stable = 0
    while True:
        ok, msg = complete()
        stable = stable + 1 if ok else 0
        print(f"watcher32b: {'ready' if ok else 'waiting'} {stable}/2 ({msg})", flush=True)
        if stable >= 2:
            break
        time.sleep(20)
    with open(LOG, "w") as logf:
        py = sys.executable
        # guard hub pin (server-restore revert)
        subprocess.run([py, "-m", "pip", "install", "-q", "huggingface-hub>=0.34.0,<1.0"],
                       cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        print("watcher32b: smoke", flush=True)
        rc = sh([py, "-u", "mllm/train_qwen.py", "--dataset", "twitter2015", "--model", MODEL,
                 "--out", f"{ROOT}/mllm/runs/smoke32", "--limit", "2", "--epochs", "1",
                 "--grad-accum", "1", "--grad-ckpt", "--dev-limit", "2"], logf)
        if rc != 0:
            print(f"watcher32b: SMOKE_FAIL rc={rc}", flush=True); return
        subprocess.run(["rm", "-rf", f"{ROOT}/mllm/runs/smoke32"])
        print("watcher32b: smoke ok -> full t2015 32B run", flush=True)
        rc = sh([py, "-u", "mllm/train_qwen.py", "--dataset", "twitter2015", "--model", MODEL,
                 "--out", OUT, "--epochs", "3", "--lr", "1e-4", "--grad-accum", "16",
                 "--lora-r", "16", "--lora-alpha", "32", "--grad-ckpt",
                 "--dev-limit", "350", "--eval-every", "1"], logf)
        if rc != 0:
            print(f"watcher32b: TRAIN_FAIL rc={rc}", flush=True); return
        for split in ("test", "dev"):
            sh([py, "-u", "mllm/eval_qwen.py", "--dataset", "twitter2015", "--split", split,
                "--model", MODEL, "--adapter", f"{OUT}/best",
                "--dump", f"{ROOT}/mllm/preds/t15_qwen32b_{split}.jsonl"], logf)
        logf.write("\n############ DONE t15_qwen32b ############\n"); logf.flush()
    print("watcher32b: DONE_T15_32B", flush=True)


if __name__ == "__main__":
    main()
