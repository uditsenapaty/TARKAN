"""Wait for the Qwen weight download to complete, then run a 4-sample end-to-end smoke.
Exits (notifying the main loop) once the smoke finishes so the real run can be launched."""
import json, os, subprocess, sys, time

MODEL = "/teamspace/studios/this_studio/mllm/qwen2_5_vl_7b"


def complete():
    idx = os.path.join(MODEL, "model.safetensors.index.json")
    cfg = os.path.join(MODEL, "config.json")
    if not (os.path.exists(idx) and os.path.exists(cfg)):
        return False, "no index/config yet"
    try:
        wm = json.load(open(idx))["weight_map"]
    except Exception as e:
        return False, f"index unreadable {e}"
    shards = set(wm.values())
    for s in shards:
        p = os.path.join(MODEL, s)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            return False, f"missing shard {s}"
    # no in-flight downloads
    cache = os.path.join(MODEL, ".cache")
    inflight = []
    for root, _, files in os.walk(cache) if os.path.isdir(cache) else []:
        inflight += [f for f in files if f.endswith((".incomplete", ".lock"))]
    if inflight:
        return False, f"{len(inflight)} in-flight"
    return True, f"{len(shards)} shards ready"


def main():
    print("watcher: waiting for", MODEL, flush=True)
    stable = 0
    while True:
        ok, msg = complete()
        if ok:
            stable += 1
            print(f"watcher: complete check {stable}/2 ({msg})", flush=True)
            if stable >= 2:
                break
        else:
            stable = 0
            print(f"watcher: waiting ({msg})", flush=True)
        time.sleep(25)
    print("watcher: weights ready -> running smoke", flush=True)
    r = subprocess.run(
        [sys.executable, "-u", "mllm/train_qwen.py", "--dataset", "twitter2015",
         "--model", MODEL, "--out", "mllm/runs/smoke", "--limit", "4",
         "--epochs", "1", "--grad-accum", "2", "--eval-every", "1"],
        cwd="/teamspace/studios/this_studio")
    print("SMOKE_EXIT", r.returncode, flush=True)


if __name__ == "__main__":
    main()
