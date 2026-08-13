"""C6 — AADG: Aspect-Aware Description Generation (MADSC, Pattern Recognition 2026).

MADSC is the current t2015 SOTA (JMASA 72.9) and, notably, its extractor is only
MATE 86.60 -- *below* DQPSA (87.7) and VLHA (88.2). Its whole edge is polarity:
a = 72.9/86.60 ~= 84.2. Its stated cause is that generic MLLM captions are the wrong
auxiliary signal ("granularity mismatch"), which is precisely what Chapter B measured when
BLIP captions DILUTED the ensemble (71.17 vs 71.26). MADSC's fix is to rewrite the caption
so that object mentions are replaced by the *aspect* they ground to.

Pipeline (paper §3.5-3.6):
  1. D_raw = MLLM(image)                                  [`--stage captions`]
  2. O = object mentions in D_raw  (spaCy noun chunks + named entities, stopwords dropped)
  3. dual similarity, CLIP embeddings, cosine on L2-normalised vectors:
        sim_final(a,o) = alpha*sim(a,o) + beta*max_j[ sim(a,r_j) * sim(o,r_j) ]
     the indirect route forces aspect and object onto the SAME region, which is what makes
     it a grounding signal rather than a global-scene prior
  4. replace o with argmax_a sim(a,o) when it clears tau; for aspects with no confident
     match, append an explicit "not clearly visible" statement            [`--stage describe`]

Documented deviations (T4 / no-detector constraints):
  * regions: MADSC uses VinVL top-36 boxes. detectron2/VinVL is not installable here, so we
    substitute a **3x3 grid of crops + the full image = 10 pseudo-regions**. This keeps the
    property the paper argues actually matters (aspect and object must agree on the same
    LOCAL region) without the detector dependency.
  * captions: MADSC uses GPT-4o; their own ablation puts BLIP2 at -1.41 MABSA Mac-F1. We use
    an open captioner, so expect to sit below their reported numbers for this reason alone.
  * calibrator: MADSC learns (w_u, b_u) jointly with the task. Used as a plug-in here
    (their §4.6 shows AADG alone lifts frozen baselines by +1.4-2.4 Acc), so we threshold
    sim_final directly; the learned calibrator needs joint training and is left to the
    gated MASC member.

    python experts/aadg.py --stage captions
    python experts/aadg.py --stage describe --spans runs/mate_ens5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import DATA, image_path, load  # noqa: E402

CAPTIONER = "Salesforce/blip-image-captioning-base"
CLIP_ID = "openai/clip-vit-base-patch32"          # the paper's own visual encoder
STOP = {"a", "an", "the", "this", "that", "these", "those", "it", "there", "some",
        "front", "background", "image", "picture", "photo", "view", "side", "top"}


def out_dir(ds: str) -> Path:
    return DATA / "aadg" / ds


# --------------------------------------------------------------------------- #
# stage 1 — captions
# --------------------------------------------------------------------------- #
def stage_captions(ds: str, batch: int = 16):
    from transformers import BlipForConditionalGeneration, BlipProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = BlipProcessor.from_pretrained(CAPTIONER)
    model = BlipForConditionalGeneration.from_pretrained(
        CAPTIONER, dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device).eval()

    ids = []
    seen = set()
    for split in ("train", "dev", "test"):
        for inst in load(ds, split):
            if inst.image_id not in seen:
                seen.add(inst.image_id)
                ids.append(inst.image_id)

    caps: Dict[str, str] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        imgs = []
        for iid in chunk:
            try:
                imgs.append(Image.open(image_path(ds, iid)).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224), (128, 128, 128)))
        inp = proc(images=imgs, return_tensors="pt").to(device, model.dtype)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=30, num_beams=3)
        for iid, g in zip(chunk, proc.batch_decode(gen, skip_special_tokens=True)):
            caps[iid] = g.strip()
        if (i // batch) % 25 == 0:
            print(f"  caption {i+len(chunk)}/{len(ids)}", flush=True)

    d = out_dir(ds); d.mkdir(parents=True, exist_ok=True)
    json.dump(caps, open(d / "captions.json", "w"), indent=0)
    print(f"wrote {len(caps)} captions -> {d/'captions.json'}")
    print("sample:", list(caps.items())[:3])



def _clip_img(clip, inp):
    """transformers>=4.56 returns BaseModelOutputWithPooling from get_image_features
    instead of the projected tensor; project + L2-normalise ourselves."""
    o = clip.get_image_features(**inp)
    if not torch.is_tensor(o):
        o = o.pooler_output          # already projected to the shared 512-d space
    return torch.nn.functional.normalize(o, dim=-1)


def _clip_txt(clip, inp):
    o = clip.get_text_features(**inp)
    if not torch.is_tensor(o):
        o = o.pooler_output
    return torch.nn.functional.normalize(o, dim=-1)


# --------------------------------------------------------------------------- #
# stage 2 — region embeddings (3x3 grid + full image)
# --------------------------------------------------------------------------- #
def stage_regions(ds: str, batch: int = 64):
    from transformers import CLIPModel, CLIPProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()

    caps = json.load(open(out_dir(ds) / "captions.json"))
    ids = list(caps)
    R = 10                                   # 9 grid crops + full image
    feats = np.zeros((len(ids), R, clip.config.projection_dim), dtype=np.float32)

    buf, meta = [], []
    def flush():
        if not buf:
            return
        inp = proc(images=buf, return_tensors="pt").to(device)
        with torch.no_grad():
            e = _clip_img(clip, inp).cpu().numpy()
        for (ri, rj), v in zip(meta, e):
            feats[ri, rj] = v
        buf.clear(); meta.clear()

    for i, iid in enumerate(ids):
        try:
            im = Image.open(image_path(ds, iid)).convert("RGB")
        except Exception:
            im = Image.new("RGB", (224, 224), (128, 128, 128))
        W, H = im.size
        crops = [im]
        for gy in range(3):
            for gx in range(3):
                crops.append(im.crop((gx * W // 3, gy * H // 3,
                                      (gx + 1) * W // 3, (gy + 1) * H // 3)))
        for j, c in enumerate(crops):
            buf.append(c); meta.append((i, j))
            if len(buf) >= batch:
                flush()
        if i % 400 == 0:
            print(f"  regions {i}/{len(ids)}", flush=True)
    flush()

    np.savez_compressed(out_dir(ds) / "regions.npz", feats=feats,
                        ids=np.array(ids, dtype=object))
    print(f"wrote region feats {feats.shape} -> {out_dir(ds)/'regions.npz'}")


# --------------------------------------------------------------------------- #
# stage 3 — build D_aspect
# --------------------------------------------------------------------------- #
def _mentions(nlp, text: str) -> List[str]:
    doc = nlp(text)
    out, seen = [], set()
    for span in list(doc.noun_chunks) + list(doc.ents):
        toks = [t.text for t in span if not t.is_stop and t.text.lower() not in STOP
                and t.is_alpha]
        if not toks:
            continue
        m = " ".join(toks)
        if m.lower() not in seen:
            seen.add(m.lower())
            out.append(m)
    return out


def stage_describe(ds: str, spans_dir: str | None, alpha: float, beta: float,
                   tau_pct: float, batch: int = 256):
    """Two-pass so the operating point is set from DATA, not from a borrowed constant.

    MADSC thresholds `sim_adjusted = u * sim_final` at tau=0.6, where `u` comes from a
    calibrator trained jointly with the task. Without that calibrator the raw CLIP
    text-text cosine has almost no dynamic range on this data (~0.78-0.93 for ANY pair),
    so a fixed 0.6 fires on essentially every mention and the rewrite degenerates -- the
    first attempt produced "a Facebook in a baseball uniform talking to a Facebook of
    Facebook". Two fixes, both recorded as deviations:
      * tau is the `tau_pct` percentile of the sim distribution measured on TRAIN only
        (never dev/test), giving a controlled replacement rate instead of a blanket rewrite;
      * matching is greedy ONE-TO-ONE, so a single aspect cannot claim every mention.
    `u` is additionally exported as a TRAIN-ECDF rank in [0,1] rather than a raw cosine, so
    the gate's calibrator in masc_gated.py sees a score with real spread.
    """
    import spacy
    from transformers import CLIPModel, CLIPProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nlp = spacy.load("en_core_web_sm")
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()

    d = out_dir(ds)
    caps = json.load(open(d / "captions.json"))
    z = np.load(d / "regions.npz", allow_pickle=True)
    reg_feats, reg_ids = z["feats"], list(z["ids"])
    reg_index = {k: i for i, k in enumerate(reg_ids)}

    def embed(texts):
        vecs = []
        for i in range(0, len(texts), batch):
            inp = proc(text=[f"a photo of {t}" for t in texts[i:i + batch]],
                       return_tensors="pt", padding=True, truncation=True,
                       max_length=77).to(device)
            with torch.no_grad():
                vecs.append(_clip_txt(clip, inp).cpu().numpy())
        return np.concatenate(vecs) if vecs else np.zeros((0, 512), dtype=np.float32)

    # ---------------- pass 1: similarities for every split ----------------
    pre = {}
    for split in ("train", "dev", "test"):
        insts = load(ds, split)
        if split == "train" or spans_dir is None:
            cand = [[(a, b) for (a, b, _) in i.aspects] for i in insts]
        else:
            cand = [[tuple(x) for x in sp]
                    for sp in json.load(open(Path(spans_dir) / f"spans_{split}.json"))]
        all_terms, all_ments, layout = [], [], []
        for inst, sp in zip(insts, cand):
            terms = [" ".join(inst.tokens[a:b]) for (a, b) in sp]
            ments = _mentions(nlp, caps.get(inst.image_id, ""))
            layout.append((len(all_terms), len(terms), len(all_ments), len(ments)))
            all_terms += terms; all_ments += ments
        E_a, E_o = embed(all_terms), embed(all_ments)

        sims, extra = [], []
        for inst, sp, (ta, na, to, no) in zip(insts, cand, layout):
            A_, O_ = E_a[ta:ta + na], E_o[to:to + no]
            R = reg_feats[reg_index[inst.image_id]] if inst.image_id in reg_index else None
            if na and no:
                direct = A_ @ O_.T
                if R is not None:
                    sa, so = A_ @ R.T, O_ @ R.T
                    indirect = np.max(sa[:, None, :] * so[None, :, :], axis=-1)
                else:
                    indirect = np.zeros_like(direct)
                sim = alpha * direct + beta * indirect
            else:
                sim = np.zeros((na, no), dtype=np.float32)
                direct = sim
            sims.append(sim)
            extra.append((A_ @ R.T if (R is not None and na) else None, direct))
        pre[split] = dict(insts=insts, cand=cand, layout=layout, sims=sims, extra=extra,
                          terms=all_terms, ments=all_ments)

    # tau + ECDF from TRAIN only -> no dev/test information enters the operating point
    train_vals = np.concatenate([s.ravel() for s in pre["train"]["sims"] if s.size]) \
        if any(s.size for s in pre["train"]["sims"]) else np.array([0.0])
    tau = float(np.percentile(train_vals, tau_pct))
    ecdf = np.sort(train_vals)
    print(f"tau = {tau:.4f} ({tau_pct:.0f}th pct of {len(train_vals)} TRAIN pairs); "
          f"sim range [{train_vals.min():.3f}, {train_vals.max():.3f}] "
          f"median {np.median(train_vals):.3f}", flush=True)

    # ---------------- pass 2: rewrite ----------------
    for split in ("train", "dev", "test"):
        P = pre[split]
        descs, rec_keys, rec_vis, rec_u, rec_y = [], [], [], [], []
        stats = {"replaced": 0, "absent": 0, "aspects": 0}
        for idx, (inst, sp, (ta, na, to, no)) in enumerate(
                zip(P["insts"], P["cand"], P["layout"])):
            raw = caps.get(inst.image_id, "")
            sim = P["sims"][idx]
            sa, direct = P["extra"][idx]
            terms = P["terms"][ta:ta + na]
            ments = P["ments"][to:to + no]
            Rf = reg_feats[reg_index[inst.image_id]] if inst.image_id in reg_index else None

            matched = {}
            if na and no:
                order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
                used_a, used_o = set(), set()
                for i_a, k_o in order:                     # greedy ONE-TO-ONE
                    if sim[i_a, k_o] <= tau:
                        break
                    if i_a in used_a or k_o in used_o:
                        continue
                    used_a.add(int(i_a)); used_o.add(int(k_o))
                    matched[int(i_a)] = int(k_o)
            new = raw
            for i_a, k_o in sorted(matched.items(), key=lambda kv: -len(ments[kv[1]])):
                new = new.replace(ments[k_o], terms[i_a], 1)
                stats["replaced"] += 1
            missing = [terms[i] for i in range(na) if i not in matched]
            if missing:
                new += " " + "; ".join(f"{m} is not clearly visible" for m in missing) + "."
                stats["absent"] += len(missing)
            stats["aspects"] += na
            descs.append(new)

            for i_a in range(na):
                best_r = int(np.argmax(sa[i_a])) if sa is not None else 0
                raw_u = float(np.max(sim[i_a])) if no else 0.0
                rec_keys.append((idx, sp[i_a][0], sp[i_a][1]))
                rec_vis.append(Rf[best_r] if Rf is not None
                               else np.zeros(reg_feats.shape[-1], dtype=np.float32))
                rec_u.append(float(np.searchsorted(ecdf, raw_u) / max(len(ecdf), 1)))
                t_norm = terms[i_a].strip().lower()
                hit = any(t_norm == m.strip().lower() for m in ments) or \
                    (no > 0 and float(np.max(direct[i_a])) >= 0.85)
                rec_y.append(1.0 if hit else 0.0)

        json.dump(descs, open(d / f"desc_{split}.json", "w"))
        np.savez_compressed(d / f"aspect_{split}.npz",
                            keys=np.array(rec_keys, dtype=np.int64),
                            vis=np.asarray(rec_vis, dtype=np.float32),
                            u=np.asarray(rec_u, dtype=np.float32),
                            y=np.asarray(rec_y, dtype=np.float32))
        print(f"[{split}] {len(descs)} desc | {stats['replaced']}/{stats['aspects']} aspects "
              f"grounded ({100*stats['replaced']/max(stats['aspects'],1):.1f}%), "
              f"{stats['absent']} absence notes | u mean {np.mean(rec_u):.3f} "
              f"| pseudo-pos {100*float(np.mean(rec_y)):.1f}%", flush=True)
    print("examples:", json.load(open(d / "desc_test.json"))[:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--stage", choices=["captions", "regions", "describe", "all"],
                    default="all")
    ap.add_argument("--spans", default=None)
    ap.add_argument("--alpha", type=float, default=0.7)   # MADSC Table 1, t2015
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--tau-pct", type=float, default=92.0,
                    help="percentile of the TRAIN similarity distribution used "
                         "as tau (raw CLIP cosines have no usable absolute scale)")
    args = ap.parse_args()
    if args.stage in ("captions", "all"):
        stage_captions(args.dataset)
    if args.stage in ("regions", "all"):
        stage_regions(args.dataset)
    if args.stage in ("describe", "all"):
        stage_describe(args.dataset, args.spans, args.alpha, args.beta, args.tau_pct)


if __name__ == "__main__":
    main()
