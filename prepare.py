"""Build local KG, CLIP concept caches, and train-only teacher labels."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
from config import load_config
from data import load_split, opinion_words
from kg import KnowledgeGraph, normalize
from kg_retrieval import AspectQuery, retrieve_triples, _triple_key
from utils import save_json


def build_index(source, cfg):
    """JSONL fields: head, relation, tail, weight, source; optional polarity."""
    destination = cfg.paths.kg_index / "kg.sqlite"
    if destination.exists():
        raise FileExistsError(f"Index already exists: {destination}; use a new output path.")
    connection = KnowledgeGraph.init_sqlite(str(destination))
    try:
        with Path(source).open(encoding="utf-8") as stream, connection:
            for line in stream:
                row = json.loads(line)
                if row["source"] not in cfg.kg_sources:
                    raise ValueError(f"Unsupported KG source: {row['source']}")
                h, t = normalize(row["head"]), normalize(row["tail"])
                values = (h, row["relation"], t, float(row.get("weight", 1)), row["source"])
                connection.execute("INSERT INTO triples VALUES (?,?,?,?,?)", values)
                connection.execute("INSERT INTO triples VALUES (?,?,?,?,?)",
                                   (t, row["relation"], h, values[3], row["source"]))
                if row.get("polarity") is not None:
                    connection.execute("INSERT OR REPLACE INTO polarity VALUES (?,?)",
                                       (h, float(row["polarity"])))
    finally:
        connection.close()


def concepts(dataset, cfg):
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from train import check_device, build_kg
    check_device(cfg)
    kg = build_kg(cfg)
    vocabulary = sorted({r[0] for r in kg._conn.execute(
        "SELECT head FROM triples UNION SELECT tail FROM triples")})
    if not vocabulary:
        raise ValueError("KG vocabulary is empty.")
    processor = CLIPProcessor.from_pretrained(cfg.visual_model_id, token=cfg.hf_token)
    model = CLIPModel.from_pretrained(cfg.visual_model_id, token=cfg.hf_token).to(cfg.device).eval()
    # Keep the vocabulary bank on CPU; score chunks to bound A100 memory use.
    bank = []
    with torch.no_grad():
        for start in range(0, len(vocabulary), 256):
            texts = [v.replace("_", " ") for v in vocabulary[start:start+256]]
            inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(cfg.device)
            features = model.get_text_features(**inputs)
            bank.append(torch.nn.functional.normalize(features, dim=-1).cpu())
    bank = torch.cat(bank)
    ids = sorted({inst.image_id for split in ("train", "dev", "test")
                  for inst in load_split(cfg.paths.data / dataset, split)})
    output = {}
    with torch.no_grad():
        for iid in ids:
            with Image.open(cfg.paths.data / "images" / dataset / iid) as im:
                inputs = processor(images=im.convert("RGB"), return_tensors="pt").to(cfg.device)
            vector = torch.nn.functional.normalize(model.get_image_features(**inputs), dim=-1).cpu()
            scores = (bank @ vector[0]).tolist()
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], vocabulary[i]))[:cfg.top_l_concepts]
            output[iid] = [vocabulary[i].replace("_", " ") for i in order]
    save_json(output, cfg.paths.data / "visual_concepts" / f"{dataset}.json")


def teacher_labels(dataset, cfg):
    from captioner import Captioner
    from teacher import TeacherCache, LLMTeacher
    from train import check_device, build_kg, load_visual_concepts
    check_device(cfg)
    instances = load_split(cfg.paths.data / dataset, "train")
    kg = build_kg(cfg)
    visual = load_visual_concepts(dataset, cfg)
    captioner = Captioner(cfg=cfg)
    descriptions = captioner.caption_dataset(
        dataset, sorted({i.image_id for i in instances}), cfg.paths.data / "images" / dataset)
    del captioner
    gc.collect()
    torch.cuda.empty_cache()
    teacher = LLMTeacher(cfg=cfg)
    rel_rows, kg_rows = [], []
    for instance in instances:
        tweet = " ".join(instance.tokens)
        for k, (s, e, _) in enumerate(instance.aspects):
            term = " ".join(instance.tokens[s:e])
            rel_rows.append(dict(instance_id=instance.id, aspect_idx=k,
                                 label=teacher.relevance_label(tweet, term, descriptions[instance.image_id])))
            query = AspectQuery(term, opinion_words(instance.tokens, (s, e), cfg.opinion_window),
                                visual[instance.image_id][:cfg.top_l_concepts])
            triples = retrieve_triples(query, kg, cfg=cfg)
            labels = teacher.kg_label_batch([(tweet, term, tr) for tr in triples])
            kg_rows.extend(dict(instance_id=instance.id, aspect_idx=k, triple_key=_triple_key(tr),
                                label=label) for tr, label in zip(triples, labels))
    TeacherCache.save(dataset, rel_rows, kg_rows, cfg=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("kg", "concepts", "teacher"))
    ap.add_argument("--config", default="configs/information_fusion.yaml")
    ap.add_argument("--dataset", choices=("twitter2015", "twitter2017"), default="twitter2015")
    ap.add_argument("--triples", help="Normalized ConceptNet 5.7 + SenticNet 7 JSONL")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage == "kg":
        if not args.triples:
            ap.error("--triples is required for kg")
        build_index(args.triples, cfg)
    elif args.stage == "concepts":
        concepts(args.dataset, cfg)
    else:
        teacher_labels(args.dataset, cfg)

if __name__ == "__main__":
    main()
