"""Build tweet-level KG evidence for AoM inputs (TARKAN graft, train/test consistent).

For each sample in AoM's {split}.json: extract noun phrases, retrieve triples from the
5.1M ConceptNet+SenticNet index, rank by retrieval_score * relation_prior (the prior is
distilled from the calibrated Llama-teacher labels — the teacher's judgment enters as a
test-legal static prior), keep top-K distinct, render as text.

Output: graft/evidence_<tag>_<split>.json  (list aligned with the dataset json order;
each entry the evidence string, possibly empty).
Run in the MAIN env: python3 graft/build_evidence.py twitter2015 t15
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kg import KnowledgeGraph  # noqa: E402
from kg_retrieval import AspectQuery, retrieve_triples  # noqa: E402

GRAFT = ROOT / "graft"
TOPK = 3


def load_prior():
    pr = {}
    for line in open(GRAFT / "relation_prior.tsv"):
        parts = line.rstrip("\n").split("\t")
        if parts[0] == "rel" or len(parts) < 2:
            continue
        try:
            pr[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return pr


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "twitter2015"
    tag = sys.argv[2] if len(sys.argv) > 2 else "t15"
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "ner"])
    kg = KnowledgeGraph(sqlite_path=str(ROOT / "data" / "kg_index" / "kg.sqlite"))
    prior = load_prior()
    data_dir = GRAFT / "AoM_full" / "src" / "data" / dataset

    for split in ("train", "dev", "test"):
        data = json.load(open(data_dir / f"{split}.json"))
        out = []
        for rec in data:
            words = rec["words"]
            doc = nlp(" ".join(words))
            phrases = list({c.text.strip() for c in doc.noun_chunks if len(c.text.strip()) > 2})[:6]
            cands = {}
            for ph in phrases:
                for tr in retrieve_triples(AspectQuery(aspect_term=ph), kg, 6):
                    key = f"{tr.head}|{tr.relation}|{tr.tail}"
                    score = tr.weight * prior.get(tr.relation, 0.15)
                    if key not in cands or score > cands[key][0]:
                        cands[key] = (score, tr)
            top = sorted(cands.values(), key=lambda x: -x[0])[:TOPK]
            ev = " ; ".join(f"{t.head.replace('_',' ')} {t.relation} {t.tail.replace('_',' ')}"
                            for _, t in top)
            out.append(ev)
        path = GRAFT / f"evidence_{tag}_{split}.json"
        json.dump(out, open(path, "w"))
        nonempty = sum(1 for e in out if e)
        print(f"{split}: {len(out)} samples, evidence for {nonempty} ({100*nonempty/len(out):.0f}%) -> {path.name}")


if __name__ == "__main__":
    main()
