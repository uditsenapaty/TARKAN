"""Manuscript defaults (TARKAN.pdf, Table 2 and Eq. 24).

Unspecified choices are recorded in docs/reproducibility.md. These defaults are
a reproducible protocol, not evidence that historical table scores were rerun.
"""
from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env.local", override=False)
    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass

BIO_TAGS = ["O", "B-POS", "I-POS", "B-NEU", "I-NEU", "B-NEG", "I-NEG"]
TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}
ID2TAG = dict(enumerate(BIO_TAGS))
NUM_BIO_TAGS = len(BIO_TAGS)
ANCHOR_TAGS = ["O", "B-ASP", "I-ASP"]
ANCHOR2ID = {t: i for i, t in enumerate(ANCHOR_TAGS)}
NUM_ANCHOR_TAGS = len(ANCHOR_TAGS)
POLARITIES = ["POS", "NEU", "NEG"]
POL2ID = {p: i for i, p in enumerate(POLARITIES)}
ID2POL = dict(enumerate(POLARITIES))
NUM_POLARITIES = len(POLARITIES)
TSV_LABEL2POL = {0: "NEG", 1: "NEU", 2: "POS"}
TXT_LABEL2POL = {-1: "NEG", 0: "NEU", 1: "POS"}

@dataclass
class Paths:
    root: Path = _ROOT
    data: Path = _ROOT / "data"
    twitter2015: Path = _ROOT / "data/twitter2015"
    twitter2017: Path = _ROOT / "data/twitter2017"
    images2015: Path = _ROOT / "data/images/twitter2015"
    images2017: Path = _ROOT / "data/images/twitter2017"
    conceptnet: Path = _ROOT / "data/conceptnet"
    senticnet: Path = _ROOT / "data/senticnet"
    kg_index: Path = _ROOT / "data/kg_index"
    captions: Path = _ROOT / "data/captions"
    teacher_labels: Path = _ROOT / "data/teacher_labels"
    results: Path = _ROOT / "results"
    checkpoints: Path = _ROOT / "results/checkpoints"
    logs: Path = _ROOT / "results/logs"
    tables: Path = _ROOT / "results/tables"
    plots: Path = _ROOT / "results/plots"
    reports: Path = _ROOT / "results/reports"

@dataclass
class TarkanConfig:
    text_model_id: str = "vinai/bertweet-base"
    visual_model_id: str = "openai/clip-vit-base-patch32"
    teacher_llm_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    captioner_id: str = "Salesforce/blip-image-captioning-base"
    teacher_precision: int = 8
    freeze_text: bool = True
    freeze_visual: bool = True
    hidden_dim: int = 768
    max_text_len: int = 128
    num_visual_tokens: int = 49
    batch_size: int = 16
    learning_rate: float = 2e-5
    dropout: float = 0.3
    max_epochs: int = 10
    early_stop_patience: int = 3
    # Implementation choices not specified in the manuscript:
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    use_amp: bool = True
    num_workers: int = 4
    seed: int = 42
    seeds: tuple = (42, 43, 44)
    device: str = "cuda"
    gpu: str = "A100"
    require_a100: bool = True

    lambda_anc: float = 0.3
    lambda_asc: float = 0.5
    lambda1: float = 0.5  # lambda_rel (legacy public field name)
    lambda2: float = 0.3  # lambda_kg
    lambda_cf: float = 0.3
    aceq_mv: float = 0.2
    aceq_mg: float = 0.2
    aceq_eta: float = 1.0
    anchor_iou_threshold: float = 0.5

    top_m_triples: int = 10
    top_l_concepts: int = 5
    opinion_window: int = 3
    kg_rank_weights: tuple = (0.5, 0.3, 0.2)
    kg_eps: float = 1e-8
    entity_emb_dim: int = 300
    kg_encoder: str = "verbalized"
    kg_sources: tuple = ("conceptnet", "senticnet")
    fusion: str = "kan"
    kan_backend: str = "efficient_kan"
    kan_hidden: tuple = (256,)
    kan_grid_size: int = 5
    kan_spline_order: int = 3
    pool_mode: str = "mean"
    use_teacher: bool = True
    use_relevance: bool = True
    use_kg_filter: bool = True
    use_kg_stream: bool = True
    use_visual_stream: bool = True
    use_kan_tag_representation: bool = True
    # Experimental options, disabled in the manuscript profile.
    evidence_dropout: float = 0.0
    bootstrap_samples: int = 1000
    bootstrap_alpha: float = 0.05
    paths: Paths = field(default_factory=Paths)

    @property
    def hf_token(self) -> Optional[str]:
        return os.environ.get("HF_TOKEN")

    def to_dict(self):
        result = asdict(self)
        result["paths"] = {k: str(v) for k, v in result["paths"].items()}
        return result

def config_from_dict(values):
    values = dict(values)
    unknown = set(values) - {f.name for f in fields(TarkanConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if "paths" in values:
        values["paths"] = Paths(**{k: Path(v) for k, v in values["paths"].items()})
    for name in ("seeds", "kg_rank_weights", "kg_sources", "kan_hidden"):
        if name in values:
            values[name] = tuple(values[name])
    cfg = TarkanConfig(**values)
    if cfg.teacher_precision != 8:
        raise ValueError("The supported teacher protocol uses 8-bit quantization.")
    if cfg.max_epochs < 1 or cfg.batch_size < 1:
        raise ValueError("max_epochs and batch_size must be positive.")
    if not 0 <= cfg.anchor_iou_threshold <= 1:
        raise ValueError("anchor_iou_threshold must be in [0, 1].")
    return cfg

def load_config(path=None, **overrides):
    if path is None:
        values = {}
    else:
        content = Path(path).read_text(encoding="utf-8")
        try:
            values = json.loads(content)  # JSON is also valid YAML; bundled profile uses it.
        except json.JSONDecodeError:
            import yaml
            values = yaml.safe_load(content)
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a mapping.")
    values.update({k: v for k, v in overrides.items() if v is not None})
    return config_from_dict(values)

CONFIG = TarkanConfig()
DATASET_OVERRIDES = {}
def cfg_for(dataset, **extra):
    return replace(CONFIG, **extra)

