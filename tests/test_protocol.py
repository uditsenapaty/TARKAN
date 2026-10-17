import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn

from anchors import decode_anchor_spans, match_anchors
from config import CONFIG, config_from_dict, load_config
from kg import KnowledgeGraph, Triple
from kg_retrieval import AspectQuery, retrieve_triples
from losses import aceq_loss, compute_losses
from models import TarkanStudent
from teacher import TeacherCache, build_targets, _parse_binary


def tiny_cfg(**kwargs):
    cfg = replace(CONFIG, hidden_dim=8, kan_hidden=(6,), kan_backend="rbf",
                  dropout=0.0, device="cpu", require_a100=False, num_workers=0)
    return replace(cfg, **kwargs)


def batch():
    return dict(
        input_ids=torch.zeros(2, 4, dtype=torch.long),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        tokens=[["good", "phone"], ["bad", "camera"]],
        word_ids=[[-1, 0, 1, -1], [-1, 0, 1, -1]], n_words=[2, 2],
        gold_aspects=[[(0, 2, "POS")], [(0, 2, "NEG")]],
        bio_labels=torch.tensor([[-100, 1, 2, -100], [-100, 5, 6, -100]]),
        anchor_labels=torch.tensor([[-100, 1, 2, -100], [-100, 1, 2, -100]]),
        instance_id=["a", "b"], image_id=["a.jpg", "b.jpg"], visual_concepts=[[], []])


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_profile(self):
        cfg = load_config("configs/information_fusion.yaml")
        self.assertEqual(cfg.to_dict(), CONFIG.to_dict())
        self.assertEqual(cfg.gpu, "A100")
        self.assertEqual(cfg.teacher_precision, 8)
        self.assertTrue(cfg.freeze_text and cfg.freeze_visual)
        self.assertEqual(cfg.max_epochs, 10)
        self.assertEqual(cfg.evidence_dropout, 0)
        self.assertEqual(config_from_dict(cfg.to_dict()).to_dict(), cfg.to_dict())
        with self.assertRaises(ValueError):
            config_from_dict({"use_crf": True})

    def test_anchor_one_to_one_iou(self):
        self.assertEqual(match_anchors([(0, 3), (0, 2), (7, 9)], [(0, 2, "POS")]),
                         [-1, 0, -1])
        self.assertEqual(match_anchors([(0, 1)], [(0, 2, "POS")]), [0])
        self.assertEqual(decode_anchor_spans([0, 1, 2, 0], [-1, 0, 1, -1], 2), [(0, 2)])

    def test_target_matching_masks_unmatched(self):
        tr = Triple("phone", "IsA", "device", 1, "conceptnet")
        b = batch()
        out = dict(tag_logits=torch.zeros(2, 4, 7),
                   aspect_word_spans=[[(0, 1), (1, 2)], []], kg_triples=[[tr], [tr]])
        cache = TeacherCache({("a", 0): 1}, {("a", 0, "phone|IsA|device"): 0})
        targets = build_targets(b, out, cache)
        self.assertEqual(targets["aspect_polarity"].tolist(), [0, -100])
        self.assertEqual(targets["teacher_relevance_mask"].tolist(), [True, False])
        self.assertEqual(targets["teacher_kg_mask"][1].tolist(), [False])

    def test_binary_parse_rejects_invalid_output(self):
        self.assertEqual(_parse_binary(" 1. "), 1)
        for value in ("10", "unknown", "0 or 1"):
            with self.assertRaises(ValueError):
                _parse_binary(value)

    def test_aceq_masks_unknown_negatives(self):
        scores = torch.tensor([0.4, 0.9], requires_grad=True)
        out = dict(tag_logits=torch.zeros(1, 1, 7), relevance=torch.tensor([0.5]),
                   cf_relevance=torch.tensor([0.8]), cf_relevance_mask=torch.tensor([True]),
                   kg_scores=[scores], cf_kg_scores=[torch.empty(0)])
        targets = dict(teacher_relevance=torch.tensor([1.0]),
                       teacher_relevance_mask=torch.tensor([False]),
                       teacher_kg=[torch.tensor([1., 0.])],
                       teacher_kg_mask=[torch.tensor([True, False])])
        self.assertEqual(aceq_loss(out, targets).item(), 0)
        targets["teacher_kg_mask"][0][1] = True
        loss = aceq_loss(out, targets)
        self.assertAlmostEqual(loss.item(), 0.7, places=6)
        loss.backward()
        self.assertLess(scores.grad[0].item(), 0)
        self.assertGreater(scores.grad[1].item(), 0)
        out["cf_kg_scores"] = [torch.tensor([0.6])]
        targets["teacher_kg_mask"][0][1] = False
        self.assertAlmostEqual(aceq_loss(out, targets).item(), 0.4, places=6)

    def test_six_loss_sum_and_no_teacher_ablation(self):
        out = dict(tag_logits=torch.randn(1, 2, 7, requires_grad=True),
                   anchor_logits=torch.randn(1, 2, 3, requires_grad=True),
                   asc_paper_logits=torch.randn(1, 3, requires_grad=True),
                   relevance=torch.tensor([0.4], requires_grad=True),
                   cf_relevance=torch.tensor([0.8], requires_grad=True),
                   cf_relevance_mask=torch.tensor([True]),
                   kg_scores=[torch.tensor([0.4, 0.7], requires_grad=True)],
                   cf_kg_scores=[torch.empty(0)])
        targets = dict(bio_labels=torch.tensor([[1, 2]]), anchor_labels=torch.tensor([[1, 2]]),
                       aspect_polarity=torch.tensor([0]), teacher_relevance=torch.tensor([1.]),
                       teacher_relevance_mask=torch.tensor([True]),
                       teacher_kg=[torch.tensor([1., 0.])],
                       teacher_kg_mask=[torch.tensor([True, True])])
        losses = compute_losses(out, targets)
        expected = (losses["l_tag"] + .3*losses["l_anc"] + .5*losses["l_asc"] +
                    .5*losses["l_rel"] + .3*losses["l_kg"] + .3*losses["l_cf"])
        torch.testing.assert_close(losses["total"], expected)
        losses["total"].backward()
        self.assertIsNotNone(out["asc_paper_logits"].grad)
        ablated = compute_losses(out, targets, replace(CONFIG, use_teacher=False))
        for key in ("l_rel", "l_kg", "l_cf"):
            self.assertEqual(ablated[key].item(), 0)
        no_cf = compute_losses(out, targets, replace(CONFIG, lambda_cf=0))
        self.assertEqual(no_cf["l_cf"].item(), 0)
        self.assertGreater(no_cf["l_rel"].item(), 0)

    @patch("data.opinion_words", return_value=[])
    def test_forward_backward_and_no_gold_leakage(self, _):
        model = TarkanStudent(tiny_cfg(), build_encoders=False)
        with torch.no_grad():
            model.anchor_head.weight.zero_()
            model.anchor_head.bias.copy_(torch.tensor([0., 0., 10.]))
        b = batch()
        text, vision = torch.randn(2, 4, 8), torch.randn(2, 3, 8)
        out = model(b, text, vision)
        self.assertEqual(out["aspect_word_spans"], [[(0, 2)], [(0, 2)]])
        targets = build_targets(b, out, TeacherCache({("a", 0): 1, ("b", 0): 1}))
        losses = compute_losses(out, targets, model.cfg)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertIsNotNone(model.anchor_head.weight.grad)
        self.assertIsNotNone(model.asc_head_paper.weight.grad)
        model.eval()
        first = model(b, text, vision)["tag_logits"]
        b["gold_aspects"] = [[], []]
        second = model(b, text, vision)["tag_logits"]
        torch.testing.assert_close(first, second)
        self.assertFalse(model(b, text, vision)["cf_relevance_mask"].any())

    @patch("data.opinion_words", return_value=[])
    def test_singleton_and_duplicate_images_skip_visual_cf(self, _):
        model = TarkanStudent(tiny_cfg(), build_encoders=False)
        b = batch()
        b["image_id"] = ["same.jpg", "same.jpg"]
        out = model(b, torch.randn(2, 4, 8), torch.randn(2, 3, 8), use_gold_anchors=True)
        self.assertFalse(out["cf_relevance_mask"].any())
        b = {k: v[:1] for k, v in b.items()}
        out = model(b, torch.randn(1, 4, 8), torch.randn(1, 3, 8), use_gold_anchors=True)
        self.assertFalse(out["cf_relevance_mask"].any())

    @patch("data.opinion_words", return_value=[])
    def test_empty_anchors_and_stream_ablations(self, _):
        for kwargs in ({}, {"use_visual_stream": False}, {"use_kg_stream": False},
                       {"use_kg_filter": False}, {"use_relevance": False}):
            model = TarkanStudent(tiny_cfg(**kwargs), build_encoders=False)
            with torch.no_grad():
                model.anchor_head.weight.zero_()
                model.anchor_head.bias.copy_(torch.tensor([10., 0., 0.]))
            b = batch()
            out = model(b, torch.randn(2, 4, 8), torch.randn(2, 3, 8))
            losses = compute_losses(out, build_targets(b, out, TeacherCache()), model.cfg)
            losses["total"].backward()
            self.assertTrue(torch.isfinite(losses["total"]))
            self.assertEqual(out["asc_paper_logits"].shape, (0, 3))

    def test_retrieval_cache_separates_graphs(self):
        one = KnowledgeGraph([("phone", "IsA", "device", 1, "conceptnet")])
        two = KnowledgeGraph([("phone", "IsA", "camera", 1, "conceptnet")])
        query = AspectQuery("phone")
        self.assertNotEqual(retrieve_triples(query, one), retrieve_triples(query, two))

    def test_frozen_backbones_keep_projections_trainable(self):
        from encoders import TextEncoder, VisualEncoder
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4))
                self.config = SimpleNamespace(hidden_size=4)
            def forward(self, **kwargs):
                return SimpleNamespace(last_hidden_state=self.weight.expand(1, 3, 4))
        factory = SimpleNamespace(from_pretrained=lambda *a, **k: Backbone())
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoModel=factory, CLIPVisionModel=factory)}):
            for cls, attr in ((TextEncoder, "bert"), (VisualEncoder, "clip")):
                enc = cls(hidden_dim=8, dropout=0, frozen=True)
                enc.train()
                self.assertFalse(getattr(enc, attr).training)
                args = (torch.ones(1, 3, dtype=torch.long),)*2 if cls is TextEncoder else (torch.zeros(1, 3, 4, 4),)
                enc(*args).sum().backward()
                self.assertIsNone(getattr(enc, attr).weight.grad)
                self.assertIsNotNone(enc.proj.weight.grad)
                trainable = cls(hidden_dim=8, frozen=False)
                self.assertTrue(getattr(trainable, attr).weight.requires_grad)

    def test_checkpoint_roundtrip(self):
        from utils import save_checkpoint
        cfg = tiny_cfg()
        model = TarkanStudent(cfg, build_encoders=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(model, path, {"config": cfg.to_dict()})
            payload = torch.load(path, weights_only=True)
            restored = TarkanStudent(config_from_dict(payload["config"]), build_encoders=False)
            restored.load_state_dict(payload["model_state"])
            self.assertEqual(restored.cfg.kan_backend, "rbf")

    def test_real_bspline_backend_and_config_width(self):
        from kan_fusion import KANFusion
        try:
            import efficient_kan
        except ImportError:
            self.skipTest("Install pinned efficient-kan to exercise B-spline backend")
        cfg = tiny_cfg(kan_backend="efficient_kan", kan_hidden=(5,))
        model = TarkanStudent(cfg, build_encoders=False)
        self.assertEqual(model.fusion.net.layers[0].out_features, 5)
        x = torch.randn(3, 8, requires_grad=True)
        model.fusion(x, x, x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    @patch("data.opinion_words", return_value=[])
    def test_kg_counterfactual_forward_and_gradient(self, _):
        graph = KnowledgeGraph([("good phone", "IsA", "device", 1, "conceptnet"),
                                ("bad camera", "HasProperty", "broken", 1, "senticnet")])
        model = TarkanStudent(tiny_cfg(), build_encoders=False, kg=graph)
        b = batch()
        out = model(b, torch.randn(2, 4, 8), torch.randn(2, 3, 8), use_gold_anchors=True)
        self.assertTrue(all(len(s) for s in out["cf_kg_scores"]))
        cache = TeacherCache({("a", 0): 1, ("b", 0): 1})
        for bidx, triples in enumerate(out["kg_triples"]):
            for tr in triples:
                cache.kg[(b["instance_id"][bidx], 0, f"{tr.head}|{tr.relation}|{tr.tail}")] = 1
        losses = compute_losses(out, build_targets(b, out, cache), model.cfg)
        losses["total"].backward()
        self.assertIsNotNone(model.kg_filter.wg.weight.grad)
        self.assertGreater(losses["l_kg"].item(), 0)

    @patch("data.opinion_words", return_value=[])
    def test_training_evaluation_checkpoint_pipeline(self, _):
        from config import Paths
        from train import train
        from evaluate import evaluate_all
        class FeatureStudent(TarkanStudent):
            def __init__(self, cfg, **kwargs):
                super().__init__(cfg, build_encoders=False, **kwargs)
            def forward(self, b, **kwargs):
                return super().forward(b, text_feats=b["test_text"],
                                       visual_feats=b["test_vision"], **kwargs)
        b = batch()
        b["test_text"], b["test_vision"] = torch.randn(2, 4, 8), torch.randn(2, 3, 8)
        with tempfile.TemporaryDirectory() as directory:
            cfg = tiny_cfg(max_epochs=1, use_teacher=False, use_kg_stream=False,
                           paths=replace(Paths(), results=Path(directory)))
            with patch("train.TarkanStudent", FeatureStudent), patch("train.make_loader", return_value=[b]):
                result = train(cfg)
            self.assertTrue(Path(result["checkpoint"]).exists())
            manifest = Path(directory) / "twitter2015/full/seed_42/manifest.json"
            self.assertEqual(json.loads(manifest.read_text())["config"]["gpu"], "A100")
            metrics = evaluate_all(result["model"], [b], "cpu")
            self.assertEqual(set(metrics), {"joint", "mate", "masc"})
            history = json.loads(manifest.with_name("history.json").read_text())
            self.assertEqual(len(history), 1)

    def test_a100_guard(self):
        from train import check_device
        with self.assertRaises(RuntimeError):
            check_device(replace(CONFIG, device="cpu"))

    def test_teacher_load_uses_exact_checkpoint_and_eight_bit(self):
        from teacher import LLMTeacher
        from unittest.mock import Mock
        tokenizer = SimpleNamespace(pad_token_id=0)
        generation = SimpleNamespace(pad_token_id=None)
        model = SimpleNamespace(eval=lambda: None, generation_config=generation)
        load_model, load_tokenizer = Mock(return_value=model), Mock(return_value=tokenizer)
        quantization = Mock(side_effect=lambda **kwargs: kwargs)
        fake = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
                               AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
                               BitsAndBytesConfig=quantization)
        with patch.dict("sys.modules", {"transformers": fake}):
            LLMTeacher()._load()
        quantization.assert_called_once_with(load_in_8bit=True)
        self.assertEqual(load_model.call_args.args[0], "meta-llama/Meta-Llama-3-8B-Instruct")
        self.assertEqual(load_model.call_args.kwargs["quantization_config"], {"load_in_8bit": True})

    def test_preparation_index_and_cache_identity(self):
        from config import Paths
        from prepare import build_index
        from teacher import cache_identity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = tiny_cfg(paths=replace(Paths(), data=root, kg_index=root/"kg"))
            source = root/"triples.jsonl"
            source.write_text(json.dumps(dict(head="Good Phone", relation="IsA", tail="Device",
                                              weight=1, source="conceptnet"))+"\n")
            build_index(source, cfg)
            graph = KnowledgeGraph(sqlite_path=str(root/"kg/kg.sqlite"))
            self.assertEqual(graph.neighbors("good phone")[0].tail, "device")
            self.assertEqual(graph.neighbors("device")[0].tail, "good_phone")
            graph.close()
            with self.assertRaises(FileExistsError):
                build_index(source, cfg)
            (root/"twitter2015").mkdir()
            (root/"twitter2015/train.tsv").write_text("original")
            (root/"visual_concepts").mkdir()
            (root/"visual_concepts/twitter2015.json").write_text("{}")
            first = cache_identity("twitter2015", cfg)
            (root/"twitter2015/train.tsv").write_text("changed")
            self.assertNotEqual(first, cache_identity("twitter2015", cfg))

if __name__ == "__main__":
    unittest.main()
