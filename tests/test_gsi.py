from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from gsi.calibrate import (
    DEFAULT_EPSILON,
    analyze_matrix,
    basis_group_key,
    candidate_ranks,
    pool_module_captures,
    usable_rows,
)
from gsi.config import GSIConfig, GSIMode, _parse_layers
from gsi.evaluate import (
    break_even_fast_fraction,
    deployed_speedup,
    effective_speedup,
    generation_agreement,
    perplexity,
    perplexity_ratio,
    speedup_report,
    top1_agreement,
)
from gsi.expert_images import build_stage_images, plan_expert_chunks
from gsi.families import routed_family
from gsi.footprint import estimate
from gsi.expert import functional_expert_forward
from gsi.math import apply_gated_linear, project_and_gate
from gsi.profile import GSIEntry, GSIProfile, sha256_file
from gsi.runtime import GSIRuntime, forward_phase
from gsi.subspace import cascade_basis, subspace_overlap


class MathTests(unittest.TestCase):
    def test_projection_energy_matches_explicit_residual(self):
        torch.manual_seed(1)
        q, _ = torch.linalg.qr(torch.randn(12, 5))
        x = torch.randn(7, 12)
        result = project_and_gate(x, q, 0.5)
        explicit = x - (x @ q) @ q.T
        expected = explicit.square().sum(-1) / x.square().sum(-1)
        torch.testing.assert_close(result.rho2, expected, atol=2e-6, rtol=2e-6)

    def test_nan_and_zero_are_forced_slow(self):
        basis = torch.eye(4)[:, :2]
        x = torch.tensor([[0.0, 0, 0, 0], [float("nan"), 0, 0, 0]])
        result = project_and_gate(x, basis, 1.0)
        self.assertFalse(bool(result.fast_mask.any()))
        self.assertFalse(bool(result.valid_mask.any()))

    def test_mixed_gated_linear_uses_baseline_only_for_slow_rows(self):
        basis = torch.eye(4)[:, :2]
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        image = weight @ basis
        x = torch.tensor([[1.0, 2, 0, 0], [1.0, 2, 9.0, 0]])
        calls = []

        def baseline(value):
            calls.append(value.clone())
            return value @ weight.T

        output, gate = apply_gated_linear(
            x, basis, image, 0.1, baseline
        )
        torch.testing.assert_close(output, x @ weight.T)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].shape[0], 1)
        self.assertEqual(gate.fast_mask.tolist(), [True, False])

    def test_always_fast_negative_control(self):
        basis = torch.eye(4)[:, :1]
        x = torch.tensor([[1.0, 8.0, 0, 0]])
        result = project_and_gate(x, basis, 0.0, gate_override="always_fast")
        self.assertTrue(bool(result.fast_mask.item()))


class ProfileTests(unittest.TestCase):
    def test_profile_round_trip_and_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            basis = root / "basis.safetensors"
            image = root / "image.tp0.safetensors"
            save_file({"basis": torch.eye(4)[:, :2].contiguous()}, str(basis))
            save_file({"image": torch.ones(3, 2)}, str(image))
            entry = GSIEntry(
                module="model.layers.0.self_attn.q_b_proj",
                family="q_latent",
                layer=0,
                rank=2,
                epsilon=0.05,
                input_dim=4,
                output_dim=3,
                basis_file=basis.name,
                basis_sha256=sha256_file(basis),
                image_file="image.tp{tp_rank}.safetensors",
            )
            profile = GSIProfile.from_entries(
                model_id="test",
                model_hash="abc",
                quantization="fp8",
                tp_size=1,
                entries=[entry],
                calibration={"split": "calibration"},
            )
            path = root / "profile.json"
            profile.save(path)
            old_rank = os.environ.get("RANK")
            os.environ["RANK"] = "0"
            try:
                loaded = GSIProfile.load(path)
            finally:
                if old_rank is None:
                    os.environ.pop("RANK", None)
                else:
                    os.environ["RANK"] = old_rank
            self.assertEqual(loaded.entries[0], entry)


class CalibrationTests(unittest.TestCase):
    def test_candidate_grids(self):
        self.assertEqual(candidate_ranks(512), [32, 64, 128, 256])
        self.assertEqual(candidate_ranks(100), [32, 64])
        self.assertEqual(candidate_ranks(2048), [64, 128, 256, 512])
        self.assertEqual(candidate_ranks(6144), [128, 256, 512, 1024])

    def test_analysis_recovers_low_rank_matrix(self):
        torch.manual_seed(3)
        matrix = torch.randn(128, 3) @ torch.randn(3, 16)
        _, analysis = analyze_matrix(matrix, [2, 3, 4])
        rank3 = next(item for item in analysis if item["rank"] == 3)
        self.assertLess(rank3["mean_rho"], 1e-3)

    def test_unusable_rows_are_dropped_before_calibration(self):
        matrix = torch.tensor(
            [[1.0, 2.0], [float("nan"), 1.0], [0.0, 0.0], [3.0, 4.0]]
        )
        self.assertEqual(usable_rows(matrix).tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_pooling_drops_a_capture_that_is_entirely_unusable(self):
        captures = {
            "m|decode": torch.ones(2, 4),
            "m|target_verify": torch.full((3, 4), float("nan")),
        }
        pooled = pool_module_captures(captures)
        self.assertEqual(pooled["m"]["matrix"].shape, (2, 4))
        self.assertEqual(list(pooled["m"]["phases"]), ["decode"])

    def test_decode_and_verify_are_pooled_per_module(self):
        captures = {
            "model.layers.0.self_attn.q_b_proj|decode": torch.ones(2, 4),
            "model.layers.0.self_attn.q_b_proj|target_verify": torch.full(
                (3, 4), 2.0
            ),
        }
        pooled = pool_module_captures(captures)
        self.assertEqual(list(pooled), ["model.layers.0.self_attn.q_b_proj"])
        item = pooled["model.layers.0.self_attn.q_b_proj"]
        self.assertEqual(item["matrix"].shape, (5, 4))
        self.assertEqual(
            {phase: value.shape[0] for phase, value in item["phases"].items()},
            {"decode": 2, "target_verify": 3},
        )


class FootprintTests(unittest.TestCase):
    def test_rank_256_fp8_estimate(self):
        result = estimate(
            hidden=6144,
            intermediate=2048,
            experts=256,
            routed_layers=75,
            gate_rank=256,
            down_rank=256,
            image_bytes=1,
            tp_size=2,
            hot_full_experts=0,
            checkpoint_full_expert_bytes=20_000_000,
        )
        self.assertEqual(result["per_expert_image_bytes"], 2_621_440)
        self.assertEqual(
            result["all_images_model_bytes"], 2_621_440 * 256 * 75
        )


class ExpertReferenceTests(unittest.TestCase):
    def test_two_stage_all_fast_matches_dense_experts(self):
        torch.manual_seed(7)
        experts, hidden, intermediate, rank = 3, 6, 4, 6
        gate_up_weights = torch.randn(experts, intermediate * 2, hidden)
        down_weights = torch.randn(experts, hidden, intermediate)
        gate_basis = torch.eye(hidden)
        down_basis = torch.eye(intermediate)
        gate_images = gate_up_weights @ gate_basis
        down_images = down_weights @ down_basis
        x = torch.randn(2, hidden)
        ids = torch.tensor([[0, 2], [1, 0]])
        weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])

        def full(expert, value):
            gate_part, up_part = (value @ gate_up_weights[expert].T).chunk(2, -1)
            return torch.nn.functional.silu(gate_part).mul(up_part) @ down_weights[
                expert
            ].T

        def down(expert, value):
            return value @ down_weights[expert].T

        output, stats = functional_expert_forward(
            x,
            ids,
            weights,
            gate_up_basis=gate_basis,
            gate_up_images=gate_images,
            down_bases=down_basis,
            down_images=down_images,
            gate_up_epsilon=0.0,
            down_epsilon=0.0,
            full_fallback=full,
            down_fallback=down,
            gate_override="always_fast",
        )
        expected = torch.stack(
            [
                sum(
                    weights[row, slot]
                    * full(int(ids[row, slot]), x[row : row + 1])[0]
                    for slot in range(ids.shape[1])
                )
                for row in range(x.shape[0])
            ]
        )
        torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)
        self.assertTrue(bool(stats.gate_up_fast.all()))
        self.assertTrue(bool(stats.down_fast.all()))

    def test_two_stage_all_slow_invokes_full_fallback_only(self):
        x = torch.randn(1, 4)
        ids = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.5, 0.5]])
        calls = {"full": 0, "down": 0}

        def full(expert, value):
            calls["full"] += 1
            return torch.full_like(value, float(expert + 1))

        def down(expert, value):
            calls["down"] += 1
            return torch.zeros((value.shape[0], 4))

        output, _ = functional_expert_forward(
            x,
            ids,
            weights,
            gate_up_basis=torch.eye(4)[:, :1],
            gate_up_images=torch.zeros(2, 4, 1),
            down_bases=torch.eye(2),
            down_images=torch.zeros(2, 4, 2),
            gate_up_epsilon=0.0,
            down_epsilon=0.0,
            full_fallback=full,
            down_fallback=down,
            gate_override="all_slow",
        )
        torch.testing.assert_close(output, torch.full_like(x, 1.5))
        self.assertEqual(calls, {"full": 2, "down": 0})


class ConfigTests(unittest.TestCase):
    def test_off_defaults(self):
        names = [
            "GSI_MODE",
            "GSI_PROFILE",
            "GSI_CAPTURE_DIR",
            "GSI_IMAGE_DTYPE",
        ]
        saved = {name: os.environ.pop(name, None) for name in names}
        try:
            config = GSIConfig.from_env()
            self.assertEqual(config.mode, GSIMode.OFF)
        finally:
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value


class RuntimeIntegrationTests(unittest.TestCase):
    class _DecodeMode:
        def is_decode(self):
            return True

        def is_target_verify(self):
            return False

        def is_draft_extend_v2(self):
            return False

        def is_draft_extend(self):
            return False

        def is_idle(self):
            return False

        def is_extend(self):
            return False

    class _QuantMethod:
        def __init__(self):
            self.rows = []

        def apply(self, layer, value, bias=None):
            self.rows.append(value.shape[0])
            output = value @ layer.weight.T
            return output if bias is None else output + bias

    class _Linear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.quant_method = RuntimeIntegrationTests._QuantMethod()

        def forward(self, value):
            return self.quant_method.apply(self, value)

    class _Layer(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.q_b_proj = RuntimeIntegrationTests._Linear(weight)

    class _Model(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [RuntimeIntegrationTests._Layer(weight)]
            )

    def test_proxy_runs_mixed_fast_and_exact_slow_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module_name = "model.layers.0.self_attn.q_b_proj"
            weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            basis = torch.eye(4)[:, :2].contiguous()
            image = weight @ basis
            save_file({"basis": basis}, str(root / "basis.safetensors"))
            save_file({"image": image}, str(root / "image.tp0.safetensors"))
            profile = GSIProfile.from_entries(
                model_id="mock",
                model_hash="mock-hash",
                quantization="mock",
                tp_size=1,
                entries=[
                    GSIEntry(
                        module=module_name,
                        family="q_latent",
                        layer=0,
                        rank=2,
                        epsilon=0.1,
                        input_dim=4,
                        output_dim=3,
                        basis_file="basis.safetensors",
                        image_file="image.tp{tp_rank}.safetensors",
                    )
                ],
                calibration={"split": "unit"},
            )
            profile_path = root / "profile.json"
            profile.save(profile_path)
            runtime = GSIRuntime(
                GSIConfig(
                    mode=GSIMode.FUNCTIONAL,
                    profile=profile_path,
                    active_phases=frozenset({"decode"}),
                )
            )
            model = self._Model(weight)
            base_method = model.model.layers[0].self_attn.q_b_proj.quant_method
            self.assertEqual(runtime.install(model), 1)
            value = torch.tensor(
                [[1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 9.0, 0.0]]
            )
            with forward_phase(self._DecodeMode()):
                output = model.model.layers[0].self_attn.q_b_proj(value)
            torch.testing.assert_close(output, value @ weight.T)
            self.assertEqual(base_method.rows, [1])
            snapshot = runtime.telemetry.snapshot()[module_name]
            self.assertEqual(snapshot["rows"], 2)
            self.assertEqual(snapshot["fast"], 1)

    def test_proxy_bypasses_prefill(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module_name = "model.layers.0.self_attn.q_b_proj"
            weight = torch.eye(4)
            basis = torch.eye(4)
            image = weight @ basis
            save_file({"basis": basis}, str(root / "basis.safetensors"))
            save_file({"image": image}, str(root / "image.tp0.safetensors"))
            profile = GSIProfile.from_entries(
                model_id="mock",
                model_hash="mock-hash",
                quantization="mock",
                tp_size=1,
                entries=[
                    GSIEntry(
                        module=module_name,
                        family="q_latent",
                        layer=0,
                        rank=4,
                        epsilon=0.1,
                        input_dim=4,
                        output_dim=4,
                        basis_file="basis.safetensors",
                        image_file="image.tp{tp_rank}.safetensors",
                    )
                ],
                calibration={"split": "unit"},
            )
            profile_path = root / "profile.json"
            profile.save(profile_path)
            runtime = GSIRuntime(
                GSIConfig(mode=GSIMode.FUNCTIONAL, profile=profile_path)
            )
            model = self._Model(weight)
            base_method = model.model.layers[0].self_attn.q_b_proj.quant_method
            runtime.install(model)
            value = torch.randn(3, 4)
            output = model.model.layers[0].self_attn.q_b_proj(value)
            torch.testing.assert_close(output, value)
            self.assertEqual(base_method.rows, [3])
            self.assertEqual(runtime.telemetry.snapshot(), {})


class ExpertLayerSelectionTests(unittest.TestCase):
    def test_blank_and_all_mean_every_layer(self):
        self.assertIsNone(_parse_layers(""))
        self.assertIsNone(_parse_layers("all"))
        self.assertEqual(_parse_layers("none"), frozenset())

    def test_indices_and_ranges(self):
        self.assertEqual(sorted(_parse_layers("3,5-8")), [3, 5, 6, 7, 8])
        self.assertEqual(sorted(_parse_layers(" 1 , 1 , 2 ")), [1, 2])

    def test_rejects_inverted_ranges_and_empty_sets(self):
        with self.assertRaises(ValueError):
            _parse_layers("8-3")
        with self.assertRaises(ValueError):
            _parse_layers(",,")


class SubspaceTests(unittest.TestCase):
    def test_overlap_is_one_for_identical_bases_and_zero_for_orthogonal(self):
        torch.manual_seed(3)
        q, _ = torch.linalg.qr(torch.randn(32, 16))
        self.assertAlmostEqual(subspace_overlap(q[:, :8], q[:, :8]), 1.0, places=5)
        self.assertAlmostEqual(subspace_overlap(q[:, :8], q[:, 8:16]), 0.0, places=5)

    def test_cascade_recovers_the_dominant_subspace_without_a_seed(self):
        torch.manual_seed(4)
        basis, _ = torch.linalg.qr(torch.randn(48, 6))
        matrix = torch.randn(200, 6) @ basis.T
        result = cascade_basis(matrix, 6, iterations=4)
        self.assertFalse(result.seeded)
        self.assertLess(result.mean_rho, 1e-3)
        self.assertGreater(subspace_overlap(basis, result.basis), 0.999)

    def test_a_good_seed_converges_and_reports_its_overlap(self):
        torch.manual_seed(5)
        basis, _ = torch.linalg.qr(torch.randn(48, 6))
        matrix = torch.randn(200, 6) @ basis.T
        seeded = cascade_basis(matrix, 6, seed=basis, iterations=2)
        self.assertTrue(seeded.seeded)
        self.assertIsNotNone(seeded.seed_overlap)
        self.assertGreater(seeded.seed_overlap, 0.99)
        self.assertLess(seeded.mean_rho, 1e-3)

    def test_a_misaligned_seed_still_converges(self):
        """Orthogonal iteration must not inherit a bad seed's error."""
        torch.manual_seed(6)
        basis, _ = torch.linalg.qr(torch.randn(48, 6))
        matrix = torch.randn(200, 6) @ basis.T
        wrong, _ = torch.linalg.qr(torch.randn(48, 6))
        result = cascade_basis(matrix, 6, seed=wrong, iterations=6)
        self.assertLess(result.mean_rho, 1e-3)
        self.assertGreater(subspace_overlap(basis, result.basis), 0.99)


class BasisScopeTests(unittest.TestCase):
    def test_layer_scope_groups_equal_width_maps_and_splits_others(self):
        attn = "model.layers.7.self_attn.fused_qkv_a_proj_with_mqa"
        mlp = "model.layers.7.mlp.gate_up_proj"
        routed = "model.layers.7.mlp.routed_gate_up"
        out = "model.layers.7.self_attn.o_proj"
        shared = {basis_group_key(name, 6144, "layer") for name in (attn, mlp, routed)}
        self.assertEqual(len(shared), 1)
        self.assertNotIn(basis_group_key(out, 8192, "layer"), shared)
        self.assertNotEqual(
            basis_group_key(attn, 6144, "layer"),
            basis_group_key("model.layers.8.mlp.gate_up_proj", 6144, "layer"),
        )

    def test_module_scope_keeps_every_map_separate(self):
        attn = "model.layers.7.self_attn.fused_qkv_a_proj_with_mqa"
        mlp = "model.layers.7.mlp.gate_up_proj"
        self.assertNotEqual(
            basis_group_key(attn, 6144, "module"),
            basis_group_key(mlp, 6144, "module"),
        )

    def test_default_epsilon_is_the_paper_operating_point(self):
        self.assertEqual(DEFAULT_EPSILON, 0.10)

    def test_routed_families_are_named(self):
        self.assertEqual(
            routed_family("model.layers.7.mlp.routed_gate_up"), "routed_gate_up"
        )
        self.assertEqual(
            routed_family("model.layers.7.mlp.routed_down"), "routed_down"
        )
        self.assertIsNone(routed_family("model.layers.7.mlp.gate_up_proj"))


class ExpertImageTests(unittest.TestCase):
    def test_chunk_plan_respects_the_row_budget(self):
        chunks = plan_expert_chunks(list(range(10)), rank=4, budget_rows=12)
        self.assertEqual(chunks, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])
        self.assertEqual(
            plan_expert_chunks([0, 1], rank=8, budget_rows=8), [[0], [1]]
        )
        with self.assertRaises(ValueError):
            plan_expert_chunks([0], rank=8, budget_rows=4)

    def test_images_reproduce_each_expert_weight_applied_to_the_basis(self):
        torch.manual_seed(7)
        width, rank, out_dim, experts = 16, 4, 6, 3
        weights = [torch.randn(out_dim, width) for _ in range(experts)]
        basis, _ = torch.linalg.qr(torch.randn(width, rank))

        def projection(rows, expert_ids):
            stacked = torch.stack([weights[int(e)] for e in expert_ids])
            return torch.bmm(stacked, rows.unsqueeze(-1)).squeeze(-1)

        result = build_stage_images(
            basis, range(experts), projection, budget_rows=2 * rank
        )
        self.assertEqual(sorted(result.images), [0, 1, 2])
        for expert in range(experts):
            expected = weights[expert] @ basis
            torch.testing.assert_close(
                result.images[expert].float(), expected, atol=2e-2, rtol=2e-2
            )

    def test_a_fast_row_reconstructs_through_the_image(self):
        """image @ (V'x) must equal Wx when x lies in the subspace."""
        torch.manual_seed(8)
        width, rank, out_dim = 20, 5, 7
        weight = torch.randn(out_dim, width)
        basis, _ = torch.linalg.qr(torch.randn(width, rank))

        def projection(rows, expert_ids):
            return rows @ weight.T

        image = build_stage_images(basis, [0], projection).images[0].float()
        inside = (torch.randn(3, rank) @ basis.T)
        # Images are stored bf16, so agreement is to bf16 precision, not exact.
        torch.testing.assert_close(
            (inside @ basis) @ image.T, inside @ weight.T, atol=2e-2, rtol=2e-2
        )


class EvaluateTests(unittest.TestCase):
    def test_equation_six_matches_the_papers_gptj_row(self):
        # Table 6: GPT-J k=256, eps=0.10, 99.8% fast path -> 15.6x.
        self.assertAlmostEqual(
            effective_speedup(0.998, 4096, 256), 15.6, delta=0.1
        )
        # eps=0.05, 77.2% fast path -> 3.6x.
        self.assertAlmostEqual(
            effective_speedup(0.772, 4096, 256), 3.6, delta=0.1
        )

    def test_speedup_is_one_when_nothing_takes_the_fast_path(self):
        self.assertAlmostEqual(effective_speedup(0.0, 6144, 512), 1.0)

    def test_deployed_speedup_penalises_the_quantized_baseline(self):
        """Against W4AFP8 the same fast fraction is worth far less."""
        paper = effective_speedup(0.998, 6144, 512)
        deployed = deployed_speedup(0.998, 6144, 512, out_dim=6144)
        self.assertGreater(paper, 11.0)
        self.assertLess(deployed, paper / 4)

    def test_break_even_matches_the_deployed_model(self):
        width, rank, out_dim = 6144, 512, 6144
        fraction = break_even_fast_fraction(width, rank, out_dim)
        self.assertAlmostEqual(
            deployed_speedup(fraction, width, rank, out_dim), 1.0, places=5
        )

    def test_break_even_is_unreachable_when_the_image_row_is_not_cheaper(self):
        # rank >= d/4 in bf16 against packed INT4: the image row costs more
        # than the weight row it replaces.
        self.assertEqual(break_even_fast_fraction(1024, 512, 6144), float("inf"))

    def test_speedup_report_aggregates_on_cost_not_on_ratios(self):
        entries = {
            "big": GSIEntry(
                module="big",
                family="attn_output",
                layer=0,
                rank=512,
                epsilon=0.1,
                input_dim=8192,
                output_dim=6144,
                basis_file="b",
            ),
            "small": GSIEntry(
                module="small",
                family="q_latent",
                layer=0,
                rank=512,
                epsilon=0.1,
                input_dim=2048,
                output_dim=64,
                basis_file="b",
            ),
        }
        telemetry = {
            "big": {"rows": 1000, "fast": 0},
            "small": {"rows": 1000, "fast": 1000},
            "unprofiled": {"rows": 1000, "fast": 1000},
        }
        report = speedup_report(telemetry, entries)
        self.assertEqual(report.rows, 2000)
        self.assertNotIn("unprofiled", report.per_module)
        # The tiny fully-fast module must not drag the headline above the big
        # fully-slow one by more than its share of the work.
        self.assertLess(report.paper_speedup, 1.05)

    def test_perplexity_and_ratio(self):
        self.assertAlmostEqual(perplexity([0.0, 0.0]), 1.0)
        self.assertAlmostEqual(perplexity([-1.0, -1.0]), math.e)
        self.assertAlmostEqual(
            perplexity_ratio([-1.0, -1.0], [-1.0, -1.0]), 1.0
        )
        with self.assertRaises(ValueError):
            perplexity([])

    def test_top1_agreement_counts_length_differences_against_the_run(self):
        self.assertAlmostEqual(top1_agreement("abcd", "abcd"), 1.0)
        self.assertAlmostEqual(top1_agreement("abcd", "abxd"), 0.75)
        self.assertAlmostEqual(top1_agreement("abcd", "ab"), 0.5)

    def test_generation_agreement_is_all_or_nothing_per_prompt(self):
        self.assertAlmostEqual(
            generation_agreement(["one", "two"], ["one", "two"]), 1.0
        )
        self.assertAlmostEqual(
            generation_agreement(["one", "two"], ["one", "three"]), 0.5
        )
        with self.assertRaises(ValueError):
            generation_agreement(["one"], ["one", "two"])


if __name__ == "__main__":
    unittest.main()
