from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import threading
import types
from dataclasses import replace
from pathlib import Path

from .capture import ActivationRecorder
from .config import GSIConfig, GSIMode
from .distributed import tensor_parallel_rank
from .families import classify_linear
from .math import apply_gated_linear, project_and_gate
from .profile import GSIEntry, GSIProfile
from .telemetry import GSITelemetry

logger = logging.getLogger(__name__)

_PHASE = contextvars.ContextVar("gsi_forward_phase", default="unknown")
_ROUTED_LAYER = contextvars.ContextVar("gsi_routed_layer", default=-1)
_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()


@contextlib.contextmanager
def routed_layer(layer: int):
    """Publish the current MoE layer id to the fused expert backend.

    The grouped-GEMM entry point owns the post-SiLU intermediate but is called
    without any layer identity, so the layer is carried in a context variable
    the same way the forward phase is.
    """
    token = _ROUTED_LAYER.set(int(layer))
    try:
        yield
    finally:
        _ROUTED_LAYER.reset(token)


def phase_name(forward_mode) -> str:
    if forward_mode is None:
        return "unknown"
    for method, name in (
        ("is_decode", "decode"),
        ("is_target_verify", "target_verify"),
        ("is_draft_extend_v2", "draft_extend_v2"),
        ("is_draft_extend", "draft_extend"),
        ("is_idle", "idle"),
        ("is_extend", "prefill"),
    ):
        predicate = getattr(forward_mode, method, None)
        if predicate is not None and predicate():
            return name
    return str(forward_mode).lower()


@contextlib.contextmanager
def forward_phase(forward_mode):
    token = _PHASE.set(phase_name(forward_mode))
    try:
        yield
    finally:
        _PHASE.reset(token)


class _TensorArtifacts:
    def __init__(self, profile: GSIProfile):
        self.profile = profile
        self._cpu: dict[tuple[str, str], object] = {}
        self._device: dict[tuple[str, str, str], object] = {}
        self._lock = threading.Lock()

    def _load_cpu(self, entry: GSIEntry, kind: str):
        from safetensors import safe_open

        key = (entry.module, kind)
        with self._lock:
            if key in self._cpu:
                return self._cpu[key]
            artifact = entry.basis_file if kind == "basis" else entry.image_file
            if artifact is None:
                raise FileNotFoundError(f"{kind} missing for {entry.module}")
            path = self.profile.resolve(artifact)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                tensor_key = kind if kind in handle.keys() else handle.keys()[0]
                tensor = handle.get_tensor(tensor_key).contiguous()
            self._cpu[key] = tensor
            return tensor

    def get(self, entry: GSIEntry, kind: str, device, dtype):
        device_key = (entry.module, kind, f"{device}:{dtype}")
        with self._lock:
            cached = self._device.get(device_key)
        if cached is not None:
            return cached
        tensor = self._load_cpu(entry, kind).to(device=device, dtype=dtype)
        with self._lock:
            self._device[device_key] = tensor
        return tensor


class _QuantMethodProxy:
    """Delegates weight lifecycle to the original method and gates apply()."""

    def __init__(self, runtime: "GSIRuntime", module_name: str, entry, base):
        self._gsi_runtime = runtime
        self._gsi_module_name = module_name
        self._gsi_entry = entry
        self._gsi_base = base

    def __getattr__(self, name):
        return getattr(self._gsi_base, name)

    def apply(self, layer, x, bias=None):
        runtime = self._gsi_runtime
        phase = _PHASE.get()
        active = phase in runtime.config.active_phases
        if not active or not hasattr(x, "shape") or isinstance(x, tuple):
            return self._gsi_base.apply(layer, x, bias)

        runtime.capture(self._gsi_module_name, x)
        entry = self._gsi_entry
        if entry is None:
            return self._gsi_base.apply(layer, x, bias)
        try:
            basis = runtime.artifacts.get(entry, "basis", x.device, x.dtype)
            gate = project_and_gate(
                x,
                basis,
                entry.epsilon,
                gate_override=runtime.config.gate_override,
            )
            runtime.telemetry.observe(self._gsi_module_name, gate)
            if runtime.config.mode == GSIMode.OBSERVE:
                return self._gsi_base.apply(layer, x, bias)
            image = runtime.artifacts.get(entry, "image", x.device, x.dtype)
            result, _ = apply_gated_linear(
                x,
                basis,
                image,
                entry.epsilon,
                lambda slow_x: self._gsi_base.apply(layer, slow_x, bias),
                bias=bias,
                gate_override=runtime.config.gate_override,
            )
            return result
        except Exception:
            if runtime.config.strict_fallback:
                logger.exception(
                    "GSI failed for %s; using exact baseline", self._gsi_module_name
                )
                return self._gsi_base.apply(layer, x, bias)
            raise


class GSIRuntime:
    def __init__(self, config: GSIConfig):
        self.config = config
        self.profile = (
            GSIProfile.load(config.profile, verify_files=not config.build_images)
            if config.profile is not None
            else None
        )
        self.entries = self.profile.by_module() if self.profile else {}
        self.artifacts = _TensorArtifacts(self.profile) if self.profile else None
        telemetry_path = config.telemetry_path
        if telemetry_path is not None:
            telemetry_path = Path(
                str(telemetry_path).format(tp_rank=tensor_parallel_rank())
            )
        self.telemetry = GSITelemetry(telemetry_path)
        self.recorder = None
        if config.capture_dir is not None:
            rank = tensor_parallel_rank()
            self.recorder = ActivationRecorder(
                config.capture_dir, config.capture_rows, rank=rank
            )
        self._installed_models: set[int] = set()
        self._image_validation: dict[str, dict] = {}

    def capture(self, module_name: str, value) -> None:
        phase = _PHASE.get()
        if self.recorder is None or phase not in self.config.active_phases:
            return
        if phase.startswith("draft") and not module_name.startswith("nextn."):
            module_name = f"nextn.{module_name}"
        self.recorder.record(f"{module_name}|{phase}", value)

    def observe_routed_input(self, layer: int, hidden_states, topk_ids=None) -> None:
        key = f"model.layers.{layer}.mlp.routed_gate_up"
        self.capture(key, hidden_states)
        if topk_ids is not None and self.recorder is not None:
            self.recorder.record(f"{key}.expert_ids|{_PHASE.get()}", topk_ids)

    def observe_routed_intermediate(self, layer: int, intermediate) -> None:
        """Capture the post-SiLU routed activation feeding each expert's down.

        The paper fits one basis per layer, so every expert at this layer shares
        a single basis over this space; rows from whichever experts fired are
        pooled rather than separated per expert.
        """
        self.capture(f"model.layers.{layer}.mlp.routed_down", intermediate)

    def install(self, model) -> int:
        if id(model) in self._installed_models:
            return 0
        self._install_nextn_phase(model)
        if self.config.build_images:
            self._build_images(model)
        installed = 0
        is_nextn = model.__class__.__name__.endswith("NextN")
        for name, module in model.named_modules():
            family = classify_linear(name)
            quant_method = getattr(module, "quant_method", None)
            if family is None or quant_method is None:
                continue
            profile_name = f"nextn.{name}" if is_nextn else name
            entry = self.entries.get(profile_name)
            if self.config.mode in {GSIMode.FUNCTIONAL, GSIMode.KERNEL} and entry is None:
                continue
            if isinstance(quant_method, _QuantMethodProxy):
                continue
            module.quant_method = _QuantMethodProxy(
                self, profile_name, entry, quant_method
            )
            installed += 1
        self._installed_models.add(id(model))
        logger.warning(
            "GSI %s installed on %d standard linear modules",
            self.config.mode.value,
            installed,
        )
        return installed

    @staticmethod
    def _install_nextn_phase(model) -> None:
        """Wrap NEXTN's override, which does not inherit the target forward."""
        if (
            not model.__class__.__name__.endswith("NextN")
            or getattr(model, "_gsi_phase_wrapped", False)
        ):
            return
        original = model.forward

        def wrapped(instance, *args, **kwargs):
            forward_batch = kwargs.get("forward_batch")
            if forward_batch is None and len(args) >= 3:
                forward_batch = args[2]
            with forward_phase(
                getattr(forward_batch, "forward_mode", None)
            ):
                return original(*args, **kwargs)

        model.forward = types.MethodType(wrapped, model)
        model._gsi_phase_wrapped = True

    def _build_images(self, model) -> None:
        if self.profile is None or self.artifacts is None:
            raise ValueError("a profile is required to build images")
        if self.config.cache_dir is None:
            raise ValueError("GSI_CACHE_DIR is required with GSI_BUILD_IMAGES=1")
        if self.config.image_dtype != "bf16":
            raise RuntimeError(
                "FP8 image construction is disabled until scaled FP8 image "
                "metadata is implemented; build the BF16 correctness cache first"
            )
        import torch
        from safetensors.torch import save_file

        rank = tensor_parallel_rank()
        modules = dict(model.named_modules())
        is_nextn = model.__class__.__name__.endswith("NextN")
        updated = []
        for entry in self.profile.entries:
            entry_is_nextn = entry.module.startswith("nextn.")
            if entry_is_nextn != is_nextn:
                updated.append(entry)
                continue
            local_name = (
                entry.module.removeprefix("nextn.")
                if entry_is_nextn
                else entry.module
            )
            if entry.family in {"routed_gate_up", "routed_down"}:
                updated.extend(
                    self._build_routed_images(entry, modules, local_name, rank)
                )
                continue
            module = modules.get(local_name)
            method = getattr(module, "quant_method", None) if module else None
            if method is None:
                # Entries belonging to the other loaded model (target vs NEXTN)
                # are preserved and built when that model installs.
                updated.append(entry)
                continue
            parameter = next(module.parameters(), None)
            if parameter is None:
                updated.append(entry)
                continue
            # Image construction is a streaming operation.  Do not populate
            # the runtime device cache or hundreds of layer bases would remain
            # resident and consume the headroom needed by the loaded model.
            basis = self.artifacts._load_cpu(entry, "basis").to(
                device=parameter.device, dtype=torch.bfloat16
            )
            with torch.no_grad():
                basis_batch = basis.transpose(0, 1).contiguous()
                output = method.apply(module, basis_batch, None)
                if isinstance(output, tuple):
                    output = output[0]
                image_device = output.transpose(0, 1).to(
                    dtype=torch.bfloat16
                ).contiguous()
                self._validate_image_on_replay(
                    entry, module, method, basis, image_device
                )
                image = image_device.to(device="cpu").contiguous()
            relative = (
                Path("images")
                / f"{entry.module.replace('.', '__')}.tp{{tp_rank}}.safetensors"
            )
            path = self.config.cache_dir / str(relative).format(tp_rank=rank)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {"image": image},
                str(path),
                metadata={
                    "module": entry.module,
                    "tp_rank": str(rank),
                    "format": "gsi-image-v1",
                },
            )
            updated.append(
                replace(
                    entry,
                    output_dim=image.shape[0],
                    image_file=str(relative),
                    image_sha256="",
                    image_dtype="bf16",
                )
            )
        self.profile.entries = updated
        self.entries = self.profile.by_module()
        validation_path = (
            self.config.cache_dir / f"image-validation.tp{rank}.json"
        )
        validation_path.write_text(
            json.dumps(self._image_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.warning("wrote GSI replay validation to %s", validation_path)
        if int(rank) == 0:
            built_profile = self.config.cache_dir / "profile.built.json"
            self.profile.save(built_profile)
            logger.warning("wrote GSI image profile to %s", built_profile)

    def _build_routed_images(self, entry, modules, local_name: str, rank: int):
        """Build one image per routed expert for a gate/up or down entry.

        Returns the profile entries that replace ``entry``: one per expert that
        was built.  A layer excluded by `GSI_EXPERT_LAYERS` keeps its basis and
        drops its image reference, which leaves the site permanently on the slow
        path rather than silently serving a stale image.
        """
        import torch
        from safetensors.torch import save_file

        from .expert_images import (
            build_stage_images,
            cutlass_down_projection,
            cutlass_gate_up_projection,
        )

        layers = self.config.expert_layers
        experts_name = local_name.rsplit(".", 1)[0] + ".experts"
        experts = modules.get(experts_name)
        method = getattr(experts, "quant_method", None) if experts else None
        if method is None or (layers is not None and entry.layer not in layers):
            return [replace(entry, image_file=None)]
        # kt_ep wraps the CUTLASS method; the wrapped method owns the strides.
        method = getattr(method, "wrapped_method", method)
        weight = getattr(experts, "w13_weight", None)
        if weight is None:
            logger.warning(
                "GSI: %s has no w13_weight; leaving %s on the slow path",
                experts_name,
                entry.module,
            )
            return [replace(entry, image_file=None)]

        basis = self.artifacts._load_cpu(entry, "basis").to(
            device=weight.device, dtype=torch.bfloat16
        )
        projection = (
            cutlass_gate_up_projection
            if entry.family == "routed_gate_up"
            else cutlass_down_projection
        )
        with torch.no_grad():
            result = build_stage_images(
                basis,
                range(weight.shape[0]),
                lambda rows, ids: projection(experts, method, rows, ids),
                budget_rows=self.config.expert_row_budget,
            )

        produced = []
        for expert, image in sorted(result.images.items()):
            relative = (
                Path("images")
                / f"{entry.module.replace('.', '__')}.e{expert}"
                ".tp{tp_rank}.safetensors"
            )
            path = self.config.cache_dir / str(relative).format(tp_rank=rank)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {"image": image.to(device="cpu").contiguous()},
                str(path),
                metadata={
                    "module": entry.module,
                    "expert": str(expert),
                    "tp_rank": str(rank),
                    "format": "gsi-image-v1",
                },
            )
            produced.append(
                replace(
                    entry,
                    expert=expert,
                    output_dim=int(image.shape[0]),
                    image_file=str(relative),
                    image_sha256="",
                    image_dtype="bf16",
                )
            )
        logger.warning(
            "GSI built %d routed images for %s", len(produced), entry.module
        )
        return produced

    def _validate_image_on_replay(
        self, entry: GSIEntry, module, method, basis, image
    ) -> None:
        """Validate WV on exact calibration rows embedded with each basis."""
        import torch
        from safetensors import safe_open

        path = self.profile.resolve(entry.basis_file)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if "replay" not in handle.keys():
                return
            replay = handle.get_tensor("replay").to(
                device=basis.device, dtype=torch.bfloat16
            )
        if replay.numel() == 0:
            return

        def apply(value):
            result = method.apply(module, value.contiguous(), None)
            return result[0] if isinstance(result, tuple) else result

        def relative_l2(reference, candidate) -> float:
            reference = reference.float()
            candidate = candidate.float()
            denominator = reference.square().sum().sqrt().clamp_min(1e-12)
            return float((reference - candidate).square().sum().sqrt() / denominator)

        with torch.no_grad():
            coefficients = replay @ basis
            projected = coefficients @ basis.transpose(0, 1)
            fast = coefficients @ image.transpose(0, 1)
            exact_raw = apply(replay)
            exact_projected = apply(projected)
            energy = replay.float().square().sum(dim=-1).clamp_min(1e-12)
            residual = (
                (replay.float() - projected.float()).square().sum(dim=-1)
                / energy
            ).clamp_min(0).sqrt()
            self._image_validation[entry.module] = {
                "rows": replay.shape[0],
                "rank": entry.rank,
                "input_mean_rho": float(residual.mean()),
                "input_max_rho": float(residual.max()),
                "raw_exact_vs_image_relative_l2": relative_l2(exact_raw, fast),
                "projected_exact_vs_image_relative_l2": relative_l2(
                    exact_projected, fast
                ),
                "raw_exact_vs_projected_exact_relative_l2": relative_l2(
                    exact_raw, exact_projected
                ),
                "raw_exact_max_abs_error": float(
                    (exact_raw.float() - fast.float()).abs().max()
                ),
                "projected_exact_max_abs_error": float(
                    (exact_projected.float() - fast.float()).abs().max()
                ),
            }


def get_runtime() -> GSIRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = GSIRuntime(GSIConfig.from_env())
    return _RUNTIME


def install_model_runtime(model) -> int:
    runtime = get_runtime()
    if runtime.config.mode == GSIMode.OFF:
        return 0
    return runtime.install(model)


def observe_routed_input(layer: int, hidden_states, topk_ids=None) -> None:
    runtime = get_runtime()
    if runtime.config.mode != GSIMode.OFF:
        runtime.observe_routed_input(layer, hidden_states, topk_ids)


def observe_routed_intermediate(layer: int, intermediate) -> None:
    runtime = get_runtime()
    if runtime.config.mode != GSIMode.OFF:
        runtime.observe_routed_intermediate(layer, intermediate)


def observe_fused_intermediate(intermediate, scale=None, valid_rows=None) -> None:
    """Capture the fused backend's post-SiLU intermediate for the active layer.

    Called from inside the grouped-GEMM path, which sees an FP8 tensor and its
    per-tensor scale; the calibrated basis must live in the same units as the
    down GEMM's input, so the rows are rescaled back to bf16 here.

    ``valid_rows`` is the grouped-GEMM row count actually written by the SiLU
    kernel.  The buffer is allocated for the worst case, so rows beyond it hold
    uninitialized memory and are dropped rather than calibrated on.
    """
    import torch

    layer = _ROUTED_LAYER.get()
    if layer < 0:
        return
    runtime = get_runtime()
    if runtime.config.mode == GSIMode.OFF or runtime.recorder is None:
        return
    if valid_rows is not None:
        count = int(valid_rows.item() if hasattr(valid_rows, "item") else valid_rows)
        if count <= 0:
            return
        intermediate = intermediate[:count]
    rows = intermediate.to(dtype=torch.bfloat16)
    if scale is not None:
        rows = rows * scale.to(device=rows.device, dtype=rows.dtype).reshape(-1)[0]
    runtime.observe_routed_intermediate(layer, rows)
