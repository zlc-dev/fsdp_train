"""Capture and analyse BF16 tensors produced by the training job.

The module deliberately has no dependency on Transformers or FSDP.  This makes
the bit-statistics useful for a saved checkpoint as well as for a live model,
and makes it possible to unit test the BF16 bit handling on a CPU-only host.

BF16 has the following IEEE-754 layout (bit 15 is the sign bit)::

    [ sign (1) ][ exponent (8) ][ fraction (7) ]
                  ^ bit 14 ... bit 7

``exponent_window_statistics`` examines every interval of consecutive numeric
exponent values. For example, a width-3 window can be ``[120, 121, 122]``.
The returned ``top`` entry is the interval containing the most tensor
elements; per-position results are returned too, so the choice is
unambiguous and reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


LOGGER = logging.getLogger(__name__)
DEFAULT_TARGET_LAYERS = (0, 16, 32)


def _bf16_cpu(value: torch.Tensor) -> torch.Tensor:
    """Detach a tensor and return a CPU BF16 copy.

    Captures are intentionally copied before the optimizer can mutate the
    parameter or before a subsequent forward pass overwrites an activation.
    Casting here also covers a few model implementations whose layer output is
    promoted to FP32 by a normalization operation.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(value)!r}")
    # FSDP2 may expose a distributed tensor (DTensor) for a sharded parameter.
    # Statistics are rank-local by design; use the local shard when available.
    if hasattr(value, "to_local"):
        value = value.to_local()
    return value.detach().to(dtype=torch.bfloat16, device="cpu").clone()


def _first_tensor(output: Any) -> torch.Tensor | None:
    """Find the tensor carrying a decoder layer's hidden states."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, Mapping):
        # ModelOutput behaves like a mapping but ``last_hidden_state`` is the
        # preferred field when present.
        preferred = output.get("last_hidden_state")
        if isinstance(preferred, torch.Tensor):
            return preferred
        for item in output.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def resolve_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList | Sequence[torch.nn.Module]:
    """Return the decoder layer list for Llama and common HF model wrappers."""

    candidates = (
        ("model", "layers"),       # LlamaForCausalLM
        ("transformer", "h"),      # GPT-style wrappers
        ("layers",),
    )
    for path in candidates:
        current: Any = model
        try:
            for component in path:
                current = getattr(current, component)
        except AttributeError:
            continue
        if isinstance(current, (torch.nn.ModuleList, list, tuple)):
            return current
    raise ValueError(
        "Unable to find decoder layers; expected model.model.layers or "
        "model.transformer.h"
    )


class LayerTensorCapture:
    """Forward-hook based capture of selected Transformer layers.

    The hooks only retain the latest activation for each selected layer.  Call
    :meth:`snapshot` after backward to copy weights, activations and gradients
    into a serializable dictionary, then call :meth:`save_snapshot`.  Retaining
    one batch at a time avoids an accidental unbounded memory leak in long
    training runs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: Iterable[int] = DEFAULT_TARGET_LAYERS,
    ) -> None:
        self.model = model
        self.target_layers = tuple(dict.fromkeys(int(i) for i in target_layers))
        if any(i < 0 for i in self.target_layers):
            raise ValueError("layer indices must be non-negative")
        self.layers = resolve_decoder_layers(model)
        missing = [i for i in self.target_layers if i >= len(self.layers)]
        if missing:
            raise IndexError(
                f"requested layer(s) {missing}, but model has {len(self.layers)} layers"
            )
        self.activations: dict[tuple[int, str], torch.Tensor] = {}
        self._activation_refs: dict[tuple[int, str], torch.Tensor] = {}
        self._handles: list[Any] = []
        for index in self.target_layers:
            self._handles.append(
                self.layers[index].register_forward_hook(self._make_hook(index))
            )

    def _make_hook(self, index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            # Capture both sides of the decoder block.  For a normal Llama
            # block these are the hidden-state tensors; masks and position
            # identifiers are ignored by _first_tensor.
            for kind, value in (("input", _first_tensor(_inputs)), ("output", _first_tensor(output))):
                if value is None:
                    LOGGER.warning("layer %d returned no %s tensor; activation not captured", index, kind)
                    continue
                # Keep the graph-connected tensor until snapshot() so its
                # backward gradient can be captured as well.  retain_grad is
                # a no-op for leaf tensors and is safe when gradients are off.
                key = (index, kind)
                if value.requires_grad:
                    value.retain_grad()
                    self._activation_refs[key] = value
                self.activations[key] = _bf16_cpu(value)

        return hook

    def clear(self) -> None:
        self.activations.clear()
        self._activation_refs.clear()

    def snapshot(self, step: int | None = None) -> dict[str, Any]:
        """Copy selected layer tensors after a forward/backward step.

        Weight and gradient names are relative to the selected decoder layer,
        e.g. ``layer_16.self_attn.q_proj.weight``.  A missing gradient is
        represented by ``None`` (this can happen for unused parameters).  The
        ``activations`` contains both ``layer_<N>.input`` and
        ``layer_<N>.output`` when available.  The ``gradients`` category also
        contains matching activation gradients when those tensors participated
        in autograd.
        """

        weights: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor | None] = {}
        for layer_index in self.target_layers:
            layer = self.layers[layer_index]
            for name, parameter in layer.named_parameters(recurse=True):
                key = f"layer_{layer_index}.{name}"
                weights[key] = _bf16_cpu(parameter)
                gradients[key] = None if parameter.grad is None else _bf16_cpu(parameter.grad)

            for kind in ("input", "output"):
                activation_ref = self._activation_refs.get((layer_index, kind))
                gradients[f"layer_{layer_index}.{kind}"] = (
                    None
                    if activation_ref is None or activation_ref.grad is None
                    else _bf16_cpu(activation_ref.grad)
                )

        activations = {
            f"layer_{index}.{kind}": tensor.clone()
            for (index, kind), tensor in self.activations.items()
            if index in self.target_layers
        }
        return {
            "metadata": {
                "step": step,
                "target_layers": list(self.target_layers),
                "dtype": "bfloat16",
                "activation_semantics": "first tensor on decoder input and output",
            },
            "weights": weights,
            "activations": activations,
            "gradients": gradients,
        }

    @staticmethod
    def save_snapshot(snapshot: Mapping[str, Any], path: str | Path) -> None:
        """Save a snapshot as a PyTorch file, preserving BF16 on disk."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(snapshot), destination)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    def __enter__(self) -> "LayerTensorCapture":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def exponent_values(tensor: torch.Tensor) -> torch.Tensor:
    """Return the numeric BF16 exponent (0..255) for every element.

    The result is ``torch.int64`` and has the same shape as ``tensor``.  NaN,
    infinity, subnormal and zero values are intentionally not filtered: their
    exponent encodings are part of the requested whole-tensor statistic.
    """

    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
        dtype = getattr(tensor, "dtype", None)
        raise TypeError(f"BF16 tensor required, got dtype={dtype}")
    # A uint16 view is a zero-copy reinterpretation of the IEEE-754 BF16 bits.
    # Keep counting on CPU: this works on every PyTorch build (including builds
    # without a CUDA implementation of ``bincount``) and only transfers the
    # compact 8-bit exponent field.
    return ((tensor.detach().contiguous().view(torch.uint16).to(torch.int64).cpu() >> 7) & 0xFF)


def _window_result(
    exponent_counts: torch.Tensor, total: int, width: int, start: int
) -> dict[str, Any]:
    """Count elements whose exponent lies in ``[start, start + width - 1]``."""
    end = start + width - 1
    count = int(exponent_counts[start : end + 1].sum().item())
    return {
        "start_value": start,
        "end_value": end,
        "width": width,
        "values": list(range(start, end + 1)),
        "count": count,
        "proportion": (count / total) if total else 0.0,
        "total": total,
    }


def exponent_window_statistics(
    tensor: torch.Tensor,
    widths: Iterable[int] = (3, 7),
) -> dict[str, Any]:
    """Compute the most frequent consecutive numeric exponent-value windows."""

    exponents = exponent_values(tensor)
    exponent_counts = torch.bincount(exponents.reshape(-1), minlength=256)
    total = int(exponents.numel())
    result: dict[str, Any] = {
        "dtype": "bfloat16",
        "numel": int(tensor.numel()),
        "windows": {},
    }
    for width in widths:
        width = int(width)
        if not 1 <= width <= 256:
            raise ValueError(f"window width must be in [1, 256], got {width}")
        positions = [
            _window_result(exponent_counts, total, width, start)
            for start in range(257 - width)
        ]
        # The denominator is the complete tensor for every candidate, so this
        # comparison is equivalent to comparing counts.
        best_count = max((entry["count"] for entry in positions), default=0)
        top = [entry for entry in positions if entry["count"] == best_count]
        top_ranges = [entry["values"] for entry in top]
        result["windows"][str(width)] = {
            "width": width,
            "top": top,
            "top_ranges": top_ranges,
            "top_count": best_count,
            "top_proportion": (best_count / int(tensor.numel())) if tensor.numel() else 0.0,
        }
    return result


# Descriptive aliases for callers that prefer an imperative name.  Keeping a
# single implementation prevents subtle differences between live-training and
# offline analysis paths.
bf16_exponent_statistics = exponent_window_statistics
compute_bf16_exponent_statistics = exponent_window_statistics


def analyze_bf16_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Analyse every BF16 tensor in a named mapping."""

    reports: dict[str, Any] = {}
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name!r} is not a tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(
                f"{name!r} has dtype {tensor.dtype}; statistics require BF16 tensors"
            )
        reports[name] = exponent_window_statistics(tensor)
    return reports


def flatten_snapshot(snapshot: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Flatten weights, activations and gradients for analysis.

    The category is included in every name.  ``None`` gradients are omitted,
    because there are no elements whose exponent could be counted.
    """

    flattened: dict[str, torch.Tensor] = {}
    for category in ("weights", "activations", "gradients"):
        values = snapshot.get(category, {})
        if not isinstance(values, Mapping):
            continue
        for name, tensor in values.items():
            if tensor is not None:
                flattened[f"{category}.{name}"] = tensor
    return flattened


def analyze_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return metadata and exponent reports for a saved capture snapshot."""

    return {
        "metadata": dict(snapshot.get("metadata", {})),
        "tensors": analyze_bf16_tensors(flatten_snapshot(snapshot)),
    }


def summarize_tensor_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compact report containing the requested 3-value and 7-value windows."""

    windows = report.get("windows", {})
    summary: dict[str, Any] = {
        "dtype": report.get("dtype", "bfloat16"),
        "numel": report.get("numel", 0),
    }
    for width in (3, 7):
        entry = windows.get(str(width), {}) if isinstance(windows, Mapping) else {}
        summary[f"top_{width}_exponent_ranges"] = entry.get("top_ranges", [])
        summary[f"top_{width}_exponent_proportion"] = entry.get("top_proportion", 0.0)
    return summary


def _iter_snapshot_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        # Prefer the naming convention emitted by LayerTensorCapture.  This
        # avoids accidentally treating an experiment's lr_scheduler.pt as a
        # tensor snapshot when the experiment root is passed to the CLI.
        snapshots = sorted(path.rglob("step-*.pt"))
        if snapshots:
            yield from snapshots
        else:
            # Do not mistake the training scheduler state for a capture.
            yield from (
                candidate
                for candidate in sorted(path.rglob("*.pt"))
                if candidate.name != "lr_scheduler.pt"
            )
    else:
        raise FileNotFoundError(path)


def analyze_path(path: str | Path) -> dict[str, Any]:
    """Analyse one snapshot or all ``*.pt`` snapshots below a directory."""

    root = Path(path)
    reports = {}
    snapshots_by_step: dict[str, list[tuple[Path, Mapping[str, Any]]]] = {}
    for snapshot_path in _iter_snapshot_paths(root):
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        report = analyze_snapshot(snapshot)
        report["summary"] = {
            name: summarize_tensor_report(tensor_report)
            for name, tensor_report in report["tensors"].items()
        }
        reports[str(snapshot_path)] = report
        step_key = str(snapshot.get("metadata", {}).get("step", snapshot_path.stem))
        snapshots_by_step.setdefault(step_key, []).append((snapshot_path, snapshot))

    # FSDP stores parameter shards per rank.  Also provide an aggregate view so
    # proportions describe the complete logical tensor, not one rank-local
    # shard.  Flattening before concatenation is sufficient because exponent
    # counts are independent of tensor shape.
    if len(snapshots_by_step) > 1 or any(len(items) > 1 for items in snapshots_by_step.values()):
        aggregated: dict[str, Any] = {}
        for step, items in snapshots_by_step.items():
            parts: dict[str, list[torch.Tensor]] = {}
            metadata = dict(items[0][1].get("metadata", {}))
            metadata["aggregated_ranks"] = [p.parent.name for p, _ in items]
            for _, snapshot in items:
                for name, tensor in flatten_snapshot(snapshot).items():
                    parts.setdefault(name, []).append(tensor.reshape(-1))
            combined = {name: torch.cat(values) for name, values in parts.items()}
            aggregated[step] = {
                "metadata": metadata,
                "tensors": analyze_bf16_tensors(combined),
            }
        reports["aggregated"] = aggregated
    return reports


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="capture .pt file or directory")
    parser.add_argument("-o", "--output", type=Path, help="optional JSON output path")
    args = parser.parse_args(argv)
    reports = analyze_path(args.path)
    encoded = json.dumps(reports, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
