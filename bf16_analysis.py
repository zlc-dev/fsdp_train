"""Capture raw tensors from selected Transformer layers.

This module intentionally preserves source dtypes and performs no numeric
statistics. FP8 E5M2 conversion and analysis belong to a separate offline
project.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


LOGGER = logging.getLogger(__name__)
DEFAULT_TARGET_LAYERS = (0, 16, 32)


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    """Detach a tensor and return a CPU copy without changing its dtype.

    Captures are intentionally copied before the optimizer can mutate the
    parameter or before a subsequent forward pass overwrites an activation.
    Numeric conversion is intentionally deferred to offline analysis.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(value)!r}")
    # FSDP2 may expose a distributed tensor (DTensor) for a sharded parameter.
    # Captures are rank-local by design; use the local shard when available.
    if hasattr(value, "to_local"):
        value = value.to_local()
    return value.detach().to(device="cpu").clone()


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
                self.activations[key] = _cpu_tensor(value)

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
                weights[key] = _cpu_tensor(parameter)
                gradients[key] = None if parameter.grad is None else _cpu_tensor(parameter.grad)

            for kind in ("input", "output"):
                activation_ref = self._activation_refs.get((layer_index, kind))
                gradients[f"layer_{layer_index}.{kind}"] = (
                    None
                    if activation_ref is None or activation_ref.grad is None
                    else _cpu_tensor(activation_ref.grad)
                )

        activations = {
            f"layer_{index}.{kind}": tensor.clone()
            for (index, kind), tensor in self.activations.items()
            if index in self.target_layers
        }
        tensor_groups = {
            "weights": weights,
            "activations": activations,
            "gradients": gradients,
        }
        tensor_dtypes = {
            f"{category}.{name}": str(tensor.dtype)
            for category, values in tensor_groups.items()
            for name, tensor in values.items()
            if tensor is not None
        }
        return {
            "metadata": {
                "step": step,
                "target_layers": list(self.target_layers),
                "tensor_dtypes": tensor_dtypes,
                "activation_semantics": "first tensor on decoder input and output",
            },
            "weights": weights,
            "activations": activations,
            "gradients": gradients,
        }

    @staticmethod
    def save_snapshot(snapshot: Mapping[str, Any], path: str | Path) -> None:
        """Save a snapshot as a PyTorch file, preserving source dtypes."""

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
