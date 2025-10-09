# SPDX-License-Identifier: MIT

"""
Tools to load functionals from serialized torchscript checkpoints.
"""

import json
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from skala.functional.base import ExcFunctionalBase

PROTOCOL_VERSION = 2


class TracedFunctional(ExcFunctionalBase):
    """
    A TracedFunctional is a functional represented by a torch.jit graph.
    It can be saved to disk and loaded again with no dependencies on our code base (only libtorch).

    You can construct a TracedFunctional from any functional that implements `ExcFunctionalBase`.
    When you do that, the execution is traced with a dummy input, and the resulting graph is saved.
    The main thing to keep in mind, is that, if the functional uses branches or loops that are
    runtime-dependent, the traced graph will only take the path that's taken by the dummy input.
    This behavior can sometimes be avoided by using the `@torch.jit.script_if_tracing` decorator.

    To learn about tracing and torchscript, see
        https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
    and for a few more advanced topics
        https://paulbridger.com/posts/mastering-torchscript/
    """

    def __init__(
        self,
        traced_model: torch.jit.ScriptModule,
        *,
        features: Iterable[str],
        metadata: Mapping[str, Any],
        expected_d3_settings: str | None,
    ):
        super().__init__()
        self._traced_model = traced_model
        self.metadata = dict(metadata)
        self.features = list(features)
        self.expected_d3_settings = expected_d3_settings

    def get_d3_settings(self) -> str | None:
        """
        Returns the D3 settings that this functional expects.
        If the functional does not use D3, it returns None.
        """
        return self.expected_d3_settings

    def get_exc_density(self, data: dict[str, torch.Tensor]) -> torch.FloatTensor:
        return self._traced_model.get_exc_density(data)

    def get_exc(self, data: dict[str, torch.Tensor]) -> torch.FloatTensor:
        return self._traced_model.get_exc(data)

    @property
    def original_name(self) -> str:
        return self._traced_model.original_name

    @classmethod
    def load(cls, fp, device: torch.device | None = None):
        extra_files = {
            "metadata": b"",
            "features": b"",
            "expected_d3_settings": b"",
            "protocol_version": b"",
        }

        traced_model = torch.jit.load(fp, _extra_files=extra_files, map_location=device)

        metadata = json.loads(extra_files["metadata"].decode("utf-8"))
        features = json.loads(extra_files["features"].decode("utf-8"))
        protocol_version = json.loads(extra_files["protocol_version"].decode("utf-8"))
        expected_d3_settings = json.loads(
            extra_files["expected_d3_settings"].decode("utf-8")
        )

        if protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"Cannot load model with protocol version {protocol_version} "
                f"with this version of the library (supports {PROTOCOL_VERSION})"
            )

        return cls(
            traced_model,
            features=features,
            metadata=metadata,
            expected_d3_settings=expected_d3_settings,
        )
