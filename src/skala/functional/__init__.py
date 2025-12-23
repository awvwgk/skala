# SPDX-License-Identifier: MIT

"""
Functional module for exchange-correlation functionals.

This module provides the main interface for loading and using various
exchange-correlation functionals, including traditional functionals
(LDA, PBE, TPSS) and the Skala neural functional.
"""

import os

import torch
from huggingface_hub import hf_hub_download

from skala.functional.base import ExcFunctionalBase
from skala.functional.load import TracedFunctional
from skala.functional.traditional import LDA, PBE, SPW92, TPSS

__all__ = [
    "ExcFunctionalBase",
    "TracedFunctional",
    "LDA",
    "PBE",
    "SPW92",
    "TPSS",
    "load_functional",
]


def load_functional(name: str, device: torch.device | None = None) -> ExcFunctionalBase:
    """
    Load an exchange-correlation functional by name.

    Parameters
    ----------
    name : str
        Name of the functional. Supported values:

        - "skala": The Skala neural functional
        - "lda": Local Density Approximation
        - "spw92": SPW92 (LDA with PW92 correlation)
        - "pbe": Perdew-Burke-Ernzerhof functional
        - "tpss": Tao-Perdew-Staroverov-Scuseria meta-GGA

    Returns
    -------
    ExcFunctionalBase
        The loaded functional instance.

    Raises
    ------
    ValueError
        If the functional name is not recognized.

    Example
    -------
    >>> func = load_functional("skala")
    >>> func.features
    ['density', 'kin', 'grad', 'grid_coords', 'grid_weights', 'coarse_0_atomic_coords']
    >>> func = load_functional("lda")
    >>> func.features
    ['density', 'grid_weights']
    """
    if name.lower() == "skala":
        env_path = os.environ.get("SKALA_LOCAL_MODEL_PATH")
        if env_path is not None:
            path = env_path
        else:
            filename = (
                "skala-1.0.fun"
                if device is None or device.type == "cpu"
                else "skala-1.0-cuda.fun"
            )
            path = hf_hub_download(repo_id="microsoft/skala", filename=filename)
        with open(path, "rb") as fd:
            return TracedFunctional.load(fd, device=device)

    if name.lower() == "lda":
        return LDA().to(device=device)

    if name.lower() == "spw92":
        return SPW92().to(device=device)

    if name.lower() == "pbe":
        return PBE().to(device=device)

    if name.lower() == "tpss":
        return TPSS().to(device=device)

    raise ValueError(
        f"Unknown functional: {name}. Please provide a valid functional name or path to a traced functional file."
    )
