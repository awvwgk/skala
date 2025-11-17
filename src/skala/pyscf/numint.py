# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor

from skala.functional.base import ExcFunctionalBase
from skala.pyscf.features import generate_features


class LibXCSpec(Protocol):
    __version__: str | None
    __references__: str | None

    @staticmethod
    def is_hybrid_xc(xc: str) -> bool: ...

    @staticmethod
    def is_nlc(xc: str) -> bool: ...


class PySCFNumInt(Protocol):
    """Interface for PySCF-compatible numint functionals."""

    libxc: LibXCSpec

    def get_rho(
        self, mol: gto.Mole, dm: np.ndarray, grids: dft.Grids, max_memory: int = 2000
    ) -> np.ndarray: ...

    def nr_rks(
        self,
        mol: gto.Mole,
        grids: dft.Grids,
        xc_code: str | None,
        dm: np.ndarray,
        max_memory: int = 2000,
    ) -> tuple[float, float, np.ndarray]:
        """Restricted Kohn-Sham method, applicable if both spin-densities as equal."""
        ...

    def nr_uks(
        self,
        mol: gto.Mole,
        grids: dft.Grids,
        xc_code: str | None,
        dm: np.ndarray,
        max_memory: int = 2000,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Unrestricted Kohn-Sham method, spin densities can be different."""
        ...

    def rsh_and_hybrid_coeff(self) -> tuple[float, float, float]:
        return 0, 0, 0

    def gen_response(
        self,
        mo_coeff: np.ndarray | None,
        mo_occ: np.ndarray | None,
        *,
        ks: dft.rks.RKS | dft.uks.UKS,
        **kwargs: Any,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Generates the response function for the functional."""
        ...


class SkalaNumInt(PySCFNumInt):
    """PySCF-compatible reimplementation of `pyscf.dft.numint.NumInt`.

    Evaluation of atomic orbitals and one-electron integrals on a grid
    is cached for speed.

    Example
    -------
    >>> from pyscf import gto, dft
    >>> from skala.functional import load_functional
    >>> from skala.pyscf.numint import SkalaNumInt
    >>>
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp", verbose=0)
    >>> ks = dft.KS(mol)
    >>> ks._numint = SkalaNumInt(load_functional("skala"))
    >>> energy = ks.kernel()
    >>> print(energy)  # DOCTEST: Ellipsis
    -1.142330...
    """

    def __init__(self, functional: ExcFunctionalBase, chunk_size: int = 1000):
        self._ao_cache: dict[tuple[gto.Mole, dft.Grids, int], tuple[Tensor, int]] = {}
        self.func = functional
        self.chunk_size = chunk_size

    def get_rho(
        self, mol: gto.Mole, dm: np.ndarray, grids: dft.Grids, max_memory: int = 2000
    ) -> np.ndarray:
        mol_features = generate_features(
            mol,
            torch.from_numpy(dm),
            grids,
            features={"density"},
            _ao_cache=self._ao_cache,
            chunk_size=self.chunk_size,
            max_memory=max_memory,
        )
        return mol_features["density"].sum(0).numpy()

    def __call__(
        self,
        mol: gto.Mole,
        grids: dft.Grids,
        xc_code: str | None,
        dm: Tensor,
        second_order: bool = False,
        max_memory: int = 2000,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dm = dm.requires_grad_()
        mol_features = generate_features(
            mol,
            dm,
            grids,
            set(self.func.features),
            _ao_cache=self._ao_cache,
            chunk_size=self.chunk_size,
            max_memory=max_memory,
        )
        for k, v in mol_features.items():
            mol_features[k] = v.to(self.device)
        E_xc = self.func.get_exc(mol_features)
        (V_xc,) = torch.autograd.grad(
            E_xc,
            dm,
            torch.ones_like(E_xc),
            retain_graph=second_order,
            create_graph=second_order,
        )

        rho = mol_features["density"]
        grid_weights = mol_features.get(
            "grid_weights", torch.from_numpy(grids.weights).to(self.device)
        )
        N = (rho * grid_weights).sum(dim=-1)
        return N.cpu(), E_xc.cpu(), V_xc.cpu()

    @property
    def device(self) -> torch.device:
        try:
            return next(self.func.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def nr_rks(
        self,
        mol: gto.Mole,
        grids: dft.Grids,
        xc_code: str | None,
        dm: np.ndarray,
        max_memory: int = 2000,
    ) -> tuple[float, float, np.ndarray]:
        """Restricted Kohn-Sham method, applicable if both spin-densities as equal."""
        assert len(dm.shape) == 2
        N, E_xc, V_xc = self(
            mol, grids, xc_code, torch.from_numpy(dm), max_memory=max_memory
        )
        return N.sum().item(), E_xc.item(), V_xc.numpy()

    def nr_uks(
        self,
        mol: gto.Mole,
        grids: dft.Grids,
        xc_code: str | None,
        dm: np.ndarray,
        max_memory: int = 2000,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Unrestricted Kohn-Sham method, spin densities can be different."""
        assert len(dm.shape) == 3 and dm.shape[0] == 2
        N, E_xc, V_xc = self(
            mol, grids, xc_code, torch.from_numpy(dm), max_memory=max_memory
        )
        return N.detach().numpy(), E_xc.item(), V_xc.numpy()

    class libxc:
        __version__ = None
        __reference__ = None

        @staticmethod
        def is_hybrid_xc(xc: str) -> bool:
            return False

        @staticmethod
        def is_nlc(xc: str) -> bool:
            return False

    def gen_response(
        self,
        mo_coeff: np.ndarray | None,
        mo_occ: np.ndarray | None,
        *,
        ks: dft.rks.RKS | dft.uks.UKS,
        **kwargs: Any,
    ) -> Callable[[np.ndarray], np.ndarray]:
        assert mo_coeff is not None
        assert mo_occ is not None
        if kwargs is not None:
            # check if kwargs are valid
            # this response function only works for KS DFT with meta GGA
            if "hermi" in kwargs:
                assert kwargs["hermi"] == 1
            if "singlet" in kwargs:
                assert kwargs["singlet"] is None
            if "with_j" in kwargs:
                assert kwargs["with_j"]

        dm0 = torch.from_numpy(ks.make_rdm1(mo_coeff, mo_occ))
        # caching V_xc saves a forward pass in each iteration
        V_xc = self(ks.mol, ks.grids, None, dm0, second_order=True)[2]

        def hessian_vector_product(dm1: np.ndarray) -> np.ndarray:
            v1 = (
                torch.autograd.grad(
                    V_xc, dm0, torch.from_numpy(dm1), retain_graph=True
                )[0]
                .detach()
                .numpy()
            )
            vj = ks.get_j(ks.mol, dm1, hermi=1)

            if ks.mol.spin == 0:
                v1 += vj
            else:
                v1 += vj[0] + vj[1]

            return v1

        return hessian_vector_product
