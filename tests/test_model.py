# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch
from pyscf import dft, gto, scf

from skala.functional import load_functional
from skala.functional.model import (
    SkalaFunctional,
    exp_radial_func,
)
from skala.pyscf.features import generate_features

torch.manual_seed(0)


@pytest.fixture(scope="session")
def mol() -> gto.Mole:
    mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="def2-qzvp", cart=True)
    return mol


def get_mf_dm(mol: gto.Mole) -> tuple[scf.hf.SCF, np.ndarray]:
    ks = dft.KS(
        mol,
        xc="pbe",
    )(
        grids=dft.Grids(mol)(level=1, radi_method=dft.radi.treutler).build(),
        max_cycle=1,
    )
    ks.kernel()
    return ks, ks.make_rdm1()


def test_vne3nn_invariance(mol: gto.Mole):
    # fix np seed
    np.random.seed(0)

    ks, dm = get_mf_dm(mol)

    model = SkalaFunctional(non_local=True)

    dm_torch1 = torch.from_numpy(dm).float()

    exc1 = model.get_exc(
        generate_features(mol, dm_torch1, ks.grids, set(model.features))
    )

    # Check that the model is invariant to the rotation of the coordinates
    Q = np.linalg.qr(np.random.randn(3, 3)).Q
    atom_coords = mol.atom_coords()
    mol.set_geom_(atom_coords @ Q, unit="bohr")
    assert mol.atom_coords() == pytest.approx(atom_coords @ Q, abs=1e-6)

    ks, dm = get_mf_dm(mol)
    dm_torch2 = torch.from_numpy(dm).float().requires_grad_(True)
    assert not torch.allclose(dm_torch1, dm_torch2)

    exc2 = model.get_exc(
        generate_features(mol, dm_torch2, ks.grids, set(model.features))
    )

    assert torch.allclose(exc1, exc2, atol=0.00001)


def test_double_precision(mol: gto.Mole):
    # this ensures the functional can handle double precision inputs
    ks, dm = get_mf_dm(mol)

    model = SkalaFunctional(non_local=True)

    model.double()
    dm_torch1 = torch.from_numpy(dm).double()

    _ = model.get_exc(generate_features(mol, dm_torch1, ks.grids, set(model.features)))


def test_exp_radial_func_normalization():
    N, num_basis = 100000, 16
    xx = torch.linspace(-10, 10, N)
    dx = 20 / N

    emb = exp_radial_func(xx, num_basis=num_basis, dim=1)
    assert list(emb.shape) == [N, num_basis]

    integrals = (emb * dx).sum(0)
    assert torch.isclose(
        integrals, torch.ones_like(integrals), atol=1e-4, rtol=1e-4
    ).all(), integrals


def test_traced_functional_and_loaded_functional_are_equal():
    # This test ensures that the traced functional and the loaded functional
    # give the same output for the same input.

    traced_model = load_functional("skala")
    clean_state_dict = {
        k.replace("_traced_model.", ""): v for k, v in traced_model.state_dict().items()
    }

    model = SkalaFunctional(lmax=3, radius_cutoff=5.0)
    model.load_state_dict(clean_state_dict, strict=True)

    # Create a dummy load_input
    num_grid_points = 10
    grid_coords = torch.randn(num_grid_points, 3)
    density = torch.randn(2, num_grid_points)
    gradients = torch.randn(2, 3, num_grid_points)
    kin = torch.randn(2, num_grid_points)
    grid_weights = torch.randn(num_grid_points)
    atom_coords = torch.tensor([[0.0, 0.0, 0.0]])
    features_dict = {
        "density": density,
        "grad": gradients,
        "kin": kin,
        "grid_weights": grid_weights,
        "grid_coords": grid_coords,
        "coarse_0_atomic_coords": atom_coords,
    }
    original_output = model.get_exc(features_dict)
    traced_model_output = traced_model.get_exc(features_dict)

    # Compare outputs
    assert torch.allclose(original_output, traced_model_output, atol=1e-5)
