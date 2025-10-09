# SPDX-License-Identifier: MIT

"""
Methods for generating and manipulating density features.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import copy

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor

DEFAULT_FEATURES = ["density", "kin", "grad", "grid_coords", "grid_weights"]
DEFAULT_FEATURES_SET = set(DEFAULT_FEATURES)


def maybe_from_numpy(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def maybe_expand_and_divide(
    feature: torch.Tensor, expand: bool, divisor: float
) -> torch.Tensor:
    """
    Expand feature along spin channels and divide its value by divisor if expand is True.
    """
    if expand:
        return torch.stack([feature / divisor, feature / divisor], dim=0)
    else:
        return feature


def generate_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: dft.Grids,
    features: set[str] | None = None,
    _ao_cache: dict[tuple[gto.Mole, dft.Grids, int], tuple[Tensor, int]] | None = None,
    max_memory: int = 2000,
    chunk_size: int = 1000,
) -> dict[str, Tensor]:
    """Generate density features for a given molecule. The density features are stored in a dictionary
    with the keys matching the requested features.

    Parameters
    ----------
    mol: gto.Mole
      the molecule
    dm: Tensor
      the density matrix
    grids: dft.Grids
      the grid
    features: set[str] | None
      the requested features
    _ao_cache:
      a cache for the atomic orbitals
    max_memory: int
      the maximum memory to use for calculating the features
    chunk_size: int
      the number of grid points to process at a time for the Hartree-Fock exchange
      energy density, for which the Jacobian-vector product is cached in a smarter way to
      avoid memory issues.

    Returns
    -------
    dict[str, Tensor]
        A dictionary containing the requested features. The keys are the feature names,
        and the values are the corresponding tensors.
    """
    features = features or DEFAULT_FEATURES_SET

    # if dm is a 3D tensor, then we have a spin-polarized system
    with_spin = True if len(dm.shape) == 3 else False

    mol_features = {}

    if "grid_coords" in features:
        mol_features["grid_coords"] = torch.from_numpy(grids.coords).to(dm.dtype)

    if "grid_weights" in features:
        mol_features["grid_weights"] = torch.from_numpy(grids.weights).to(dm.dtype)

    if "coarse_0_atomic_coords" in features:
        mol_features["coarse_0_atomic_coords"] = maybe_from_numpy(
            mol.atom_coords()
        ).double()

    with_mgga_feature = (
        "density" in features
        or "grad" in features
        or "kin" in features
        or "lapl" in features
    )
    if with_mgga_feature:
        mgga_features = auto_chunk(
            dm,
            mol,
            grids,
            MGGAFeatureFunction(
                with_density="density" in features,
                with_grad="grad" in features,
                with_kin="kin" in features,
                with_lapl="lapl" in features,
            ),
            max_memory_cpu=max_memory,
        )

        for feature in mgga_features:
            mol_features[feature] = maybe_expand_and_divide(
                mgga_features[feature], not with_spin, 2
            )

    return mol_features


def is_density_feature(feature: str) -> bool:
    return feature in {"density", "grad", "kin"}


def eval_ao(
    mol: gto.Mole,
    grids: dft.Grids,
    *,
    deriv: int,
    cache: dict[tuple[gto.Mole, dft.Grids, int], tuple[Tensor, int]] | None = None,
) -> Tensor:
    cache = cache or {}
    if cached := cache.get((mol, grids, deriv)):
        ao, size = cached
        if size == grids.size:
            return ao
    ao = torch.from_numpy(dft.numint.eval_ao(mol, grids.coords, deriv=deriv))
    if deriv == 0:
        ao = ao[None, ...]
    cache[mol, grids, deriv] = ao, grids.size
    return ao


def partial_feature_function_over_aos(
    feature_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ao: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a function that computes the feature function with the given ao,
    but not the dm already passed to the function.

    Purpose is to allow for chaining of derivatives.
    """

    def partial_feature_function(dm: torch.Tensor) -> torch.Tensor:
        return feature_function(dm, ao)

    return partial_feature_function


def partial_jvp_function_over_tangents(
    func: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
    tangents: torch.Tensor,
):
    """Returns a function that computes the jvp of the given function with tangents,
    but not primals already passed to the function.

    Purpose is to allow for chaining of derivatives over primals."""

    def reduced_jvp(primals: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.func.jvp(func, (primals,), (tangents,))[1]

    return reduced_jvp


def partial_vjp_function_over_tangents(
    func: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
    tangents: torch.Tensor,
):
    """Returns a function that computes the vjp of the given function with tangents,
    but not primals already passed to the function.

    Purpose is to allow for chaining of derivatives over primals."""

    def reduced_vjp(primals: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.func.vjp(func, primals)[1](tangents)[0]

    return reduced_vjp


class FeatureFunction(torch.nn.Module, ABC):
    deriv: int
    nfeats: int
    only_linear_feats: bool

    @abstractmethod
    def forward(self, dm: torch.Tensor, ao: torch.Tensor) -> torch.Tensor: ...


class MGGAFeatureFunction(FeatureFunction):
    with_density: bool
    with_grad: bool
    with_kin: bool
    with_lapl: bool
    with_ked_var: bool
    with_ked_det: bool

    def __init__(
        self,
        with_density: bool = True,
        with_grad: bool = True,
        with_kin: bool = True,
        with_lapl: bool = False,
        with_ked_var: bool = False,
        with_ked_det: bool = False,
    ):
        super().__init__()

        self.with_density = with_density
        self.with_grad = with_grad
        self.with_kin = with_kin
        self.with_lapl = with_lapl
        self.with_ked_var = with_ked_var
        self.with_ked_det = with_ked_det

        self.deriv = 0
        if with_grad or with_kin or with_ked_var or with_ked_det:
            self.deriv = 1
        if with_lapl:
            self.deriv = 2

        self.nfeats = (
            with_density
            + with_grad * 3
            + with_kin
            + with_lapl
            + with_ked_var
            + with_ked_det
        )

        if self.nfeats == 0:
            raise ValueError("At least one feature must be selected.")

        self.only_linear_feats = not (with_ked_var or with_ked_det)

    def to_dict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert the features to a dictionary with the keys being the feature names."""
        feature_index = 0
        feature_dict = {}
        if self.with_density:
            feature_dict["density"] = features[..., feature_index, :]
            feature_index += 1
        if self.with_grad:
            feature_dict["grad"] = features[..., feature_index : feature_index + 3, :]
            feature_index += 3
        if self.with_kin:
            feature_dict["kin"] = features[..., feature_index, :]
            feature_index += 1
        if self.with_lapl:
            feature_dict["lapl"] = features[..., feature_index, :]
            feature_index += 1
        if self.with_ked_var:
            feature_dict["ked_var"] = features[..., feature_index, :]
            feature_index += 1
        if self.with_ked_det:
            feature_dict["ked_det"] = features[..., feature_index, :]
            feature_index += 1
        return feature_dict

    def forward(self, dm: torch.Tensor, ao: torch.Tensor) -> torch.Tensor:
        with_Q: bool = self.with_ked_var or self.with_ked_det

        # Flatten all but the last two dimensions
        # then restore the original shape at the end
        dm_view = dm.view(-1, dm.shape[-2], dm.shape[-1])
        # Explicit symmetrization for autodiff
        dm_view = 0.5 * (dm_view + dm_view.transpose(-1, -2))

        features = torch.zeros(
            (dm_view.shape[0], self.nfeats, ao.shape[-1]),
            device=dm.device,
            dtype=dm.dtype,
        )

        # Handle the density only case, where ao has one dim less
        if self.deriv == 0:
            c0 = dm_view @ ao
            features[..., 0, :] = torch.sum(c0 * ao[None, :, :], dim=-2)
            if len(dm.shape) == 2:
                return features.reshape((self.nfeats, -1))
            else:
                return features.reshape((*dm.shape[:-2], self.nfeats, -1))

        c0 = dm_view @ ao[0]

        feat_idx = 0
        if self.with_density:
            features[..., feat_idx, :] = torch.sum(c0 * ao[0][None, :, :], dim=-2)
            feat_idx += 1

        if self.with_grad:
            for i in range(3):
                features[..., feat_idx, :] = 2 * torch.sum(
                    c0 * ao[i + 1][None, :, :], dim=-2
                )
                feat_idx += 1

        if (self.with_kin or self.with_lapl) and not with_Q:
            for i in range(3):
                ci = dm_view @ ao[i + 1]
                features[..., feat_idx, :] += 0.5 * torch.sum(
                    ci * ao[i + 1][None, :, :], dim=-2
                )

            if self.with_kin:
                feat_idx += 1
                if self.with_lapl:
                    features[..., feat_idx, :] = 4 * features[..., feat_idx - 1, :]
            else:
                # Multiply times four for the laplacian
                features[..., feat_idx, :] *= 4.0

            if self.with_lapl:
                # 0 is without derivative
                # 1 2 3 are x y z derivatives
                # 4 5 6 are xx xy xz derivatives
                # 7 8 9 are yy yz zz derivatives
                for i in (4, 7, 9):
                    features[..., feat_idx, :] += 2 * torch.sum(
                        c0 * ao[i][None, :, :], dim=-2
                    )

        if with_Q:
            Q = torch.zeros(
                (dm_view.shape[0], ao.shape[-1], 3, 3), device=dm.device, dtype=dm.dtype
            )

            for i in range(3):
                ci = dm_view @ ao[i + 1]
                for j in range(i, 3):
                    Q = torch.sum(ci * ao[j + 1][None, :, :], dim=-2)

            if self.with_kin:
                features[..., feat_idx, :] = 0.5 * torch.einsum("...ii->...", Q)
                feat_idx += 1

            if self.with_lapl:
                features[..., feat_idx, :] = 2 * torch.einsum("...ii->...", Q)
                # 0 is without derivative
                # 1 2 3 are x y z derivatives
                # 4 5 6 are xx xy xz derivatives
                # 7 8 9 are yy yz zz derivatives
                for i in (4, 7, 9):
                    features[..., feat_idx, :] += 2 * torch.sum(
                        c0 * ao[i][None, :, :], dim=-2
                    )
                feat_idx += 1

            if self.with_ked_var:
                if not self.with_kin:
                    trace = torch.einsum("...ii->...", Q)
                else:
                    trace = 2 * features[:, feat_idx - 1, :]
                features[..., feat_idx, :] = 0.5 * torch.sum(
                    (
                        trace[:, None, None]
                        * torch.eye(3, device=dm.device, dtype=dm.dtype)[None, :, :]
                        - Q
                    )
                    ** 2,
                    dim=(-2, -1),
                )
                feat_idx += 1

            if self.with_ked_det:
                features[..., feat_idx, :] = torch.det(Q)
                feat_idx += 1
        if len(dm.shape) == 2:
            return features.reshape((self.nfeats, -1))
        else:
            return features.reshape((*dm.shape[:-2], self.nfeats, -1))


class ChunkEvalForward(torch.autograd.Function):
    @staticmethod
    def setup_context(
        ctx,
        inputs: tuple[
            torch.Tensor,
            gto.Mole,
            dft.Grids,
            FeatureFunction,
            int,
            int,
            bool,
            bool,
            torch.Tensor,
        ],
        output: torch.Tensor,
    ):
        (
            ctx.dm,
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.blksize,
            ctx.compile_feature_function,
            *ctx.vectors_jvp,
        ) = inputs
        ctx.save_for_backward(ctx.dm)

    @staticmethod
    def forward(
        dm: torch.Tensor,
        mol: gto.Mole,
        grids: dft.Grids,
        feature_function: FeatureFunction,
        blksize: int,
        compile_feature_function: bool,
        *vectors_jvp: torch.Tensor,
    ) -> torch.Tensor:
        ngrids = grids.weights.size
        block_loop_args = (mol, grids, mol.nao)
        block_loop_kwargs = {"deriv": feature_function.deriv, "blksize": blksize}
        ni = dft.numint.NumInt()
        sort_idx = np.arange(mol.nao_nr())

        features = torch.zeros(
            *dm.shape[:-2],
            feature_function.nfeats,
            ngrids,
            device=dm.device,
            dtype=dm.dtype,
        )
        if len(vectors_jvp) > 1 and features.only_linear_feats:
            return features

        end = 0
        for ao_block, mask, weights, _ in ni.block_loop(
            *block_loop_args, **block_loop_kwargs
        ):
            start, end = end, end + weights.size
            # Mask dm to only include the relevant AOs
            mask = torch.arange(mol.nao_nr(), device=dm.device)
            masked_dm = dm[..., sort_idx, :][..., sort_idx][
                ..., mask[:, None], mask[None, :]
            ]

            # Apply chain rule for this particular block
            partial_func = partial_feature_function_over_aos(
                feature_function,
                torch.from_numpy(ao_block).transpose(-1, -2).to(dm.device),
            )
            for vector_jvp in vectors_jvp:
                partial_func = partial_jvp_function_over_tangents(
                    partial_func,
                    vector_jvp[..., sort_idx, :][..., sort_idx][
                        ..., mask[:, None], mask[None, :]
                    ],
                )

            # Compute feature (or its jvp) for this block with masked dm
            if compile_feature_function:
                temp_feature = torch.compile(partial_func)(masked_dm)
            else:
                temp_feature = partial_func(masked_dm)

            features[..., start:end] = temp_feature
        return features

    @staticmethod
    def jvp(ctx, grad_input: torch.Tensor) -> torch.Tensor:
        # Chain rule for the jvp
        return ChunkEvalForward.apply(
            ctx.dm,
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.blksize,
            ctx.compile_feature_function,
            *ctx.vectors_jvp,
            grad_input,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # After one vjp (backward) the signature of the function changes from dm.shape -> (*dm.shape[:-2], nfeats, ngrid) to dm.shape -> dm.shape
        # therefore we move to a different function that does essentially the same thing, but with the new signature

        # Derivative to dm
        grads = [
            ChunkEvalBackward.apply(
                ctx.dm,
                ctx.mol,
                ctx.grids,
                ctx.feature_function,
                ["jvp"] * len(ctx.vectors_jvp) + ["first_vjp"],
                ctx.blksize,
                ctx.compile_feature_function,
                *ctx.vectors_jvp,
                grad_output,
            )
        ]

        # We need to provide None for the gradients of the non-differentiable inputs
        # these are mol (1), grids (2), feature_function (3), blksize (4),
        # compile_feature_function (5)
        num_non_differentiable_inputs = 5

        grads += [None] * num_non_differentiable_inputs

        # Gradients of earlier tangents
        for i in range(len(ctx.vectors_jvp)):
            derivative_types = ["jvp"] * len(ctx.vectors_jvp)
            derivative_types[i] = "first_vjp"
            grads.append(
                ChunkEvalBackward.apply(
                    ctx.dm,
                    ctx.mol,
                    ctx.grids,
                    ctx.feature_function,
                    derivative_types,
                    ctx.blksize,
                    ctx.compile_feature_function,
                    *ctx.vectors_jvp[:i],
                    grad_output,
                    *ctx.vectors_jvp[i + 1 :],
                )
            )
        return tuple(grads)


class ChunkEvalBackward(torch.autograd.Function):
    @staticmethod
    def setup_context(
        ctx,
        inputs: tuple[
            torch.Tensor,
            gto.Mole,
            dft.Grids,
            FeatureFunction,
            list[str],
            int,
            bool,
            torch.Tensor,
        ],
        output: tuple[torch.Tensor, ...],
    ):
        (
            ctx.dm,
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.derivative_types,
            ctx.blksize,
            ctx.compile_feature_function,
            *ctx.vectors,
        ) = inputs

        ctx.save_for_backward(ctx.dm)

    @staticmethod
    def forward(
        dm: torch.Tensor,
        mol: gto.Mole,
        grids: dft.Grids,
        feature_function: FeatureFunction,
        derivative_types: list[str],
        blksize: int,
        compile_feature_function: bool,
        *vectors: torch.Tensor,
    ) -> torch.Tensor:
        block_loop_args = (mol, grids, mol.nao)
        block_loop_kwargs = {"deriv": feature_function.deriv, "blksize": blksize}
        ni = dft.numint.NumInt()
        sort_idx = np.arange(mol.nao_nr())

        end: int = 0
        out = torch.zeros_like(dm)
        if len(vectors) > 1 and feature_function.only_linear_feats:
            return out

        unsort_idx = torch.argsort(torch.tensor(sort_idx))
        for ao_block, mask, weights, _ in ni.block_loop(
            *block_loop_args,
            **block_loop_kwargs,
        ):
            start, end = end, end + weights.size

            # Mask to only include the relevant AOs
            mask = torch.arange(mol.nao_nr(), device=dm.device)

            # Apply chain rule for this particular block
            # but be careful with signature change upon first vjp
            partial_func = partial_feature_function_over_aos(
                feature_function,
                torch.from_numpy(ao_block).transpose(-1, -2).to(dm.device),
            )
            for derivative_type, vector in zip(derivative_types, vectors, strict=True):
                if derivative_type == "jvp":
                    partial_func = partial_jvp_function_over_tangents(
                        partial_func,
                        vector[..., sort_idx, :][..., sort_idx][
                            ..., mask[:, None], mask[None, :]
                        ],
                    )
                elif derivative_type == "vjp":
                    partial_func = partial_vjp_function_over_tangents(
                        partial_func,
                        vector[..., sort_idx, :][..., sort_idx][
                            ..., mask[:, None], mask[None, :]
                        ],
                    )
                elif derivative_type == "first_vjp":
                    partial_func = partial_vjp_function_over_tangents(
                        partial_func, vector[..., start:end]
                    )
                else:
                    raise ValueError(
                        f"Unknown derivative {derivative_type} (must be one of 'jvp', 'vjp', 'first_vjp')"
                    )
            if compile_feature_function:
                out[..., mask[:, None], mask[None, :]] += torch.compile(partial_func)(
                    dm[..., sort_idx, :][..., sort_idx][
                        ..., mask[:, None], mask[None, :]
                    ]
                )
            else:
                out[..., mask[:, None], mask[None, :]] += partial_func(
                    dm[..., sort_idx, :][..., sort_idx][
                        ..., mask[:, None], mask[None, :]
                    ]
                )
        return out[..., unsort_idx, :][..., unsort_idx]

    @staticmethod
    def jvp(ctx, *grad_input: torch.Tensor) -> torch.Tensor:
        # Chain rule for the jvp
        return ChunkEvalBackward.apply(
            ctx.dm,
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.derivative_types + ["jvp"],
            ctx.blksize,
            ctx.compile_feature_function,
            *ctx.vectors,
            grad_input,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Chain rule for the vjp

        # Gradient corresponding to dm
        grads = [
            ChunkEvalBackward.apply(
                ctx.dm,
                ctx.mol,
                ctx.grids,
                ctx.feature_function,
                ctx.derivative_types + ["vjp"],
                ctx.blksize,
                ctx.compile_feature_function,
                *ctx.vectors,
                grad_output,
            )
        ]
        # We need to provide None for the gradients of the non-differentiable inputs
        # these are mol (1), grids (2), feature_function (3), derivative_types (4), blksize (5),
        # compile_feature_function (6)
        num_non_differentiable_inputs = 6

        grads += [None] * num_non_differentiable_inputs
        # Gradients of gradients
        for i, derivative_type in enumerate(ctx.derivative_types):
            derivative_types = copy(ctx.derivative_types)
            if derivative_type == "jvp" or derivative_type == "vjp":
                derivative_types[i] = "vjp"
                grads.append(
                    ChunkEvalBackward.apply(
                        ctx.dm,
                        ctx.mol,
                        ctx.grids,
                        ctx.feature_function,
                        derivative_types,
                        ctx.blksize,
                        ctx.compile_feature_function,
                        *ctx.vectors[:i],
                        grad_output,
                        *ctx.vectors[i + 1 :],
                    )
                )
            elif derivative_type == "first_vjp":
                grads.append(
                    ChunkEvalForward.apply(
                        ctx.dm,
                        ctx.mol,
                        ctx.grids,
                        ctx.feature_function,
                        ctx.blksize,
                        ctx.compile_feature_function,
                        *ctx.vectors[:i],
                        grad_output,
                        *ctx.vectors[i + 1 :],
                    )
                )
            else:
                raise ValueError(
                    f"Unknown derivative {derivative_type} (must be one of 'jvp', 'vjp', 'first_vjp')"
                )
        return tuple(grads)


def non_chunk(
    dm: torch.Tensor,
    mol: gto.Mole,
    grids: dft.Grids,
    feature_function: FeatureFunction,
    compile_feature_function: bool = False,
) -> torch.Tensor:
    ni = dft.numint.NumInt()
    ao = torch.from_dlpack(
        ni.eval_ao(mol, grids.coords, deriv=feature_function.deriv, non0tab=None)
    ).to(dm.device)
    if compile_feature_function:
        return torch.compile(feature_function.forward)(dm, ao.transpose(-1, -2))
    else:
        return feature_function(dm, ao.transpose(-1, -2))


def auto_chunk(
    dm: torch.Tensor,
    mol: gto.Mole,
    grids: dft.Grids,
    feature_function: FeatureFunction,
    block_size: int | None = None,
    max_memory_cpu: int = 2000,
    fix_block_size: bool = True,
    compile_feature_function: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Automatically splits feature evaluation into smaller chunks if needed.

    This function determines the appropriate chunk size for evaluating a feature
    function on molecular grids, based on available memory and number of basis
    functions. If the computed chunk size is larger than the size of the grid, or
    if a fixed block size was provided, it uses a non-chunked approach.

    Parameters
    ----------
    dm: torch.Tensor
        Density matrix or set of density matrices used for
        evaluating the feature function.
    mol: gto.Mole
        PySCF molecule object representing the system of interest.
    grids: dft.Grids
        Grids object defining the points in space on which
        the feature function is evaluated.
    feature_function: FeatureFunction
        The object representing the feature function to evaluate. The number of derivatives (deriv) determines
        how many components to compute.
    block_size: int | None, optional
        Manually specified block size for chunking.
        Defaults to None.
    max_memory_cpu: int, optional
        Maximum memory in MB to use for chunking (CPU only)
    fix_block_size: bool, optional
        Whether to fix the block size or compute it
        automatically based on system resources. Defaults to True.
    compile_feature_function: bool, optional
        If True, compiles the feature function for efficiency. Defaults to False.

    Returns
    -------
    dict[str, torch.Tensor]:
        The evaluated feature function on the specified grids, either
        computed in smaller chunks or in a single pass, depending on the block size.
    """

    if block_size is None and fix_block_size:
        comp = (
            (feature_function.deriv + 1)
            * (feature_function.deriv + 2)
            * (feature_function.deriv + 3)
            // 6
        )
        nao = mol.nao_nr()
        BLKSIZE = dft.gen_grid.BLKSIZE
        blksize = int(max_memory_cpu * 1e6 / ((comp + 1) * nao * 8 * BLKSIZE))
        blksize = max(4, min(blksize, 1200)) * BLKSIZE
    else:
        blksize = block_size  # type: ignore

    if blksize is not None:
        blksize = blksize - blksize % dft.gen_grid.BLKSIZE

    if blksize is not None and blksize >= grids.weights.shape[0]:
        features = non_chunk(
            dm.double(),
            mol,
            grids,
            feature_function,
            compile_feature_function=compile_feature_function,
        )
    else:
        features = ChunkEvalForward.apply(
            dm.double(),
            mol,
            grids,
            feature_function,
            blksize,
            compile_feature_function,
        )
    return feature_function.to_dict(features)
