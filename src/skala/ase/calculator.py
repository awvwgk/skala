# SPDX-License-Identifier: MIT

from ase.atoms import Atoms
from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    InputError,
    Parameters,
    all_changes,
)
from ase.units import Bohr, Debye, Hartree
from pyscf import grad, gto

from skala.pyscf import SkalaKS


class Skala(Calculator):
    """
    ASE calculator for the Skala exchange-correlation functional.

    This calculator integrates the Skala functional into ASE, allowing
    for efficient density functional theory calculations using the Skala
    neural network-based exchange-correlation functional.
    """

    atoms: Atoms | None = None
    """Atoms object associated with the calculator."""

    implemented_properties = [
        "energy",
        "forces",
        "dipole",
    ]

    default_parameters = {
        "xc": "skala",
        "basis": None,
        "with_density_fit": False,
        "with_newton": False,
        "with_dftd3": True,
        "charge": None,
        "multiplicity": None,
        "verbose": 0,
    }

    _mol: gto.Mole | None = None
    _ks: grad.rhf.GradientsBase | None = None

    def __init__(self, atoms: Atoms | None = None, **kwargs):
        super().__init__(atoms=atoms, **kwargs)

    def set(self, **kwargs) -> dict:
        """
        Set parameters for the Skala calculator.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to set for the calculator.
        """
        changed_parameters = super().set(**kwargs)
        if "verbose" in changed_parameters:
            if self._mol is not None:
                self._mol.verbose = self.parameters.verbose
            if self._ks is not None:
                self._ks.verbose = self.parameters.verbose
                self._ks.base.verbose = self.parameters.verbose

        if (
            "charge" in changed_parameters
            or "multiplicity" in changed_parameters
            or "basis" in changed_parameters
        ):
            self._mol = None
            self._ks = None
            self.reset()

        if (
            "xc" in changed_parameters
            or "with_density_fit" in changed_parameters
            or "with_newton" in changed_parameters
            or "with_dftd3" in changed_parameters
        ):
            self._ks = None
            self.reset()

        return changed_parameters

    def reset(self) -> None:
        """
        Reset the calculator to its initial state.
        """
        super().reset()

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        """
        Perform the calculation for the given atoms.

        Parameters
        ----------
        atoms : Atoms, optional
            The atoms object to calculate properties for.
        properties : list of str, optional
            List of properties to calculate.
        system_changes : list of str, optional
            List of changes in the system that trigger recalculation.
        """
        if not properties:
            properties = ["energy"]
        if system_changes is None:
            system_changes = all_changes

        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        if self.parameters.basis is None:
            raise InputError("Basis set must be specified in the parameters.")

        if self.atoms.pbc.any():  # type: ignore
            raise CalculatorError(
                "Skala functional does not support periodic boundary conditions (PBC) yet."
            )

        atom = [(atom.symbol, atom.position) for atom in self.atoms]  # type: ignore
        if set(system_changes) - {"positions"}:
            self._mol = None
            self._ks = None

        if self._mol is None:
            self._mol = gto.M(
                atom=atom,
                basis=self.parameters.basis,
                unit="Angstrom",
                verbose=self.parameters.verbose,
                charge=_get_charge(self.atoms, self.parameters),
                spin=_get_uhf(self.atoms, self.parameters),
            )
            self._ks = None
        else:
            self._mol = self._mol.set_geom_(atom, inplace=False)

        if self._ks is None:
            self._ks = SkalaKS(
                self._mol,
                xc=self.parameters.xc,
                with_density_fit=self.parameters.with_density_fit,
                with_newton=self.parameters.with_newton,
                with_dftd3=self.parameters.with_dftd3,
            ).nuc_grad_method()
        else:
            self._ks.reset(self._mol)

        energy = self._ks.base.kernel()
        gradient = self._ks.kernel()

        self.results["energy"] = energy * Hartree
        self.results["dipole"] = (
            self._ks.base.dip_moment(unit="debye", verbose=self._mol.verbose) * Debye
        )
        self.results["forces"] = -gradient * Hartree / Bohr


def _get_charge(atoms: Atoms, parameters: Parameters) -> int:
    """
    Get the total charge of the system.
    If no charge is provided, the total charge of the system is calculated
    by summing the initial charges of all atoms.
    """
    return (
        atoms.get_initial_charges().sum()
        if parameters.charge is None
        else parameters.charge
    )


def _get_uhf(atoms: Atoms, parameters: Parameters) -> int:
    """
    Get the number of unpaired electrons.
    If no multiplicity is provided, the number of unpaired electrons
    is calculated by summing the initial magnetic moments of all atoms.
    """
    return (
        int(atoms.get_initial_magnetic_moments().sum().round())
        if parameters.multiplicity is None
        else parameters.multiplicity - 1
    )
