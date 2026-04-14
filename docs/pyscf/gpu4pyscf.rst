Using Skala with gpu4pyscf
==========================

The Skala functional can also be used in GPU4PySCF with an appropriate PyTorch CUDA version by creating a new Kohn-Sham calculator based on the `SkalaKS` constructor from the ``skala.gpu4pyscf`` module.

.. code-block:: python

   from pyscf import gto

   from skala.gpu4pyscf import SkalaKS

   mol = gto.M(
       atom="""H 0 0 0; H 0 0 1.4""",
       basis="def2-tzvp",
   )
   ks = SkalaKS(mol, xc="skala")
   ks.kernel()

   print(ks.dump_scf_summary())


Installation
------------

Install the latest version of the ``skala`` package from PyPI or conda-forge together with a compatible PyTorch CUDA version.
For pip the default PyTorch installation will be used, which is typically the latest version with CUDA support.

.. code-block:: bash

   pip install skala cupy-cuda12x cutensor-cuda12x

For conda-forge, select the pytorch CUDA version that matches your system and CUDA installation. For example, for CUDA 12.8:

.. code-block:: bash

   mamba install -c conda-forge skala 'cuda-version=12.*' 'pytorch=*=cuda*' cupy cutensor

In both cases you need to install GPU4PySCF separately from PyPI.

.. code-block:: bash

   pip install --no-deps "gpu4pyscf-cuda12x>=1.0,<2" "gpu4pyscf-libxc-cuda12x>=0.4,<1"

We are using ``--no-deps`` to avoid overriding the already installed cupy and cutensor packages in the previous step.