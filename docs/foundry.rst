Using Skala on Azure AI Foundry
===============================

Azure AI Foundry is the fastest way to run Skala. On Foundry, you can easily deploy a Skala server that
runs full DFT computations with Skala on the GPU.


Deploying the model
-------------------

Find Skala in the `Azure AI Foundry Catalog <https://ai.azure.com/explore/models>`_ and create a deployment.
A deployment constitutes a server that can run Skala computations.
This deployment has a URL and an access token that you will need to run computations.


Example usage
-------------

The input geometry is provided in `QCSchema Molecule format <https://molssi-qc-schema.readthedocs.io/en/latest/auto_topology.html>`_, using element symbols and Cartesian coordinates in Bohr.
For providing the input geometry the `QCElemental package <https://molssi.github.io/QCElemental/>`_ can be used.
Additionally, the calculation can be configured to change the basis set, grid level and convergence criteria via the `SkalaConfig` class.

.. code-block:: python

    import qcelemental as qcel
    from skala.foundry import SkalaFoundryClient, SkalaConfig

    client = SkalaFoundryClient(endpoint="https://<your-endpoint>/score", credential="<your-access-token>")

    water = qcel.models.Molecule.from_data("""
    O  0.000000  0.000000  0.000000
    H  0.758602  0.000000  0.504284
    H -0.758602  0.000000  0.504284
    """, molecular_charge=0, molecular_multiplicity=1)

    config = SkalaConfig(basis="def2-tzvp", grid_level="superfine", max_num_scf_steps=80)

    status = client.run(water, config=config)

    if status.status == "succeeded":
        print("Total energy (Ha):", status.output.total_energy)
    else:
        raise RuntimeError(f"Task failed: {status.exception}")


Output format
-------------

The output of a successful computation is a `SkalaOutput` object, which looks as follows:

.. code-block:: python

    SkalaOutput(
        total_energy=-76.4117530597598, 
        energy_breakdown={
            'nuclear_repulsion_energy': 9.643579239174146,
            'scf_one_electron_energy': -124.0289768620489,
            'scf_two_electron_energy': 47.334283713531015,
            'scf_xc_energy': -9.360065488748322,
            'dftd3_dispersion_energy': -0.0005736616677424679
        },
        num_scf_iterations=19,
        dipole_moment=[1.3522570910077873e-08, 1.30533265138765e-10, 0.760798338213357]
    )

All values are provided in atomic units, i.e. energy is Hartree and dipole moments in electron Bohr.

Mode of operation
-----------------

The Foundry server is always running and ready to accept jobs. It internally keeps a queue of tasks and works through them one by one.
When you call the ``run`` method, the `SkalaFoundryClient` adds a task to the queue and waits for it to complete.
When you kill the computation (e.g. via Ctrl-C), the client will send a cancellation request to the server to avoid wasting compute resources.
