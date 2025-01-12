Comparisons with other tools
=============================


Simulation of different scenarios for testing the performance of the methods against cxt(kit)s' language model.

The data simulation involves generating tree sequences under various demographic models.The first
scenario uses a basic parameterized tree sequence with a fixed seed under constant demography. The
second scenario simulates a sawtooth demographic model with periodic population size changes. The
third scenario employs an island model with three populations and migration between them. 

.. code-block:: python

    from cxt.utils import simulate_parameterized_tree_sequence
    SEED = 103370001
    ts = simulate_parameterized_tree_sequence(SEED)
    with open('./ts_seed_' + str(SEED) + '.vcf', 'w') as f:
        f.write(ts.as_vcf())

.. code-block:: python

    from functools import partial
    simulate_parameterized_tree_sequence_sawtooth = partial(simulate_parameterized_tree_sequence,
        demography=create_sawtooth_demogaphy_object(Ne=20e3, magnitue=3))
    ts = simulate_parameterized_tree_sequence_sawtooth(SEED)
    with open('./ts_seed_' + str(SEED) + '_sawtooth.vcf', 'w') as f:
        f.write(ts.as_vcf())

.. code-block:: python

    samples = {0: 15, 1: 5, 2: 5}
    island_demography = msprime.Demography.island_model([10000, 5000, 5000], migration_rate=0.1)
    simulate_parameterized_tree_sequence_island = partial(simulate_parameterized_tree_sequence, island_demography=island_demography, samples=samples)
    ts = simulate_parameterized_tree_sequence_island(SEED)
    with open('./ts_seed_' + str(SEED) + '_island.vcf', 'w') as f:
        f.write(ts.as_vcf())


Singer
------

Singer is method developed by Deng et al. (2024) and accesible here: https://github.com/popgenmethods/SINGER


.. code-block:: bash

    for scenario in ts_seed_103370001 ts_seed_103370001_sawtooth ts_seed_103370001_island ; do
        SINGER/releases/singer_master -vcf ${scenario} -output ${scenario} -m 1.29e-8 -n 100 -thin 20 -start 0 -end 1000000 -Ne 20000 -polar 0.99 -fast
        SINGER/releases/convert_to_tskit -input ${scenario}_fast -output ${scenario} -start 0 -end 1000000 
    done




Gamma-SMC
---------