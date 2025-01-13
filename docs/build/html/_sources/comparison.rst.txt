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
    from cxt.utils import create_sawtooth_demogaphy_object
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
        singer_master -vcf ${scenario} -output ${scenario} -m 1.29e-8 -n 100 -thin 20 -start 0 -end 1000000 -Ne 20000 -polar 0.99 -fast
        convert_to_tskit -input ${scenario}_fast -output ${scenario} -start 0 -end 1000000 
    done

.. code-block:: python

    import tskit
    import numpy as np
    from tqdm import tqdm
    from itertools import combinations
    from cxt.utils import interpolate_tmrcas
    from multiprocessing import Pool, cpu_count
    from cxt.plotting import plot_inference_scatter

    SEED = 103370001
    SCENARIO = ""
    WINDOW_SIZE = 2000
    NUM_PROCESSES = 100  

    def process_file(file_idx):
        file = f"ts_seed_{SEED}{SCENARIO}_{file_idx}.trees"
        ts = tskit.load(file)
        combs = combinations(range(50), 2)
        tmrcas_comb = []
        for a, b in combs:
            ts_simple = ts.simplify(samples=[a, b])
            tmrcas_comb.append(interpolate_tmrcas(ts_simple, window_size=WINDOW_SIZE))
        return tmrcas_comb

    num_files = 100
    with Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(process_file, range(num_files)), total=num_files))
    tmrcas = results  
    yhats_singer = np.array(tmrcas).mean(0)


    from cxt.utils import simulate_parameterized_tree_sequence
    ts = simulate_parameterized_tree_sequence(SEED)
    combs = list(combinations(range(50), 2))
    tmrcas = []
    for a,b in tqdm(combs):
        ts_simple = ts.simplify(samples=[a,b])
        tmrcas.append(interpolate_tmrcas(ts_simple, window_size=2000))
    ytrues = np.array(tmrcas)

    yhats_singer_log = np.log(yhats_singer)
    ytrues_log = np.log(ytrues)

    plot_inference_scatter(
        yhats_singer_log, ytrues_log,
        "inference_scatter_singer_constant.png",
        subtitle="Constant Demography ",
        tool=r'$\mathbf{singer}$', stackit=False
        )


.. image:: ./inference_scatter_singer_constant.png
  :width: 400
  :alt: Constant demography inference 

.. image:: ./inference_scatter_singer_sawtooth.png
  :width: 400
  :alt: Sawooth demography inference 

**For some unknown reason, the island demography inference failed to using Singer for now.**

Gamma-SMC
---------

Gamma-SMC is method developed by Schweiger and Durbin (2023) and accesible here: https://github.com/regevs/gamma_smc/tree/main 

An example command for running it is shown below:

.. code-block:: bash
    
    singularity run -B /home/kkor/cxt/nbs/gamma-smc:/mnt docker://docker.io/regevsch/gamma_smc:v0.2  -i /mnt/ts_seed_103370001.vcf  -o /mnt/ts_seed_103370001.zst -t 1

And TMRCAs can be extracted using the following command:

.. code-block:: python

    alphas, betas, meta = open_posteriors("ts_seed_103370001.zst")
    tmrca_gamma_smc = alphas / betas
    tmrca_gamma_smc = np.log(tmrca_gamma_smc*2*20_000)
    tmrca_gamma_smc = np.array(tmrca_gamma_smc)


.. image:: ./inference_scatter_gamma_smc_constant.png
  :width: 400
  :alt: Constant demography inference 

.. image:: ./inference_scatter_gamma_smc_sawtooth.png
  :width: 400
  :alt: Sawooth demography inference 

.. image:: ./inference_scatter_gamma_smc_island.png
  :width: 400
  :alt: Island demography inference 