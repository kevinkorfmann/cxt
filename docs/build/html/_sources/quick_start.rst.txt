Quick start
===========

From simulation to inference in a few lines of code.

First, we simulate a tree sequence with stdpopsim and msprime. Here we simulate 25 diploid individuals from the Zigzag_1S14 demographic model of humans, for the first 2Mb of chromosome 1.
We hardcoded the requirement of 50 samples, thus we specify 25 generic diploid individuals, which is a minimum requirement for the inference to work. Larger sample sizes need to be downsampled accordingly.

.. code-block:: python

    import stdpopsim
    species = stdpopsim.get_species("HomSap")
    demogr = species.get_demographic_model("Zigzag_1S14")
    sample = {"generic" : 25}
    engine = stdpopsim.get_engine("msprime")
    contig = species.get_contig("chr1", right=2e6)
    ts = engine.simulate(contig=contig, samples=sample, demographic_model=demogr, seed=42).trim()
    with open('example.vcf', 'a') as f:
        f.write(ts.as_vcf())

Then, we can infer pairwise coalescence times using the `translate` function. Here we infer the pairwise coalescence times of sample 0 and 1, using 15 replicates. Since 50 samples are the requirement, the pivot_combinations argument takes a list of tuples, where each tuple specifies the indices of the two samples to infer the pairwise coalescence times for.

.. code-block:: python

    from cxt.api import translate
    tmrca = translate("example.vcf", num_replicates=15, pivot_combinations=[(0, 1)])


If a mutation rate is provided, we can can apply a correction, which is recommended for real data to deal with scaling offsets.

.. code-block:: python

    from cxt.api import translate
    mutation_rate = 1.29e-8
    tmrca = translate("example.vcf", num_replicates=15, mutation_rate=mutation_rate, pivot_combinations=[(0, 1)])


The output is of shape (1MB segments, num_replicates, num_pivot_combinations, windows), e.g. (2, 15, 1, 500) for the example above, where 500 is the number of 2kb windows in 1MB. Usually the mean over the 15 replicates is taken for downstream analysis.

.. image:: ./figures/tmrca_example.png
    :alt: Inference of pairwise coalescence times of sample 0 and 1.
    :align: center