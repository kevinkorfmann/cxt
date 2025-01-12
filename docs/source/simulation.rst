Simulation
==========

``cxt`` is trained on simulated data. The following code demonstrates how to simulate data for training and inference.

Constant Demography Dataset (CDD):

.. code-block:: bash

    python simulate.py \\
        --num_processes 30 \\
        --num_samples 2_000_000 \\
        --batch_size 1000 \\
        --start_batch 0 \\
        --pivot_A 0 \\
        --pivot_B 1 \\
        --data_dir /sietch_colab/kkor/tiny_batches_base_dataset \\
        --scenario constant

Sawtooth Demography Dataset (SDD) for fine-tuning:

.. code-block:: bash

    python simulate.py \\
        --num_processes 30 \\
        --num_samples 200_000 \\
        --batch_size 1000 \\
        --start_batch 0 \\
        --pivot_A 0 \\
        --pivot_B 1 \\
        --data_dir /sietch_colab/kkor/tiny_batches_sawtooth_dataset \\
        --scenario sawtooth


