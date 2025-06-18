Simulation
==========

``cxt`` is trained on simulated data. The following code demonstrates how to simulate data for training and inference.

Constant Demography Dataset (cdd):

.. code-block:: bash

    python simulation.py \
        --num_processes 30 \
        --num_samples 2_000_000 \
        --batch_size 1000 \
        --start_batch 0 \
        --pivot_A 0 \
        --pivot_B 1 \
        --data_dir /sietch_colab/kkor/base_dataset \
        --scenario constant

Sawtooth Demography Dataset (sdd) for fine-tuning:

.. code-block:: bash

    python simulation.py \
        --num_processes 30 \
        --num_samples 200_000 \
        --batch_size 1000 \
        --start_batch 0 \
        --pivot_A 0 \
        --pivot_B 1 \
        --data_dir /sietch_colab/kkor/ssd \
        --scenario sawtooth

Island Demography Dataset (idd) for fine-tuning:

.. code-block:: bash

    python simulation.py \
        --num_processes 30 \
        --num_samples 200_000 \
        --batch_size 1000 \
        --start_batch 0 \
        --pivot_A 0 \
        --pivot_B 1 \
        --data_dir /sietch_colab/kkor/idd \
        --scenario island \
        --randomize_pivots True


