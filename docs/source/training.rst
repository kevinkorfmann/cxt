Training
========

``cxt`` is trained using the Pytorch Lightning framework. The following code demonstrates how to train the model.

.. code-block:: bash

    python train.py --dataset_path /sietch_colab/kkor/tiny_batches_base_dataset --gpus 0 1 2 --num_epochs 5


To fine-tune the model, use the following command for the sawtooth demography dataset (sdd):

.. code-block:: bash

    python train.py --dataset_path /sietch_colab/kkor/sdd \ 
        --gpus 0 1 \ 
        --num_epochs 2 \ 
        --learning_rate 3e-5 \
        --test_batches 20 \ 
        --checkpoint_path ./models/base_model/checkpoints/epoch=4-step=16160.ckpt


Or for the island demography dataset (idd):

.. code-block:: bash

    python train.py --dataset_path /sietch_colab/kkor/idd \
        --gpus 0 1 \
        --num_epochs 2 \
        --learning_rate 3e-5 \
        --test_batches 20 \
        --checkpoint_path ./models/base_model/checkpoints/epoch=4-step=16160.ckpt


LLM training, the directory llm contains main subdirectories for various scenarios, ranging from constant, to variable population sizes, including sweeps and island models.


.. code-block:: bash

    python train.py --dataset_path /sietch_colab/kkor/llm --gpus 0  --num_epochs 5

