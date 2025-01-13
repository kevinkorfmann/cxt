Tutorial for Inference
======================

A few examples of how to use the cxt package.

Constant Demography (Pairwise) Inference
--------------------------------------

The following code demonstrates the example usage for the base case of inferring
pairwise tmrca times of a sample size 50 (→ 1225 pairwise tmrca times) over a 1 MB
simulated fragment.

Loading a few functions for simulating the data and for inference and processing it.


.. code-block:: python

    import torch
    import numpy as np
    torch.set_float32_matmul_precision('medium')

    from cxt.config import TokenFreeDecoderConfig
    from cxt.utils import post_process, accumulating_mses, mse
    from cxt.inference import generate, load_model, prepare_ts_data
    from cxt.utils import simulate_parameterized_tree_sequence, TIMES


Model configuration and loading it. The was trained using Pytorch Lighting.

.. code-block:: python

    model = load_model(
        config=TokenFreeDecoderConfig(), 
        model_path='../cxt/models/base_model/checkpoints/epoch=4-step=16160.ckpt'
    )

Simulating the data.

.. code-block:: python

    SEED = 102000
    ts = simulate_parameterized_tree_sequence(SEED)
    src, tgt = prepare_ts_data(ts, num_samples=50, B=1225)

Running the actual inference, note that during each inference run the cache has
to be reset to zero. In order to avoid running unnessary replicates, a simple 
heuristic has been implemented to stop the inference when the derivative of the mse
is overall samples becomes less than less than 0.001.

.. code-block:: python

    max_replicates = 20
    yhats, ytrues = [], []
    for i in range(max_replicates):
        sequence = generate(model, src, B=1225)
        yhat, ytrue = post_process(tgt, sequence, TIMES)
        yhats.append(yhat)
        ytrues.append(ytrue)

        # early stopping criteria
        if i > 1:
            mses = accumulating_mses(yhats, ytrues)
            derivatives = np.diff(mses)
            if abs(derivatives[-1]) < 0.001:
                print(f"Stopping at {i} because derivative is {derivatives[-1]}.")
                break


** Plotting code not shown **

.. image:: ./heatmap_comparison.png
  :width: 800
  :alt: Alternative text


Out-of-sample: Sawooth Demography Inference (no fine-tuning)
------------------------------------------------------------

.. image:: ./inference_scatter_cxtkit_sawtooth_no_finetune.png
  :width: 800
  :alt: Alternative text

Rescue: Sawooth Demography Inference (with fine-tuning)
-------------------------------------------------------

.. image:: ./inference_scatter_cxtkit_sawtooth_with_finetune.png
  :width: 800
  :alt: Alternative text


Out-of-sample: Island Demography Inference (no fine-tuning)
-------------------------------------------------------------

.. image:: ./inference_scatter_cxtkit_island_no_finetune.png
  :width: 800
  :alt: Alternative text

"Rescue": Island Demography Inference (with fine-tuning)
-------------------------------------------------------------

Why mode-collapse?

.. image:: ./inference_scatter_cxtkit_island_with_finetune.png
  :width: 800
  :alt: Alternative text


