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
    from cxt.utils import post_process, decreasing_mses
    from cxt.inference import generate, load_model, prepare_ts_data
    from cxt.utils import simulate_parameterized_tree_sequence, TIMES

    device = 'cuda'
    torch.set_float32_matmul_precision('medium')
    mse = lambda yhat, ytrue: ((yhat - ytrue)**2).mean()


Model configuration and loading it. The was trained using Pytorch Lighting.

.. code-block:: python

    B = 1225
    class TokenFreeDecoderConfig:
        num_samples: int = 50
        sample_scale_embd: int = 2 
        output_dim: int = 256+2
        n_embd: int = 400
        combined_dim: int = 1001
        n_layer: int = 6
        bias: bool = False
        dropout: float = 0.1
        n_head: int = 4
        device: str = 'cuda'
        batch_size: int = B
        

    model_path = '../cxt/lightning_logs/version_8/checkpoints/epoch=4-step=16160.ckpt'
    config = TokenFreeDecoderConfig()
    model = load_model(config, model_path)


Simulating the data.

.. code-block:: python

    SEED = 102000
    ts = simulate_parameterized_tree_sequence(SEED, sequence_length=1e6)
    src, tgt = prepare_ts_data(ts, num_samples=50, B=1225)

Running the actual inference, note that during each inference run the cache has
to be reset to zero. In order to avoid running unnessary replicates, a simple 
heuristic has been implemented to stop the inference when the derivative of the mse
is overall samples becomes less than less than 0.001.

.. code-block:: python

    num_layers = 6
    max_replicates = 20

    yhats, ytrues = [], []
    src = src.to(device).to(torch.float32)
    mses = []
    for i in range(max_replicates):

        # reset cache
        for j in range(num_layers):
            model.transformer.h[j].attn.cache_k *= 0. 
            model.transformer.h[j].attn.cache_v *= 0. 

        sequence = generate(model, src, B=B, device=device)
        yhat, ytrue = post_process(tgt, sequence, TIMES)
        mses.append(mse(yhat, ytrue))
        yhats.append(yhat)
        ytrues.append(ytrue)

        # early stopping criteria
        if i > 1:
            mses = decreasing_mses(yhats, ytrues)
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


Rescue: Sawooth Demography Inference (with fine-tuning)
-------------------------------------------------------


Out-of-sample: Island Demography Inference (no fine-tuning)
-------------------------------------------------------------



