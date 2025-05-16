import stdpopsim
import os
import numpy as np
import pickle
import argparse

import matplotlib.pyplot as plt

from cxt.inference import translate_from_multi_ts_multi_gpu
from cxt.config import BroadModelConfig 
from cxt.utils import diversity_bias_correction
from cxt.utils import diversity_bias_correction_by_rep

from model_rescaling import scale_model, scale_growth_rates

TokenFreeDecoderConfig = BroadModelConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutation-scale", type=float, help="Scale mutation rate by this", default=1.0)
    parser.add_argument("--recombination-scale", type=float, help="Scale recombination rate by this", default=1.0)
    parser.add_argument("--growth-rate-scale", type=float, help="Scale exponential growth rates by this", default=1.0)
    parser.add_argument("--coal-unit-scale", type=float, help="Scale coalescent units by this", default=1.0)
    parser.add_argument("--overwrite-cache", action="store_true", help="Recalculate cached results")
    parser.add_argument("--correct-by-rep", action="store_true", help="Do correction rep by rep rather than on rep average")
    parser.add_argument("--outpath", type=str, help="Output plot", default="correction-bias-and-rmse.png")
    args = parser.parse_args()
    
    num_pairs = 25
    species = stdpopsim.get_species("HomSap")
    demogr = species.get_demographic_model("Zigzag_1S14")
    demogr = scale_model(scale_growth_rates(demogr, args.growth_rate_scale), args.coal_unit_scale)
    contig = species.get_contig("chr1")
    contig = species.get_contig(
        length=1e6, 
        mutation_rate=contig.mutation_rate * args.mutation_scale, 
        recombination_rate=contig.recombination_map.mean_rate * args.recombination_scale,
    )
    sample = {"generic" : num_pairs}
    engine = stdpopsim.get_engine("msprime")
    
    config = TokenFreeDecoderConfig()
    model_path = "/sietch_colab/data_share/cxt/broad_model/epoch=0-step=11296.ckpt"
    max_replicates = 15
    
    cache_file = "cache.pkl"
    if not os.path.exists(cache_file) or args.overwrite_cache:
        ts = engine.simulate(contig=contig, samples=sample, demographic_model=demogr, seed=1).trim()
        yhats, ytrues = translate_from_multi_ts_multi_gpu(
            ts_list = [ts],
            max_replicates=max_replicates,
            model_config=config,
            model_path=model_path,
            devices=['cuda:0', 'cuda:1', 'cuda:2'],
        )
        pickle.dump(
            {"yhats": yhats, "ytrues": ytrues, "ts": ts},
            open(cache_file, "wb"),
        )
    else:
        cache = pickle.load(open(cache_file, "rb"))
        ts = cache["ts"]
        yhats = cache["yhats"]
        ytrues = cache["ytrues"]
    
    pivot_pairs = np.array([
        (i, j) 
        for i in range(ts.num_samples) 
        for j in range(i + 1, ts.num_samples)
    ])
    correction = diversity_bias_correction_by_rep if args.correct_by_rep else \
        diversity_bias_correction
    corrected_yhats_pool, corrected_baselines_pool = \
        correction(
            tree_sequence=ts,
            mutation_rate=contig.mutation_rate,
            predictions=yhats,
            pivot_pairs=pivot_pairs,
            return_intercept=True,
        )
    corrected_yhats_pool = corrected_yhats_pool.mean(axis=0) # average over reps
    rmse = np.sqrt(np.mean(np.power(yhats - ytrues, 2), axis=-1))
    corrected_rmse_pool = np.sqrt(np.mean(np.power(corrected_yhats_pool - ytrues, 2), axis=-1))
    corrected_rmse_baseline = np.sqrt(np.mean(np.power(corrected_baselines_pool - ytrues, 2), axis=-1))
    bias = np.mean(yhats - ytrues, axis=-1)
    corrected_bias_pool = np.mean(corrected_yhats_pool - ytrues, axis=-1)
    corrected_bias_baseline = np.mean(corrected_baselines_pool - ytrues, axis=-1)
    
    rmse_order = np.argsort(np.mean(rmse, axis=0))
    corrected_rmse_pool_order = np.argsort(np.mean(corrected_rmse_pool, axis=0))
    corrected_rmse_baseline_order = np.argsort(np.mean(corrected_rmse_baseline, axis=0))
    bias_order = np.argsort(np.mean(bias, axis=0))
    corrected_bias_pool_order = np.argsort(np.mean(corrected_bias_pool, axis=0))
    corrected_bias_baseline_order = np.argsort(np.mean(corrected_bias_baseline, axis=0))
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True, squeeze=False, sharex=True)
    # rmse
    axs[0,0].plot(
        np.arange(pivot_pairs.shape[0]), 
        np.mean(rmse[:, rmse_order], axis=0), 
        "o", 
        color="black",
        markersize=1,
        label="cxt-only",
    )
    axs[0,0].plot(
        np.arange(pivot_pairs.shape[0]), 
        np.mean(corrected_rmse_baseline[:, corrected_rmse_baseline_order], axis=0), 
        "o", 
        color="blue", 
        markersize=1,
        label="correction-only",
    )
    axs[0,0].plot(
        np.arange(pivot_pairs.shape[0]), 
        np.mean(corrected_rmse_pool[:, corrected_rmse_pool_order], axis=0), 
        "o", 
        color="red", 
        markersize=1,
        label="cxt+correction",
    )
    axs[0,0].legend()
    axs[0,0].set_xticklabels([])
    axs[0,0].set_ylabel("RMSE")
    axs[0,0].set_title(
        f"mut-scale: {args.mutation_scale}, "
        f"rec-scale: {args.recombination_scale}, "
        f"coal-scale: {args.coal_unit_scale}, "
        f"growth-scale: {args.growth_rate_scale}"
    )
    # bias
    axs[1,0].plot(
        np.arange(pivot_pairs.shape[0]), 
        np.mean(bias[:, bias_order], axis=0), 
        "o", 
        color="black",
        markersize=1,
        label="cxt-only",
    )
    axs[1,0].plot(
        np.arange(pivot_pairs.shape[0]), 
        np.mean(corrected_bias_baseline[:, corrected_bias_baseline_order], axis=0), 
        "o", 
        color="blue", 
        markersize=1,
        label="correction-only",
    )
    axs[1,0].plot(
        np.arange(pivot_pairs.shape[0]), 
        np.mean(corrected_bias_pool[:, corrected_bias_pool_order], axis=0), 
        "o", 
        color="red", 
        markersize=1,
        label="cxt+correction",
    )
    axs[1,0].axhline(y=0.0, linestyle="dashed", color="black")
    axs[1,0].legend()
    axs[1,0].set_xticklabels([])
    axs[1,0].set_ylabel("Bias")
    fig.supxlabel("Pivot pair (sorted)")
    plt.savefig(f"{args.outpath}")


