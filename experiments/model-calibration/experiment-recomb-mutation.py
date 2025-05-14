import numpy as np
import os
import stdpopsim
import pickle
import ray
import argparse
import matplotlib.pyplot as plt
import torch
import pickle

from matplotlib.colors import Normalize
from torch import Tensor
from cxt.config import BroadModelConfig 
import cxt.utils as utils
import cxt.inference as inference

TokenFreeDecoderConfig = BroadModelConfig

parser = argparse.ArgumentParser(
    "Calculate bias in cxt along a grid of recombination and mutation rates "
    "for a fixed demographic model. Do a first order correction by adjusting "
    "TMRCA distribution to have a mean that equals site diversity given "
    "mutation rates."
)
parser.add_argument(
    "--random-seed", help="Global random seed", 
    type=int, default=1,
)
parser.add_argument(
    "--num-cpus", help="Number of parallel processes", 
    type=int, default=50,
)
parser.add_argument(
    "--grid-size", help="Size of parameter grid in one dimension", 
    type=int, default=35,
)
parser.add_argument(
    "--num-reps", help="Number of reps per grid cell", 
    type=int, default=30,
)
parser.add_argument(
    "--demography", help="Demographic model", 
    type=str, default="Zigzag_1S14",
)
parser.add_argument(
    "--output-path", help="Where to save plots", type=str, 
    default="/home/natep/public_html/cxt/experiment-recomb-mutation/",
)
parser.add_argument(
    "--model-path", help="Lightning checkpoint to load model from", type=str,
    default="/sietch_colab/data_share/cxt/broad_model/epoch=0-step=11296.ckpt",
)
parser.add_argument(
    "--overwrite-cache", action="store_true",
    help="Overwrite saved results",
)
parser.add_argument(
    "--device", help="Device to use for tensor computations", type=str,
    default="cuda:0",
)
args = parser.parse_args()

if not os.path.exists(args.output_path): 
    os.makedirs(args.output_path)

demography = args.demography
rng = np.random.default_rng(args.random_seed)
ray.init(num_cpus=args.num_cpus)
torch.manual_seed(1024)

# get contig from which to base parameter values around
num_diploids = 25
species = stdpopsim.get_species("HomSap")
base_contig = species.get_contig("chr1")
fold_change = np.linspace(-1, 1, args.grid_size + 1)
base_m = base_contig.mutation_rate
base_r = base_contig.recombination_map.mean_rate
m_grid = 2 ** fold_change * base_m
r_grid = 2 ** fold_change * base_r
parameter_grid = np.array([
    (m, r) 
    for m in (m_grid[1:] + m_grid[:-1]) / 2 
    for r in (r_grid[1:] + r_grid[:-1]) / 2
])

@ray.remote
def simulate_parallel(seed, m, r):
    demogr = species.get_demographic_model(demography)
    contig = species.get_contig(length=1e6, mutation_rate=m, recombination_rate=r)
    sample = {demogr.populations[0].name: num_diploids}
    engine = stdpopsim.get_engine("msprime")
    div = 0.0
    attempts = 0
    while div == 0.0:  # rejection sample
        ts = engine.simulate(
            contig=contig, 
            samples=sample, 
            demographic_model=demogr, 
            seed=seed + attempts,
        )
        pivot = [0, 1]
        ts_pivot = ts.simplify(samples=pivot) 
        div = ts_pivot.diversity()
        attempts += 1
    summary_stats = [
        ts_pivot.diversity(), 
        np.mean(np.diff(ts_pivot.breakpoints(as_array=True))),
    ]
    args = (ts, 0, 1, 0.0, False)   # (TreeSequence, pivot_A, pivot_B, offset, ignore_target)
    observed, target = utils.process_pair(args) 
    return observed, target, summary_stats

model_path = args.model_path
device = args.device
config = TokenFreeDecoderConfig(device=device)
model = inference.load_model(
    config=config,
    model_path=model_path,
    device=device,
)

cache_file = f"{args.output_path}/cache.pkl"
if not os.path.exists(cache_file) or args.overwrite_cache:
    statistics = np.zeros((parameter_grid.shape[0], 2))
    bias = np.zeros(parameter_grid.shape[0])
    bias_corr = np.zeros(parameter_grid.shape[0])
    rmse = np.zeros(parameter_grid.shape[0])
    rmse_corr = np.zeros(parameter_grid.shape[0])

    ytrue_store = []
    ypred_store = []
    ycorr_store = []
    stats_store = []
    
    for rep in range(args.num_reps):
        print(f"Running rep {rep}", flush=True)
        seed_array = rng.integers(2 ** 32 - 1, size=parameter_grid.shape[0])
        job_list = [
            simulate_parallel.remote(s, m, r) 
            for s, (m, r) in zip(seed_array, parameter_grid)
        ]
        src, tgt, stats = zip(*ray.get(job_list))
        src = np.stack(src)
        tgt = np.stack(tgt)
        stats = np.stack(stats)
    
        sequence = inference.generate(model, Tensor(src).to(device), B=src.shape[0], device=device)
        ypred, ytrue = inference.post_process(Tensor(tgt).to(torch.int32), sequence, utils.TIMES)
    
        # correction
        mut_rate = parameter_grid[:, 0]  # nonconstant across grid
        obsv_div = stats[:, 0]
        true_div = np.mean(2 * np.exp(ytrue), axis=1) * mut_rate
        pred_div = np.mean(2 * np.exp(ypred), axis=1) * mut_rate
        assert np.all(obsv_div > 0)
        assert np.all(pred_div > 0)
        ycorr = ypred + (np.log(obsv_div) - np.log(pred_div))[:, None]
    
        bias += np.mean(ypred - ytrue, axis=1)
        bias_corr += np.mean(ycorr - ytrue, axis=1)
        rmse += np.sqrt(np.mean((ypred - ytrue) ** 2, axis=1))
        rmse_corr += np.sqrt(np.mean((ycorr - ytrue) ** 2, axis=1))
        statistics += stats

        ytrue_store.append(ytrue)
        ypred_store.append(ypred)
        ycorr_store.append(ycorr)
        stats_store.append(stats)
    
    statistics /= args.num_reps
    bias_corr /= args.num_reps
    rmse_corr /= args.num_reps
    bias /= args.num_reps
    rmse /= args.num_reps
    cache = {
        # per-rep
        "ytrue_store": ytrue_store,
        "ypred_store": ypred_store,
        "ycorr_store": ycorr_store,
        "stats_store": stats_store,
        # averages
        "statistics": statistics,
        "bias_corr": bias_corr,
        "rmse_corr": rmse_corr,
        "bias": bias,
        "rmse": rmse,
    }
    pickle.dump(cache, open(cache_file, "wb"))
else:
    cache = pickle.load(open(cache_file, "rb"))
    statistics = cache["statistics"]
    bias_corr = cache["bias_corr"]
    bias = cache["bias"]
    rmse_corr = cache["rmse_corr"]
    rmse = cache["rmse"]

# plot error and rmse along grid
fig, axs = plt.subplots(
    2, 2, figsize=(10, 8), sharex=True, sharey=True, 
    constrained_layout=True, squeeze=False,
)
# bias
axs[0, 0].set_title("Bias")
img = axs[0, 0].pcolormesh(
    m_grid, r_grid,
    bias.reshape(args.grid_size, args.grid_size).T,
    cmap=plt.get_cmap("seismic"),
    norm=Normalize(vmin=-1, vmax=1),
)
axs[0, 0].plot(base_m, base_r, "o", markersize=4, c="green")
axs[0, 0].set_xscale('log', base=2)
axs[0, 0].set_yscale('log', base=2)
plt.colorbar(img, ax=axs[0, 0], label="bias (logspace)")
img = axs[1, 0].pcolormesh(
    m_grid, r_grid,
    bias_corr.reshape(args.grid_size, args.grid_size).T,
    cmap=plt.get_cmap("seismic"),
    norm=Normalize(vmin=-1, vmax=1),
)
axs[1, 0].plot(base_m, base_r, "o", markersize=4, c="green")
axs[1, 0].set_xscale('log', base=2)
axs[1, 0].set_yscale('log', base=2)
plt.colorbar(img, ax=axs[1, 0], label="bias after correction (logspace)")
# RMSE
axs[0, 1].set_title("RMSE")
img = axs[0, 1].pcolormesh(
    m_grid, r_grid,
    rmse.reshape(args.grid_size, args.grid_size).T,
    cmap=plt.get_cmap("magma"),
    norm=Normalize(vmin=rmse.min(), vmax=rmse.max()),
)
axs[0, 1].plot(base_m, base_r, "o", markersize=4, c="green")
axs[0, 1].set_xscale('log', base=2)
axs[0, 1].set_yscale('log', base=2)
plt.colorbar(img, ax=axs[0, 1], label="RMSE (logspace)")
img = axs[1, 1].pcolormesh(
    m_grid, r_grid,
    rmse_corr.reshape(args.grid_size, args.grid_size).T,
    cmap=plt.get_cmap("magma"),
    norm=Normalize(vmin=rmse.min(), vmax=rmse.max()),
)
axs[1, 1].plot(base_m, base_r, "o", markersize=4, c="green")
axs[1, 1].set_xscale('log', base=2)
axs[1, 1].set_yscale('log', base=2)
plt.colorbar(img, ax=axs[1, 1], label="RMSE after correction (log)")
fig.supxlabel("Mutation rate")
fig.supylabel("Recombination rate")
plt.savefig(f"{args.output_path}/bias-and-rmse.png")
plt.clf()

# plot a sanity check wrt parameter grid
fig, axs = plt.subplots(
    2, 1, figsize=(5, 8), sharex=True, sharey=True, 
    constrained_layout=True, squeeze=False,
)
img = axs[0, 0].pcolormesh(
    m_grid, r_grid,
    statistics[:, 0].reshape(args.grid_size, args.grid_size).T,
    cmap=plt.get_cmap("plasma"),
)
axs[0, 0].set_xscale('log', base=2)
axs[0, 0].set_yscale('log', base=2)
axs[0, 0].plot(base_m, base_r, "o", markersize=4, c="green")
plt.colorbar(img, ax=axs[0, 0], label="Mean nucleotide diversity")
img = axs[1, 0].pcolormesh(
    m_grid, r_grid,
    statistics[:, 1].reshape(args.grid_size, args.grid_size).T,
    cmap=plt.get_cmap("plasma"),
    norm=Normalize(
        vmin=statistics[:, 1].min(), 
        vmax=np.quantile(statistics[:, 1], 0.95).item(),
    ),
)
axs[1, 0].set_xscale('log', base=2)
axs[1, 0].set_yscale('log', base=2)
axs[1, 0].plot(base_m, base_r, "o", markersize=4, c="green")
plt.colorbar(img, ax=axs[1, 0], label="Mean breakpoint length")
fig.suptitle("Summary stats sanity check")
fig.supxlabel("Mutation rate")
fig.supylabel("Recombination rate")
plt.savefig(f"{args.output_path}/summary-stats.png")
plt.clf()

