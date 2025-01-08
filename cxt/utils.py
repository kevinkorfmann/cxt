import msprime
import tskit
from typing import List, Tuple
from scipy.interpolate import interp1d
import numpy as np
import torch

retrieve_site_positions = lambda ts: np.array([site.position for site in ts.sites()])
xor = lambda a, b: (a^b).astype(int)
xnor = lambda a, b: (1 - xor(a, b)).astype(int)
mse = lambda yhat, ytrue: ((yhat - ytrue)**2).mean()

def discretize(sequence, population_time):
    indices = np.searchsorted(population_time, sequence, side="right") - 1
    indices = np.clip(indices, 0, len(population_time) - 1)
    return indices.tolist()

def simulate_parameterized_tree_sequence(
        seed,
        samples=25,
        population_size=2e4,
        sequence_length=1e6,
        recombination_rate=1.28e-8,
        ploidy=2,
        mutation_rate=1.29e-8,
        demography=None
        ):
    np.random.seed(seed)
    SEED = np.random.uniform(1, 2**32 - 1)
    if demography:
        ts = msprime.sim_ancestry(
            samples=samples, 
            demography=demography,
            sequence_length=sequence_length,
            recombination_rate=recombination_rate, ploidy=ploidy, random_seed=SEED)

    else:
        ts = msprime.sim_ancestry(
            samples=samples, 
            population_size=population_size,
            sequence_length=sequence_length,
            recombination_rate=recombination_rate, ploidy=ploidy, random_seed=SEED)
    ts = msprime.mutate(ts, rate=mutation_rate, random_seed=SEED)
    return ts

def calculate_window_sfs_vectorized(site_positions, pivot_frequencies, window_size, step_size, sequence_length, num_samples):
    """Memory-efficient vectorized calculation of SFS"""
    n_windows = int(np.ceil(sequence_length / step_size))
    window_starts = np.arange(n_windows) * step_size
    window_ends = np.minimum(window_starts + window_size, sequence_length)
    site_in_window = (site_positions[:, np.newaxis] >= window_starts) & \
                     (site_positions[:, np.newaxis] < window_ends)
    freq_ranges = np.arange(num_samples)
    sfs_array = np.zeros((n_windows, num_samples), dtype=int)
    for i in range(n_windows):
        window_freqs = pivot_frequencies[site_in_window[:, i]]
        if len(window_freqs) > 0:
            sfs_array[i] = np.bincount(window_freqs, minlength=num_samples)
    return sfs_array


w_multipliers = np.array([2, 8, 32, 64])
def ts2X_vectorized(ts, window_size=4000, step_size=2000, xor_ops=xor, pivot_A=0, pivot_B=1):
    """Memory-efficient conversion of tree sequence to feature matrix"""
    site_positions = retrieve_site_positions(ts)
    gm = ts.genotype_matrix().T
    num_samples = gm.shape[0]
    frequencies = gm.sum(0)
    sequence_length = ts.sequence_length
    
    # Calculate xor product
    xor_freqs = frequencies * xor_ops(gm[pivot_A], gm[pivot_B])
    
    # Pre-allocate output array
    w_multipliers = np.array([2, 8, 32, 64])
    Xs = np.zeros((len(w_multipliers), 
                   int(np.ceil(sequence_length / step_size)), 
                   num_samples), dtype=int)
    
    # Calculate for each window size
    for i, w in enumerate(w_multipliers):
        Xs[i] = calculate_window_sfs_vectorized(
            site_positions=site_positions,
            pivot_frequencies=xor_freqs,
            window_size=window_size * w,
            step_size=step_size,
            sequence_length=sequence_length,
            num_samples=num_samples
        )
    
    return Xs

def interpolate_tmrca_per_window(
    position: List[float],
    tmrca: List[float],
    interval_start: int = 0,
    interval_end: int = 500_000,
    interval_size: int = 2000,
    num_points: int = 200
) -> np.ndarray:
    """Calculates average TMRCA estimate for given intervals."""
    interp_function = interp1d(position, tmrca, kind='previous', fill_value="extrapolate")
    intervals = np.arange(interval_start, interval_end, interval_size)
    def calculate_interval_average(start: float, end: float) -> float:
        x_vals = np.linspace(start, end, num_points)
        y_vals = interp_function(x_vals)
        return np.mean(y_vals)
    averages = [calculate_interval_average(start, end) 
                for start, end in zip(intervals[:-1], intervals[1:])]
    return np.array(averages)

def interpolate_tmrcas(ts: tskit.TreeSequence, window_size: int) -> np.ndarray:
    """Calculate the interpolated TMRCA values for a given tree sequence."""
    def extract_tmrca_data(tree: tskit.Tree) -> Tuple[float, float, int, float]:
        left, right = tree.interval
        node = next(node for node in tree.nodes() if node not in [0, 1])
        tmrca = tree.time(node)
        return left, right, node, tmrca
    tmrca_landscape = [extract_tmrca_data(tree) for tree in ts.trees()]
    tmrca_array = np.array(tmrca_landscape)

    y_tmrca_interpolated = interpolate_tmrca_per_window(
        position=tmrca_array[:, 0],
        tmrca=tmrca_array[:, 3],
        interval_end=int(ts.sequence_length) + window_size,
        interval_size=window_size
    )
    return y_tmrca_interpolated



def ts2input(ts, pivot_A, pivot_B):
    Xxor = ts2X_vectorized(ts, window_size=2000, xor_ops=xor, pivot_A=pivot_A, pivot_B=pivot_B).astype(np.float16)
    Xxnor = ts2X_vectorized(ts, window_size=2000, xor_ops=xnor, pivot_A=pivot_A, pivot_B=pivot_B).astype(np.float16)
    src = np.stack([Xxor, Xxnor], axis=0).astype(np.float16)
    tgt = np.log(interpolate_tmrcas(ts.simplify(samples=[pivot_A, pivot_B]), window_size=2000)).astype(np.float16)
    src = torch.from_numpy(src).float()
    src = torch.log1p(src)
    tgt = np.array(discretize(tgt, np.linspace(3, 14, 256)))
    tgt = torch.from_numpy(tgt).long() + 2
    tgt = torch.cat([torch.tensor([1]), tgt])
    return src, tgt

def ts2input_numpy(ts, pivot_A, pivot_B):
    Xxor = ts2X_vectorized(ts, window_size=2000, xor_ops=xor, pivot_A=pivot_A, pivot_B=pivot_B).astype(np.float16)
    Xxnor = ts2X_vectorized(ts, window_size=2000, xor_ops=xnor, pivot_A=pivot_A, pivot_B=pivot_B).astype(np.float16)
    src = np.stack([Xxor, Xxnor], axis=0).astype(np.float16)
    tgt = np.log(interpolate_tmrcas(ts.simplify(samples=[pivot_A, pivot_B]), window_size=2000)).astype(np.float16)
    src = np.log1p(src)  # Log transformation
    tgt = np.array(discretize(tgt, np.linspace(3, 14, 256)))
    tgt = np.concatenate([[1], tgt + 2])  # Add special token and shift indices
    return src, tgt

def generate_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]




TIMES = np.linspace(3, 14, 256)

def create_sawtooth_demogaphy_object(Ne = 2*10**4, magnitue=4):
    demography = msprime.Demography()
    demography.add_population(initial_size=(Ne))
    demography.add_population_parameters_change(time=20, population=None,
    growth_rate=6437.7516497364/(4*10**4))
    demography.add_population_parameters_change(time=30, growth_rate=-378.691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=200, growth_rate=-643.77516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=300, growth_rate=37.8691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=2000, growth_rate=64.377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=3000, growth_rate=-3.78691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=20000, growth_rate=-6.4377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=30000, growth_rate=0.378691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=200000, growth_rate=0.64377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=300000, growth_rate=-0.0378691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=2000000, growth_rate=-0.064377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=3000000, growth_rate=0.00378691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=20000000, growth_rate=0,initial_size=Ne)
    return demography

def population_time(time_rate:float=0.06, tmax:int = 510_000,
                        num_time_windows:int = 40) -> np.array :
    population_time = np.repeat([(np.exp(np.log(1 + time_rate * tmax) * i /
                              (num_time_windows - 1)) - 1) / time_rate for i in
                              range(num_time_windows)], 1, axis=0)
    population_time[0] = 1
    return population_time

def post_process(tgt, sequence, TIMES):
    yhat = sequence[:, 1:].cpu().numpy() - 2
    ytrue = tgt[:, 1:].cpu().numpy() - 2
    return TIMES[yhat], TIMES[ytrue]

def process_pair(args):
    ts, pivot_A, pivot_B = args
    return ts2input_numpy(ts, pivot_A, pivot_B)


def decreasing_mses(yhats, ytrues):
    mse_values = []
    for i in range(1, len(yhats) + 1):
        yhat_mean = sum(yhats[:i]) / i
        ytrue_mean = sum(ytrues[:i]) / i
        mse_values.append(mse(yhat_mean, ytrue_mean))
    return mse_values


