import msprime
import tskit
from typing import List, Tuple
from scipy.interpolate import interp1d
import numpy as np
import torch

#TIMES = np.linspace(3, 14, 256)
TIMES = np.linspace(3, 17, 324)


retrieve_site_positions = lambda ts: np.array([site.position for site in ts.sites()])
xor = lambda a, b: (a^b).astype(int)
xnor = lambda a, b: (1 - xor(a, b)).astype(int)
mse = lambda yhat, ytrue: ((yhat - ytrue)**2).mean()

def discretize(sequence, population_time):
    indices = np.searchsorted(population_time, sequence, side="right") - 1
    indices = np.clip(indices, 0, len(population_time) - 1)
    return indices.tolist()

def sample_population_size(n_min: int = 10, n_max: int = 100_000, num_time_windows: int = 20, seed: int = None) -> list[float]:
    rng = np.random.default_rng(seed)  # Use a local random generator
    n_min_log = np.log(n_min)
    n_max_log = np.log(n_max)
    population_size = [rng.uniform(low=n_min_log, high=n_max_log)]
    
    for _ in range(num_time_windows - 1):
        next_population_size = population_size[-1] * rng.uniform(0.5, 1.5)
        while not (n_min_log <= next_population_size <= n_max_log):
            next_population_size = population_size[-1] * rng.uniform(0.5, 1.5)
        population_size.append(next_population_size)
    
    # Smooth the trajectory using a moving average
    population_size = np.convolve(population_size, np.ones(3) / 3, mode="same")
    population_size[0] = population_size[1]
    population_size[-1] = population_size[-2]
    return np.exp(population_size).tolist()  

def sample_demography(n_min=10_000, n_max=200_000, num_time_windows=20, seed=None):
    time_steps = np.linspace(3, 14, num_time_windows)
    population_size = sample_population_size(n_min, n_max, num_time_windows, seed=seed)
    demography=msprime.Demography()
    demography.add_population(initial_size=(population_size[0]))
    for i, (time, size) in enumerate(zip(time_steps, population_size)):
        demography.add_population_parameters_change(time=time, initial_size=size)
    return demography



class DemographyStorage:
    def __init__(self, mutation_rate_range=(1e-9, 1e-8), recombination_rate_range=(1e-9, 1e-8)):
        self.mutation_rate_range = mutation_rate_range
        self.recombination_rate_range = recombination_rate_range
        
        self.demography_seeds  = [
            0x8534AA, 0x4BAA50, 0xC47D25, 0x835F0D, 0x28CEC5, 0x29C8FB, 0xEC5342, 0xDF16F4,
            0x6BB979, 0x72645C, 0x5A0518, 0xA2DB99, 0xB08ED9, 0xC6464C, 0x2259B4, 0xAEE45A,
            0x907817, 0x6271C5, 0xF61BBE, 0xD4E56A, 0x451CA4, 0xFEFF41, 0x994C38, 0xFC971C,
            0x7EA959, 0xB893CD, 0x7623E6, 0x725A75, 0x8DFBFF, 0xD50EDE, 0x8097F1, 0xECB91F,
            0xBB2A96, 0x221B88, 0x2EC6F3, 0x5CF515, 0x634F11, 0xAB689E, 0x711F54, 0xD16D4D,
            0x5F846E, 0xAF8922, 0x1C016C, 0xEA592C, 0x72D1FE, 0xF6389F, 0xCBD018, 0x59E61C,
            0x24D3EF, 0x944E92, 0x1E9A5E, 0x88C3A4, 0x61860D, 0xC27EA8, 0x568673, 0xD4B5A0,
            0xE32F8D, 0xA4E4A3, 0x83ECD2, 0xC5A2D9, 0xCE4F5C, 0x34DEA3, 0x5E33B3, 0x589D1B
        ]
        self.combinations = self._generate_combinations()
    
    def _generate_combinations(self, local_seed=0x8534AA):
        combinations = []
        rng = np.random.default_rng(local_seed)
        mutation_rates = rng.uniform(*self.mutation_rate_range, size=len(self.demography_seeds))
        recombination_rates = rng.uniform(*self.recombination_rate_range, size=len(self.demography_seeds))
        for i, demography_seed in enumerate(self.demography_seeds):
            mutation_rate = mutation_rates[i]
            recombination_rate = recombination_rates[i]
            demography = sample_demography(seed=demography_seed)
            combinations.append((mutation_rate, recombination_rate, demography))
        return combinations

    
    def get_combinations(self):
        return self.combinations
    

storage = DemographyStorage()


def simulate_parameterized_tree_sequence(
        seed,
        samples=25,
        population_size=2e4,
        sequence_length=1e6,
        recombination_rate=1.28e-8,
        ploidy=2,
        mutation_rate=1.29e-8,
        demography=None,
        island_demography=None, #msprime.Demography.island_model([10000, 5000, 5000], migration_rate=0.1)
        hard_sweep=None,
        random_scenario=None,
        selection_coefficient=None,
        selection_position=0.5e6
        ):
    np.random.seed(seed)
    SEED = np.random.uniform(1, 2**32 - 1)

    assert sum(x is not None for x in [demography, island_demography, hard_sweep, random_scenario]) in [0, 1]

    if demography:
        ts = msprime.sim_ancestry(
            samples=samples, 
            demography=demography,
            sequence_length=sequence_length,
            recombination_rate=recombination_rate, ploidy=ploidy, random_seed=SEED)
        
    elif random_scenario:
        # [1, ... ,65]
        assert random_scenario-1 < len(storage.demography_seeds), "Invalid random scenario index"
        random_scenario -= 1
        mutation_rate, recombination_rate, demography = storage.get_combinations()[random_scenario]
        ts = msprime.sim_ancestry(
            samples=samples, 
            demography=demography,
            sequence_length=sequence_length,
            recombination_rate=recombination_rate, ploidy=ploidy, random_seed=SEED)

    elif island_demography:
        ts = msprime.sim_ancestry(
            samples=samples, # {0: 15, 1: 5, 2: 5}
            sequence_length=sequence_length,
            demography=island_demography,
            recombination_rate=recombination_rate, ploidy=ploidy, random_seed=SEED)
    elif hard_sweep:
        sweep_model = msprime.SweepGenicSelection(
        position=selection_position,  
        start_frequency=1.0 / (population_size),
        end_frequency=1.0 - (1.0 / (population_size)),
        s=selection_coefficient,
        dt=1e-6
    )
        ts = msprime.sim_ancestry(
            samples=samples,
            model=[sweep_model, msprime.StandardCoalescent()],
            population_size=population_size,
            recombination_rate=recombination_rate,
            sequence_length=sequence_length,
        )

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
def ts2X_vectorized(ts, window_size=4000, step_size=2000, xor_ops=xor, pivot_A=0, pivot_B=1, offset=0):
    """Memory-efficient conversion of tree sequence to feature matrix"""
    site_positions = retrieve_site_positions(ts)
    site_positions -= offset
    gm = ts.genotype_matrix().T

    mask = np.logical_or(np.any(gm >= 2, axis=0), gm.sum(0) >= ts.num_samples)
    gm = gm[:, ~mask]
    site_positions = site_positions[~mask]

    num_samples = gm.shape[0]
    
    frequencies = gm.sum(0)
    sequence_length = 1e6#ts.sequence_length
    
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

    sequence_length = 1e6
    y_tmrca_interpolated = interpolate_tmrca_per_window(
        position=tmrca_array[:, 0],
        tmrca=tmrca_array[:, 3],
        interval_end=int(sequence_length) + window_size,
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
    #tgt = np.array(discretize(tgt, np.linspace(3, 14, 256)))
    tgt = np.array(discretize(tgt, TIMES))
    tgt = torch.from_numpy(tgt).long() + 2
    tgt = torch.cat([torch.tensor([1]), tgt])
    return src, tgt

def ts2input_numpy(ts, pivot_A, pivot_B, offset=0, ignore_target=False):
    Xxor = ts2X_vectorized(ts, window_size=2000, xor_ops=xor, pivot_A=pivot_A, pivot_B=pivot_B, offset=offset).astype(np.float16)
    Xxnor = ts2X_vectorized(ts, window_size=2000, xor_ops=xnor, pivot_A=pivot_A, pivot_B=pivot_B, offset=offset).astype(np.float16)
    src = np.stack([Xxor, Xxnor], axis=0).astype(np.float16)
    if not ignore_target:
        tgt = np.log(interpolate_tmrcas(ts.simplify(samples=[pivot_A, pivot_B]), window_size=2000)).astype(np.float16)
    src = np.log1p(src)  # Log transformation
    #tgt = np.array(discretize(tgt, np.linspace(3, 14, 256)))
    if not ignore_target:
        tgt = np.array(discretize(tgt, TIMES))
        tgt = np.concatenate([[1], tgt + 2])  # Add special token and shift indices
    else:
        tgt = np.ones(src.shape[1], dtype=int) * 2
    return src, tgt

#def generate_causal_mask(seq_len, device):
#    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
#    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

def generate_causal_mask(seq_len, full_attention_n=None, device="cpu"):
    full_attention_n = full_attention_n if full_attention_n is not None else 0
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask[:full_attention_n, :full_attention_n] = 1  # Full attention for first n tokens
    return mask.bool().unsqueeze(0).unsqueeze(0)







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
    ts, pivot_A, pivot_B, offset, ignore_target = args
    return ts2input_numpy(ts, pivot_A, pivot_B, offset, ignore_target)


def accumulating_mses(yhats, ytrues):
    mse_values = []
    for i in range(1, len(yhats) + 1):
        yhat_mean = sum(yhats[:i]) / i
        ytrue_mean = sum(ytrues[:i]) / i
        mse_values.append(mse(yhat_mean, ytrue_mean))
    return mse_values


def diversity_bias_correction(
    tree_sequence: tskit.TreeSequence,
    mutation_rate: float,
    predictions: np.ndarray,
    pivot_pairs: np.ndarray,
    zero_offset: float = 0.0,
    return_intercept: bool = False,
) -> (np.ndarray, np.ndarray):
    """
    Correct the predicted TMRCAs such that expected diversity matches
    observed diversity, for a given mutation rate. 

    The input predictions are assumed to have dimensions 
    `(replicates, pairs, windows)`.

    The correction is undefined for pairs with zero observed diversity.
    In this case, an offset equal to the minimum nonzero diversity in the sample 
    is added (if `zero_offset` is zero, default).
    """
    assert predictions.ndim == 3
    assert pivot_pairs.ndim == 2
    assert pivot_pairs.shape[0] == predictions.shape[1]
    assert pivot_pairs.shape[1] == 2
    obs_div = tree_sequence.trim().diversity(sample_sets=pivot_pairs)
    # NB: average over replicates in log space
    fit_div = 2 * np.exp(predictions.mean(axis=0)).mean(axis=-1) * mutation_rate
    obs_div[obs_div == 0] = obs_div[obs_div > 0].min() if zero_offset == 0 \
        else zero_offset / tree_sequence.trim().sequence_length
    assert np.all(obs_div > 0)
    assert np.all(fit_div > 0)
    corrected = predictions + (np.log(obs_div) - np.log(fit_div))[np.newaxis, :, np.newaxis]
    intercept = np.log(obs_div / mutation_rate / 2)[None, :, None]
    return corrected if not return_intercept else (corrected, intercept)



