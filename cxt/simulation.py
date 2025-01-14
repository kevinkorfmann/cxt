import os
import msprime
import numpy as np
from tqdm import tqdm
from functools import partial
from cxt.utils import xor, xnor
from multiprocessing import Pool
from cxt.utils import ts2X_vectorized
from cxt.utils import simulate_parameterized_tree_sequence, interpolate_tmrcas
import argparse


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



def generate_data(args) -> tuple:
    """
    Generate data for simulation.
    Parameters
    ----------
    i : int
        A seed integer parameter used by the ts_simulation_func.
    pivot_A : int
        The first pivot index for processing, by default 0.
    pivot_B : int
        The second pivot index for processing, by default 1.
    ts_simulation_func : callable
        A function to simulate tree sequence data, by default None.
    Returns
    -------
    tuple
        A tuple containing:
        - X : np.ndarray
            A 3D array with processed data using XOR and XNOR operations.
        - y : np.ndarray
            A 1D array with log-interpolated TMRCA values.
    """
    i, pivot_A, pivot_B, ts_simulation_func, randomize_pivots = args 
    ts = ts_simulation_func(i)
    
    if randomize_pivots:
        pivots = np.arange(0, 50)
        np.random.shuffle(pivots)
        pivot_A = pivots[0]
        pivot_B = pivots[1]

    # processing
    Xxor = ts2X_vectorized(ts, window_size=2000, xor_ops=xor, pivot_A=pivot_A, pivot_B=pivot_B).astype(np.float16)
    Xxnor = ts2X_vectorized(ts, window_size=2000, xor_ops=xnor, pivot_A=pivot_A, pivot_B=pivot_B).astype(np.float16)
    X = np.stack([Xxor, Xxnor], axis=0).astype(np.float16)
        
    y = np.log(interpolate_tmrcas(ts.simplify(samples=[pivot_A, pivot_B]), window_size=2000)).astype(np.float16)
    return X, y

def save_batch(data, idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/X_{idx}.npy', np.stack([X for X, _ in data]))
    np.save(f'{output_dir}/y_{idx}.npy', np.stack([y for _, y in data]))

def process_batches(num_samples: int, start_batch: int,
                    batch_size: int, num_processes: int,
                    output_dir: str, pivot_A: int, pivot_B: int,
                    ts_simulation_func: callable, randomize_pivots: bool) -> None:
    """
    Process data in batches using multiprocessing and save the results.

    Parameters
    ----------
    num_samples : int
        Total number of samples to process.
    start_batch : int
        The starting batch index.
    batch_size : int
        The number of samples per batch.
    num_processes : int
        The number of processes to use for multiprocessing.
    output_dir : str
        The directory where the output batches will be saved.
    pivot_A : int, optional
            The first pivot index for processing, by default 0.
    pivot_B : int, optional
            The second pivot index for processing, by default 1.
    ts_simulation_func : callable
        The simulation function to generate tree sequence data.
    randomize_pivots : bool
        Randomize pivot indices, by default False.

    Returns
    -------
    None
    """
    with Pool(num_processes) as pool:
        batch = []
        batch_idx = 0 + start_batch
        start_sample = batch_idx * batch_size
        args = []
        for i in range(start_sample, num_samples):
            args.append((i, pivot_A, pivot_B, ts_simulation_func, randomize_pivots))

        for i, result in enumerate(tqdm(pool.imap(generate_data, args), total=num_samples-start_sample)):
            batch.append(result)
            if len(batch) == batch_size or i == num_samples - 1:
                save_batch(batch, batch_idx, output_dir)
                batch = []
                batch_idx += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process simulation parameters.')
    parser.add_argument('--num_processes', type=int, default=30, help='Number of processes to use')
    parser.add_argument('--num_samples', type=int, default=2_000_000, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for saving data')
    parser.add_argument('--start_batch', type=int, default=0, help='Starting batch index')
    parser.add_argument('--pivot_A', type=int, default=0, help='Pivot A index')
    parser.add_argument('--pivot_B', type=int, default=1, help='Pivot B index')
    parser.add_argument('--data_dir', type=str, default='/sietch_colab/kkor/tiny_batches_base_dataset', help='Directory to save data')
    parser.add_argument('--scenario', type=str, choices=['constant', 'sawtooth', 'island'], default='constant', help='Scenario type')
    parser.add_argument('--randomize_pivots', default=False, help='Randomize pivot indices')
    args = parser.parse_args()

    num_processes = args.num_processes
    num_samples = args.num_samples
    batch_size = args.batch_size
    start_batch = args.start_batch
    pivot_A = args.pivot_A
    pivot_B = args.pivot_B
    data_dir = args.data_dir
    scenario = args.scenario
    randomize_pivots = args.randomize_pivots

    if scenario == "constant":
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B, simulate_parameterized_tree_sequence, randomize_pivots)

    elif scenario == "sawtooth":
        simulate_parameterized_tree_sequence_sawtooth = partial(simulate_parameterized_tree_sequence, demography=create_sawtooth_demogaphy_object(Ne=20e3, magnitue=3))
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B, simulate_parameterized_tree_sequence_sawtooth, randomize_pivots)

    elif scenario == "island":
        samples = {0: 15, 1: 5, 2: 5}
        island_demography = msprime.Demography.island_model([10000, 5000, 5000], migration_rate=0.1)
        simulate_parameterized_tree_sequence_island = partial(simulate_parameterized_tree_sequence, island_demography=island_demography, samples=samples)
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B, simulate_parameterized_tree_sequence_island, randomize_pivots)
