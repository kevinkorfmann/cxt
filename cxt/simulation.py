import os
import msprime

import numpy as np
from tqdm import tqdm
from functools import partial
from cxt.utils import xor, xnor
from multiprocessing import Pool
from cxt.utils import ts2X_vectorized
from cxt.utils import simulate_parameterized_tree_sequence, interpolate_tmrcas


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


def generate_data(i, pivot_A=0, pivot_B=1, ts_simulation_func=None):

    # ts simulation
    ts = ts_simulation_func(i)

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

def process_batches(num_samples, start_batch, batch_size, num_processes, output_dir, pivot_A, pivot_B, ts_simulation_func):
    with Pool(num_processes) as pool:
        batch = []
        batch_idx = 0 + start_batch
        start_sample = batch_idx * batch_size
        args = []
        for i in range(start_sample, num_samples):
            args.append((i, pivot_A, pivot_B, ts_simulation_func))
        for i, result in enumerate(tqdm(pool.imap(generate_data, args), total=num_samples-start_sample)):
            batch.append(result)
            if len(batch) == batch_size or i == num_samples - 1:
                save_batch(batch, batch_idx, output_dir)
                batch = []
                batch_idx += 1


if __name__ == '__main__':

    num_processes = 30
    num_samples = 2_000_000
    batch_size = 1000
    start_batch = 0
    pivot_A, pivot_B = 0, 1
    #simulate_parameterized_tree_sequence_sawtooth = partial(simulate_parameterized_tree_sequence, demography=create_sawtooth_demogaphy_object(Ne=20e3, magnitue=3))
    process_batches(num_samples, start_batch, batch_size, num_processes, '/sietch_colab/kkor/tiny_batches_base_dataset', pivot_A, pivot_B, simulate_parameterized_tree_sequence)