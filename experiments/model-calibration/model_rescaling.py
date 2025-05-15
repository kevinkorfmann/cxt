import numpy as np
import argparse
import stdpopsim
import copy

import matplotlib.pyplot as plt
from msprime import demography


def scale_model(demogr: stdpopsim.DemographicModel, scale: float = 1.0):
    """
    Scale model parameters such that the shape of the SFS is unchanged, but
    segregating sites increases / decreases. For independent loci, this is
    equivalent to changing the mutation rate. However, for recombinant sequences
    the rescaled model is not equivalent, as recombination is influenced by
    population sizes.
    """
    demogr = copy.deepcopy(demogr)
    for pop in demogr.model.populations:
        if pop.initial_size is not None:
            pop.initial_size *= scale
        if pop.growth_rate is not None:
            pop.growth_rate /= scale
    demogr.model.migration_matrix /= scale
    for event in demogr.model.events:
        event.time *= scale
        match type(event):
            case demography.PopulationParametersChange:
                if event.initial_size is not None:
                    event.initial_size *= scale
                if event.growth_rate is not None:
                    event.growth_rate /= scale
            case demography.MigrationRateChange:
                if event.rate is not None:
                    event.rate /= scale
            case demography.MassMigration:
                pass
            case demography.PopulationSplit:
                pass
            case _:
                raise ValueError(f"Could not match {event}")
    return demogr


def scale_migration_rates(demogr: stdpopsim.DemographicModel, scale: float = 1.0):
    """
    Scale migration rates by provided factor while keeping other
    parameters unchanged.
    """
    demogr = copy.deepcopy(demogr)
    demogr.model.migration_matrix *= scale
    for event in demogr.model.events:
        match type(event):
            case demography.MigrationRateChange:
                if event.rate is not None:
                    event.rate *= scale
            case _:
                pass
    return demogr


def scale_growth_rates(demogr: stdpopsim.DemographicModel, scale: float = 1.0):
    """
    Scale exponential growth rates by provided factor while keeping other
    parameters unchanged.
    """
    demogr = copy.deepcopy(demogr)
    for pop in demogr.model.populations:
        if pop.growth_rate is not None:
            pop.growth_rate *= scale
    for event in demogr.model.events:
        match type(event):
            case demography.PopulationParametersChange:
                if event.growth_rate is not None:
                    event.growth_rate *= scale
            case _:
                pass
    return demogr


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visual tests for functions")
    parser.add_argument(
        "--output-path", 
        type=str,
        default="/home/natep/public_html/cxt/model-calibration/", 
    )
    parser.add_argument(
        "--demography",
        type=str,
        default="Zigzag_1S14", 
    )
    parser.add_argument(
        "--population",
        type=str,
        default="generic",
    )
    args = parser.parse_args()

    homsap = stdpopsim.get_species("HomSap")
    contig = homsap.get_contig("chr1", right=10e6)
    demogr = homsap.get_demographic_model(args.demography)
    ## DEBUG: 
    #demogr.model.migration_matrix *= 10
    #print(demogr.model.migration_matrix)
    ## /DEBUG
    populn = demogr.populations[0].name \
        if args.population is None else args.population
    sample = {populn: 10}
    engine = stdpopsim.get_engine("msprime")
    
    # sanity check: normalised SFS should match exactly
    num_reps = 10
    afs_1 = None
    afs_2 = None
    for rep in range(num_reps):
        print(f"Running {rep}")
        ts_1 = engine.simulate(
            contig=contig, 
            demographic_model=demogr, 
            samples=sample, 
            seed=rep,
        )
        ts_2 = engine.simulate(
            contig=contig, 
            demographic_model=scale_model(demogr, 2), 
            samples=sample, 
            seed=1000 + rep,
        )
        if afs_1 is None:
            afs_1 = ts_1.allele_frequency_spectrum(polarised=True)
        else:
            afs_1 += ts_1.allele_frequency_spectrum(polarised=True)
        if afs_2 is None:
            afs_2 = ts_2.allele_frequency_spectrum(polarised=True)
        else:
            afs_2 += ts_2.allele_frequency_spectrum(polarised=True)
        
    afs_1 /= num_reps
    afs_2 /= num_reps
    afs_1[0] = 0
    afs_1[-1] = 0
    afs_2[0] = 0
    afs_2[-1] = 0
    
    m1 = np.mean(np.log10(afs_1[afs_1 > 0]))
    m2 = np.mean(np.log10(afs_1[afs_1 > 0]/afs_1.sum()))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axs[0].plot(np.log10(afs_1.flatten()), np.log10(afs_2.flatten()), "o", color="red", markersize=4)
    axs[0].axline((m1, m1), slope=1, linestyle="dashed", color="black")
    axs[1].plot(np.log10(afs_1.flatten()/afs_1.sum()), np.log10(afs_2.flatten()/afs_2.sum()), "o", color="red", markersize=4)
    axs[1].axline((m2, m2), slope=1, linestyle="dashed", color="black")
    fig.supxlabel("E[SFS] from original model (log10)")
    fig.supylabel("E[SFS] from scaled model (log10)")
    plt.savefig(f"{args.output_path}/sanity-check-scale-model.png")
    
    
    
