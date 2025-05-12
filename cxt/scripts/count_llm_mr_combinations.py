import pandas as pd
from collections import defaultdict

# Provided directory names
dir_names = [
    "hard_sweeps_0.01_2.5e+05_2e+04_1.0e-08_1.0e-08", "island_3pop_0.2_1e+04_5.0e-08_1.0e-08",
    "hard_sweeps_0.01_2.5e+05_2e+04_1.0e-08_5.0e-08", "ne_constant_1e+04_1.0e-08_1.0e-08",
    "hard_sweeps_0.01_2.5e+05_4e+04_1.0e-08_1.0e-08", "ne_constant_1e+04_1.0e-08_5.0e-08",
    "hard_sweeps_0.01_5.0e+05_2e+04_5.0e-08_1.0e-08", "ne_constant_1e+04_5.0e-08_1.0e-08",
    "hard_sweeps_0.01_5.0e+05_4e+04_5.0e-08_1.0e-08", "ne_constant_2e+04_1.0e-08_1.0e-08",
    "hard_sweeps_0.01_7.5e+05_1e+04_1.0e-08_1.0e-08", "ne_constant_2e+04_1.0e-08_5.0e-08",
    "hard_sweeps_0.01_7.5e+05_1e+04_1.0e-08_5.0e-08", "ne_constant_2e+04_5.0e-08_1.0e-08",
    "hard_sweeps_0.01_7.5e+05_1e+04_5.0e-08_1.0e-08", "ne_constant_4e+04_1.0e-08_1.0e-08",
    "hard_sweeps_0.01_7.5e+05_4e+04_1.0e-08_5.0e-08", "ne_constant_4e+04_1.0e-08_5.0e-08",
    "hard_sweeps_0.1_2.5e+05_1e+04_1.0e-08_1.0e-08", "ne_constant_4e+04_5.0e-08_1.0e-08",
    "hard_sweeps_0.1_2.5e+05_1e+04_1.0e-08_5.0e-08", "ne_sawtooth_3_1e+04_1.0e-08_1.0e-08",
    "hard_sweeps_0.1_2.5e+05_4e+04_1.0e-08_1.0e-08", "ne_sawtooth_3_1e+04_1.0e-08_5.0e-08",
    "hard_sweeps_0.1_5.0e+05_1e+04_5.0e-08_1.0e-08", "ne_sawtooth_3_1e+04_5.0e-08_1.0e-08",
    "hard_sweeps_0.1_5.0e+05_2e+04_1.0e-08_5.0e-08", "ne_sawtooth_3_2e+04_1.0e-08_1.0e-08",
    "hard_sweeps_0.1_5.0e+05_2e+04_5.0e-08_1.0e-08", "ne_sawtooth_3_2e+04_1.0e-08_5.0e-08",
    "hard_sweeps_0.1_7.5e+05_2e+04_1.0e-08_1.0e-08", "ne_sawtooth_3_2e+04_5.0e-08_1.0e-08",
    "hard_sweeps_0.1_7.5e+05_4e+04_1.0e-08_5.0e-08", "ne_sawtooth_3_4e+04_1.0e-08_1.0e-08",
    "hard_sweeps_0.1_7.5e+05_4e+04_5.0e-08_1.0e-08", "ne_sawtooth_3_4e+04_1.0e-08_5.0e-08",
    "hard_sweeps_1_2.5e+05_1e+04_1.0e-08_5.0e-08", "ne_sawtooth_3_4e+04_5.0e-08_1.0e-08",
    "hard_sweeps_1_2.5e+05_2e+04_5.0e-08_1.0e-08", "ne_sawtooth_4_1e+04_1.0e-08_1.0e-08",
    "hard_sweeps_1_2.5e+05_4e+04_1.0e-08_1.0e-08", "ne_sawtooth_4_1e+04_1.0e-08_5.0e-08",
    "hard_sweeps_1_5.0e+05_2e+04_1.0e-08_1.0e-08", "ne_sawtooth_4_1e+04_5.0e-08_1.0e-08",
    "hard_sweeps_1_5.0e+05_2e+04_1.0e-08_5.0e-08", "ne_sawtooth_4_2e+04_1.0e-08_1.0e-08",
    "hard_sweeps_1_5.0e+05_4e+04_5.0e-08_1.0e-08", "ne_sawtooth_4_2e+04_1.0e-08_5.0e-08",
    "hard_sweeps_1_7.5e+05_1e+04_1.0e-08_1.0e-08", "ne_sawtooth_4_2e+04_5.0e-08_1.0e-08",
    "hard_sweeps_1_7.5e+05_1e+04_5.0e-08_1.0e-08", "ne_sawtooth_4_4e+04_1.0e-08_1.0e-08",
    "hard_sweeps_1_7.5e+05_4e+04_1.0e-08_5.0e-08", "ne_sawtooth_4_4e+04_1.0e-08_5.0e-08",
    "island_3pop_0.05_1e+04_1.0e-08_1.0e-08", "ne_sawtooth_4_4e+04_5.0e-08_1.0e-08",
    "island_3pop_0.05_1e+04_1.0e-08_5.0e-08", "ne_sawtooth_5_1e+04_1.0e-08_1.0e-08",
    "island_3pop_0.05_1e+04_5.0e-08_1.0e-08", "ne_sawtooth_5_1e+04_1.0e-08_5.0e-08",
    "island_3pop_0.05_2e+04_1.0e-08_1.0e-08", "ne_sawtooth_5_1e+04_5.0e-08_1.0e-08",
    "island_3pop_0.05_2e+04_1.0e-08_5.0e-08", "ne_sawtooth_5_2e+04_1.0e-08_1.0e-08",
    "island_3pop_0.05_2e+04_5.0e-08_1.0e-08", "ne_sawtooth_5_2e+04_1.0e-08_5.0e-08",
    "island_3pop_0.05_4e+04_1.0e-08_1.0e-08", "ne_sawtooth_5_2e+04_5.0e-08_1.0e-08",
    "island_3pop_0.05_4e+04_1.0e-08_5.0e-08", "ne_sawtooth_5_4e+04_1.0e-08_1.0e-08",
    "island_3pop_0.05_4e+04_5.0e-08_1.0e-08", "ne_sawtooth_5_4e+04_1.0e-08_5.0e-08",
    "island_3pop_0.2_1e+04_1.0e-08_1.0e-08", "ne_sawtooth_5_4e+04_5.0e-08_1.0e-08"
]

# sample counts by scenario prefix
scenario_counts = {
    'ne_constant': 100_000,
    'hard_sweeps': 10_000,
    'island_3pop': 10_000,
    'ne_sawtooth': 25_000,
}

# aggregate totals
totals = defaultdict(int)
for name in dir_names:
    scenario = next((s for s in scenario_counts if name.startswith(s)), None)
    if not scenario:
        continue
    count = scenario_counts[scenario]
    parts = name.split('_')
    mut, rec = parts[-2], parts[-1]
    totals[(mut, rec)] += count

# build DataFrame
df = pd.DataFrame([{'Mutation': m, 'Recombination': r, 'Total_Samples': total}
                   for (m, r), total in totals.items()])
df = df.sort_values(['Mutation', 'Recombination']).reset_index(drop=True)
print(df)
