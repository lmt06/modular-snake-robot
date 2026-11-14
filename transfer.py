# transfer.py
# Tests evolved modular controllers on body lengths they weren't trained on

import numpy as np
from fitness import evaluate

SEEDS = [1, 2, 3, 4, 5]
BODIES = {                      # body length -> XML file (add more as you make them), tells what bodies to test genome on
    # 5: "./swimmer_n5.xml",     
    # 8: "./swimmer_n8.xml",
    # 3: "./swimmer_n3.xml"
    10: "./swimmer_n10.xml"
}
EPISODES = 10

# results[N] holds one score per seed
results = {n: [] for n in BODIES}

for seed in SEEDS:
    genome = np.load(f"n5_mod_e3_s{seed}.npy")
    for n, xml in BODIES.items():
        score = evaluate(genome, n_episodes=EPISODES, controller="modular", xml_file=xml)
        results[n].append(score)
        print(f"Seed {seed}, N={n}: {score:.3f}")

# Summary table
print("\nN    mean     std     per-seed scores")
for n in BODIES:
    scores = np.array(results[n])
    print(f"{n:<4} {scores.mean():<8.3f} {scores.std():<7.3f} {np.round(scores, 2)}")

# Save so plots can be made later without rerunning
np.savez("transfer5n10_e3_results.npz", **{f"N{n}": np.array(results[n]) for n in BODIES})
print("\nSaved transfer_results.npz")