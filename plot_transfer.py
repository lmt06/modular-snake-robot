# plot_transfer.py
# Plots how far the modular (evolved at N=5) and centralized controllers swim at each body length

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", help="transfer result files, e.g. transfer_n3.npz transfer_n8.npz")
parser.add_argument("--lengths", type=int, nargs="+", default=None,
                    help="which body lengths to show, e.g. --lengths 5 8 (default: all)")
parser.add_argument("--output", type=str, default="transfer.png")
parser.add_argument("--title", type=str, default="Transfer of modular controllers to new body lengths")
args = parser.parse_args()

SEEDS = [1, 2, 3, 4, 5]

# Sine-wave baseline at each length (parameters tuned at N=5). Add values as you measure them.
# SINE = {5: 8.4, 8: 4.1}

# Centralized run names at each length ({s} gets replaced by the seed number), with each naming convention i picked for each bc i didn't stay consistent ;/
CENTRAL_RUNS = {
    3: "n3_central_s{s}",
    5: "n5_central_e3_s{s}",
    8: "centraln8_s{s}",
    10: "n10_central_e3_s{s}"
}


def load_final_scores(pattern):
    # Load the 10-episode final_score for each seed; skip if the files don't exist yet
    files = [f"{pattern.format(s=s)}_curves.npz" for s in SEEDS]
    if not all(os.path.exists(f) for f in files):
        return None
    return np.array([float(np.load(f)["final_score"]) for f in files])


# Modular: N=5 from evolution, other lengths from transfer files
modular = {5: load_final_scores("n5_mod_e3_s{s}")}
for f in args.files:
    data = np.load(f)
    for key in data.files:              # keys look like "N8", "N3"
        modular[int(key[1:])] = data[key]

# Centralized: each length evolved separately
central = {}
for n, pattern in CENTRAL_RUNS.items():
    scores = load_final_scores(pattern)
    if scores is not None:
        central[n] = scores

# Keep only the requested lengths (always keeping N=5 as the reference)
if args.lengths is not None:
    keep = set(args.lengths) | {5}
    modular = {n: s for n, s in modular.items() if n in keep}
    central = {n: s for n, s in central.items() if n in keep}

OFFSET = 0.1   # nudge the two groups apart so their dots don't overlap for readability

plt.figure(figsize=(7, 5))


def plot_group(data, color, marker, label, shift):
    lengths = sorted(data)
    xs = [n + shift for n in lengths]
    for x, n in zip(xs, lengths):
        plt.scatter([x] * len(data[n]), data[n], color=color, alpha=0.4, s=25)
    plt.errorbar(xs, [data[n].mean() for n in lengths], yerr=[data[n].std() for n in lengths],
                 color=color, marker=marker, markersize=8, linewidth=2, capsize=5, linestyle="none", label=label)


plot_group(modular, "steelblue", "o", "Modular (evolved at N=5, transferred)", -OFFSET)
plot_group(central, "darkorange", "s", "Centralized (evolved at each N)", +OFFSET)

# Sine baseline, only at lengths that are being plotted
# all_lengths = sorted(set(modular) | set(central))
# sine_n = [n for n in all_lengths if n in SINE]
# if sine_n:
#     plt.plot(sine_n, [SINE[n] for n in sine_n], color="red", linestyle=":", marker="x",
#              markersize=8, linewidth=2, label="Sine wave (tuned at N=5)")

plt.axvline(5, color="gray", linestyle="--", alpha=0.5, label="Length modular genome evolved at")
plt.xlabel("Body length N (segments)")
plt.ylabel("Distance traveled (units)")
plt.title(args.title)
plt.xticks(sorted(set(modular) | set(central)))
plt.grid(alpha=0.3)
plt.legend(loc="lower right")
plt.savefig(args.output, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {args.output}")