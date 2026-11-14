# plot_curves.py
# Plots fitness over generations from saved *_curves.npz files

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", help="run names, e.g. central_s1 central_s2")
parser.add_argument("--output", type=str, default="fitness_curves.png")
parser.add_argument("--title", type=str, default="Fitness over generations")
parser.add_argument("--sine", type=float, default=8.68, help="sine-wave distance (reference line)")
args = parser.parse_args()

plt.figure(figsize=(8, 5))

if len(args.runs) == 1:
    # One run: show best, mean, and worst of the population
    data = np.load(f"{args.runs[0]}_curves.npz")
    gens = np.arange(1, len(data["best"]) + 1)
    plt.plot(gens, data["best"], label="Best", linewidth=2)
    plt.plot(gens, data["mean"], label="Mean", linewidth=2)
    plt.plot(gens, data["worst"], label="Worst", linewidth=1, linestyle="--")
else:
    # Several runs: each run's best curve faintly, plus the average across runs
    all_best = []
    for name in args.runs:
        data = np.load(f"{name}_curves.npz")
        all_best.append(data["best"])
        gens = np.arange(1, len(data["best"]) + 1)
        plt.plot(gens, data["best"], alpha=0.5, linewidth=1, label=name)        #, label=name) added to this line it adds seeds to legend, but this makes plot look weird so just screenshotting and then removing to add next to plot
    all_best = np.array(all_best)
    mean_best = all_best.mean(axis=0)
    std_best = all_best.std(axis=0)
    plt.plot(gens, mean_best, color="black", linewidth=2, label=f"Mean best ({len(args.runs)} seeds)")
    plt.fill_between(gens, mean_best - std_best, mean_best + std_best, color="gray", alpha=0.3, label="± 1 std")

# Reference line for the hand-designed controller at n=5
# plt.axhline(args.sine, color="red", linestyle=":", linewidth=2, label=f"Sine wave ({args.sine:.2f})")

plt.xlabel("Generation")
plt.ylabel("Distance traveled (units)")
plt.title(args.title)

# ymin, ymax = plt.ylim()
# plt.ylim(ymin, ymax + 0.3 * (ymax - ymin))      #increases y axis slightly so large legends don't cover curves

plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(args.output, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {args.output}")