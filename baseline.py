#baseline.py

#this file generates 100 random genomes to get a random baseline as outlined in milestone 3

import numpy as np
import matplotlib
matplotlib.use("Agg")   # save plots to files without opening a window (no tkinter needed)
import matplotlib.pyplot as plt
import argparse
from controller import genome_size, genome_size_modular
from fitness import evaluate, get_env


def generate_genomes(n, length):
    return np.random.uniform(-1, 1, (n, length))
    #generates a number of random genomes given by the n passed in and based on the size configured in controller.py

# def score_genomes(genomes):
#     scores= []

#     counter=0
#     for genome in genomes:  #takes single row of 2d array and evaluates using evaluate() in fitness.py and adds to array of scores
#         counter+=1
#         score=evaluate(genome)
#         scores.append(score)
#         if counter%10==0:
#             print(f"{counter}/{len(genomes)} done, latest score: {score:.3f}")  #prints progress (how many of total done) and latest score

#     return np.array(scores) #make np array from python list so easier to do mean and std calculations later

def score_genomes(genomes, controller, xml_file):
    scores = []
    counter = 0
    for genome in genomes:
        counter += 1
        score = evaluate(genome, controller=controller, xml_file=xml_file)
        scores.append(score)
        if counter % 10 == 0:
            print(f"{counter}/{len(genomes)} done, latest score: {score:.3f}")
    return np.array(scores)

def summarize(data):
    print(f"Number of genomes: {len(data)}")
    print(f"Mean distance:     {data.mean():.3f}")
    print(f"Std deviation:     {data.std():.3f}")
    print(f"Best distance:     {data.max():.3f}")
    print(f"Worst distance:    {data.min():.3f}")
    print(f"Moved forward:     {(data > 0).mean() * 100:.1f}%")


def plot_histogram(scores, title, filename, reference_lines=None, xlabel="Distance traveled (units)"):
    # reference_lines: a dictionary of {label: value}, e.g. {"Sine wave": 8.68}
    
    plt.figure(figsize=(8, 5)) #sets figure dimension

    # The histogram itself
    plt.hist(scores, bins=20, color="steelblue", edgecolor="black", alpha=0.8, label="Random genomes")

    # Vertical lines for any reference values
    if reference_lines is not None:
        line_styles = ["--", ":", "-.", "-"]
        for i, (label, value) in enumerate(reference_lines.items()):
            plt.axvline(value, linestyle=line_styles[i % len(line_styles)],
                        linewidth=2, label=f"{label} ({value:.2f})")

    # Labels so the figure stands on its own
    plt.xlabel(xlabel)
    plt.ylabel("Number of genomes")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="central", choices=["central", "modular"])
    parser.add_argument("--xml", type=str, default="./swimmer_n5.xml")
    parser.add_argument("--n", type=int, default=100, help="number of random genomes")
    parser.add_argument("--sine", type=float, default=None, help="sine-wave distance at this length (optional reference line)")
    parser.add_argument("--output", type=str, default="baseline_central_n5")
    args = parser.parse_args()

    np.random.seed(42)

    # Genome length depends on the controller (and, for centralized, the body)
    n_joints = get_env(args.xml).action_space.shape[0]
    length = genome_size(n_joints) if args.controller == "central" else genome_size_modular()
    print(f"{args.controller} controller on {args.xml}: {n_joints} joints, genome length {length}")

    genomes = generate_genomes(args.n, length)

    start = time.time()
    scores = score_genomes(genomes, args.controller, args.xml)
    print(f"Scoring took {time.time() - start:.1f} seconds")

    np.save(f"{args.output}_scores.npy", scores)
    summarize(scores)

    refs = {"Sine wave": args.sine} if args.sine is not None else None
    plot_histogram(scores, f"Random genome baseline: {args.controller}, {args.xml}",
                   f"{args.output}_hist.png", reference_lines=refs)