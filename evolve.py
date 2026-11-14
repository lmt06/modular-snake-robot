# evolve.py
# Evolves controller genomes with EvoTorch's genetic algorithm,
# using evaluate() from fitness.py as the fitness function

import argparse
import time
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver

from controller import genome_size, genome_size_modular      # NEW: modular size
from fitness import evaluate, get_env


# def make_fitness_fn(n_episodes, controller):                  # NEW: controller argument
#     # EvoTorch hands each genome over as a PyTorch tensor,
#     # so convert it to NumPy before passing it to evaluate()
#     def fitness_fn(genome_tensor):
#         genome = genome_tensor.detach().cpu().numpy()
#         return float(evaluate(genome, n_episodes=n_episodes, controller=controller))   # NEW
#     return fitness_fn

def make_fitness_fn(n_episodes, controller, xml_file):
    def fitness_fn(genome_tensor):
        genome = genome_tensor.detach().cpu().numpy()
        return float(evaluate(genome, n_episodes=n_episodes, controller=controller, xml_file=xml_file))
    return fitness_fn


def run_evolution(seed, popsize, gens, n_episodes, mut_stdev, controller, xml_file):   # NEW
    np.random.seed(seed)
    torch.manual_seed(seed)

    # NEW: genome length depends on which controller is being evolved and the body being used
    n_joints = get_env(xml_file).action_space.shape[0]
    length = genome_size(n_joints) if controller == "central" else genome_size_modular()

    problem = Problem(
        "max",
        make_fitness_fn(n_episodes, controller, xml_file),               # NEW
        solution_length=length,                                # NEW
        initial_bounds=(-1.0, 1.0),
        seed=seed,
    )

    searcher = GeneticAlgorithm(
        problem,
        popsize=popsize,
        operators=[
            SimulatedBinaryCrossOver(problem, tournament_size=3, eta=20),
            GaussianMutation(problem, stdev=mut_stdev),
        ],
    )

    best_curve, mean_curve, worst_curve = [], [], []

    for gen in range(gens):
        searcher.step()
        evals = searcher.population.evals.flatten().cpu().numpy()
        best_curve.append(evals.max())
        mean_curve.append(evals.mean())
        worst_curve.append(evals.min())
        print(f"Gen {gen + 1}/{gens}  best: {evals.max():.3f}  "
              f"mean: {evals.mean():.3f}  worst: {evals.min():.3f}")

    best_genome = searcher.status["best"].values.detach().cpu().numpy()
    return best_genome, np.array(best_curve), np.array(mean_curve), np.array(worst_curve)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="central",
                        choices=["central", "modular"])         # NEW
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--popsize", type=int, default=30)
    parser.add_argument("--gens", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--mut_stdev", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="run")
    parser.add_argument("--xml", type=str, default="./swimmer_n5.xml")
    args = parser.parse_args()

    print(f"Evolving {args.controller} controller, seed {args.seed}, popsize {args.popsize}")   # NEW

    start = time.time()
    best_genome, best_curve, mean_curve, worst_curve = run_evolution(
        args.seed, args.popsize, args.gens, args.episodes, args.mut_stdev, args.controller, args.xml)
    print(f"Evolution took {(time.time() - start) / 60:.1f} minutes")

    # Re-evaluate the best genome over more episodes for a fair final score
    final_score = evaluate(best_genome, n_episodes=10, controller=args.controller, xml_file=args.xml)
    print(f"Best genome re-evaluated over 10 episodes: {final_score:.3f}")

    np.save(f"{args.output}.npy", best_genome)
    np.savez(f"{args.output}_curves.npz", best=best_curve, mean=mean_curve,
             worst=worst_curve, final_score=final_score)
    print(f"Saved {args.output}.npy and {args.output}_curves.npz")