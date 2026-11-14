# fitness.py
# Runs the snake with a given genome and returns how far it swam

import gymnasium as gym
import numpy as np
from controller import (get_inputs, decode, act, genome_size,
                        get_inputs_modular, decode_modular, act_modular, genome_size_modular)

# Environments are created once per body file and reused (creating one is slow)
_envs = {}

def get_env(xml_file):
    if xml_file not in _envs:
        _envs[xml_file] = gym.make('Swimmer-v5', xml_file=xml_file, ctrl_cost_weight=0)
    return _envs[xml_file]


def evaluate(genome, n_episodes=3, controller="central", xml_file="./swimmer_n5.xml"):
    
    n_joints = get_env(xml_file).action_space.shape[0]
    # Pick the matching set of controller functions
    if controller == "central":
        decode_fn, inputs_fn, act_fn = decode, get_inputs, act
    elif controller == "modular":
        decode_fn, inputs_fn, act_fn = decode_modular, get_inputs_modular, act_modular
    else:
        raise ValueError(f"Unknown controller: {controller}")

    env = get_env(xml_file)

    # 1. Decode the genome into weights (once, before any episodes)
    if controller == "central":
        w1, b1, w2, b2 = decode(genome, n_joints)
    else:
        w1, b1, w2, b2 = decode_modular(genome)

    # 2. Collect each episode's distance here
    distances = []

    # 3. Run each episode
    for episode in range(n_episodes):
        observation, info = env.reset()
        x_start = env.unwrapped.data.qpos[0]

        while True:
            inputs = inputs_fn(observation)
            torques = act_fn(w1, b1, w2, b2, inputs)
            observation, reward, terminated, truncated, info = env.step(torques)
            if terminated or truncated:
                break

        x_end = env.unwrapped.data.qpos[0]
        distances.append(x_end - x_start)

    # 4. Return the average distance across episodes
    return np.mean(distances)


# Quick test: only runs when you run this file directly
if __name__ == "__main__":
    central_genome = np.random.uniform(-1, 1, genome_size())
    print(f"Centralized random genome: {evaluate(central_genome):.3f}")

    modular_genome = np.random.uniform(-1, 1, genome_size_modular())
    print(f"Modular random genome:     {evaluate(modular_genome, controller='modular'):.3f}")