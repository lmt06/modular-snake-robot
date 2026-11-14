import gymnasium as gym
import numpy as np
from controller import get_inputs, genome_size, decode, act

# Make the 5-segment swimmer (no rendering needed for a test)
env = gym.make('Swimmer-v5', xml_file='./swimmer_n5.xml')
observation, info = env.reset()

# Take a few random steps so the numbers aren't all tiny
for step in range(20):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

# --- Test get_inputs ---
inputs = get_inputs(observation)
print("Inputs shape:", inputs.shape)            # expect (8,)

# --- Test genome_size and decode ---
print("Genome size:", genome_size())             # expect 108
genome = np.random.uniform(-1, 1, genome_size())  # a random stand-in genome
w1, b1, w2, b2 = decode(genome)
print("w1 shape:", w1.shape)                      # expect (8, 8)
print("b1 shape:", b1.shape)                      # expect (8,)
print("w2 shape:", w2.shape)                      # expect (4, 8)
print("b2 shape:", b2.shape)                      # expect (4,)

# --- Test act ---
torques = act(w1, b1, w2, b2, inputs)
print("Torques:", torques)                        # expect 4 numbers between -1 and 1
print("Torques shape:", torques.shape)            # expect (4,)

from controller import get_inputs_modular, genome_size_modular, decode_modular, act_modular

mod_genome = np.random.uniform(-1, 1, genome_size_modular())
mw1, mb1, mw2, mb2 = decode_modular(mod_genome)
print("Modular genome size:", genome_size_modular())          # expect 49
print("Shapes:", mw1.shape, mb1.shape, mw2.shape, mb2.shape)  # expect (8, 4) (8,) (1, 8) (1,)

mod_inputs = get_inputs_modular(observation)
print("Modular inputs shape:", mod_inputs.shape)               # expect (4, 4)
print(mod_inputs)                                              # check the zeros at the ends

mod_torques = act_modular(mw1, mb1, mw2, mb2, mod_inputs)
print("Modular torques:", mod_torques)                         # expect 4 numbers in [-1, 1]
print("Modular torques shape:", mod_torques.shape)             # expect (4,)

env.close()

