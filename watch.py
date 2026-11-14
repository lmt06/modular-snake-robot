# watch.py
# Loads a saved genome and renders the snake swimming with it

import argparse
import time
import numpy as np
import gymnasium as gym
from controller import (get_inputs, decode, act,
                        get_inputs_modular, decode_modular, act_modular)
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("genome", help="saved genome file, e.g. mod_s1.npy")
parser.add_argument("--controller", type=str, default="modular", choices=["central", "modular"])
parser.add_argument("--xml", type=str, default="./swimmer_n5.xml", help="body file to swim with")
parser.add_argument("--slow", type=float, default=0.02, help="seconds to pause each step (0 = full speed)")
parser.add_argument("--gif", type=str, default=None, help="save a GIF instead of showing a window, e.g. mod_s1.gif")
parser.add_argument("--frame_skip", type=int, default=2, help="keep every Nth frame in the GIF (smaller file)")         #note: for best results keep frame skip between 3-4
args = parser.parse_args()

# Pick the matching controller functions
if args.controller == "central":
    decode_fn, inputs_fn, act_fn = decode, get_inputs, act
else:
    decode_fn, inputs_fn, act_fn = decode_modular, get_inputs_modular, act_modular

# Load the genome and turn it into weights
genome = np.load(args.genome)
w1, b1, w2, b2 = decode_fn(genome)

# Make the environment WITH rendering this time
render_mode = "rgb_array" if args.gif else "human"
env = gym.make('Swimmer-v5', xml_file=args.xml, ctrl_cost_weight=0,
               render_mode=render_mode, camera_name="track")            #uses command from xml file so camera tracks snake from above
observation, info = env.reset()
x_start = env.unwrapped.data.qpos[0]

# Swim one full episode
# while True:
#     inputs = inputs_fn(observation)
#     torques = act_fn(w1, b1, w2, b2, inputs)
#     observation, reward, terminated, truncated, info = env.step(torques)
#     time.sleep(args.slow)
#     if terminated or truncated:
#         break

#version of episode loop that allows us to grab gifs
frames = []
step = 0
while True:
    inputs = inputs_fn(observation)
    torques = act_fn(w1, b1, w2, b2, inputs)
    observation, reward, terminated, truncated, info = env.step(torques)

    if args.gif:
        if step % args.frame_skip == 0:
            frames.append(env.render())     # grab this frame as an image
    else:
        time.sleep(args.slow)

    step += 1
    if terminated or truncated:
        break

x_end = env.unwrapped.data.qpos[0]
print(f"Distance traveled: {x_end - x_start:.3f}")

if args.gif:
    fps = int(1 / (env.unwrapped.dt * args.frame_skip))   # play back at real speed
    imageio.mimsave(args.gif, frames, fps=fps)
    print(f"Saved {args.gif} ({len(frames)} frames)")

env.close()