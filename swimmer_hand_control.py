import gymnasium as gym
import time
import numpy as np
import imageio

FRAME_SKIP = 2      # keep every 2nd frame (smaller file, still smooth)
SECONDS = 20        # how much of the episode to record (the gait repeats, so 20 s is plenty)


# Create our training environment - swimmer
# env = gym.make('Swimmer-v5', xml_file= './swimmer_n8.xml', render_mode="human", ctrl_cost_weight=0)

#this env is the one that makes gif possible, above is the popup env
env = gym.make('Swimmer-v5', xml_file='./swimmer_n5.xml', ctrl_cost_weight=0,
               render_mode="rgb_array", camera_name="track")


# Reset environment to start a new episode
observation, info = env.reset()
# observation: 
    #qpos (3 elements by default): Position values of the robot’s body parts.
    #qvel (5 elements): The velocities of these individual body parts (their derivatives).
    #By default, the observation does not include the x- and y-coordinates of the front tip. These can be included by passing exclude_current_positions_from_observation=False during construction

# info: extra debugging information (usually not needed for basic learning)
x_start = env.unwrapped.data.qpos[0]
#tells us starting position

#Parameters of snake: 
    # n: number of body parts
    # mi: mass of part i (i ∈ {1…n})
    # li: length of part i (i ∈ {1…n})
    # k: viscous-friction coefficient

print(f"Starting observation: {observation}")
# Example output: [ -0.04890713 -0.04682811 -0.04984891  0.09476142  0.04318762  0.00018765, 0.07390617 -0.07554847] (observation output for 3 segment snake)
# [position values of robots body parts (5), velocity of individual body parts (7)] (generalized observation gives N position values and (N+2) velocities, so observation is 2N+2 length)

episode_over = False
total_reward = 0


    
    # action = env.action_space.sample()  #Picks random valid action, here is picking random value for each torque (four values each between -1 and 1) 
frequency=.5
amplitude = 1

max_steps = int(SECONDS / env.unwrapped.dt)
frames = []

#action for N=5
for step in range(max_steps):
    t = env.unwrapped.dt * step
    action = [amplitude*np.sin(2*(np.pi)*frequency*(env.unwrapped.dt*step)), #front joint motion
              amplitude*np.sin(2*(np.pi)*frequency*(env.unwrapped.dt*step) -np.pi/2),
              amplitude*np.sin(2*(np.pi)*frequency*(env.unwrapped.dt*step) - np.pi),
              amplitude*np.sin(2*(np.pi)*frequency*(env.unwrapped.dt*step)-3*np.pi/2)]
        
        # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step % FRAME_SKIP == 0:
        frames.append(env.render())        # grab this frame as an image

    if terminated or truncated:
        break
        

    # #action for N=8
    # action = [amplitude*np.sin(2*(np.pi)*frequency*(env.unwrapped.dt*step)), #front joint motion
    #           amplitude*np.sin(2*(np.pi)*frequency*(env.unwrapped.dt*step) -np.pi/4),
    #           amplitude*np.sin(2*(np.pi)*frequency*(env.unwrapped.dt*step) - np.pi/2),
    #           amplitude*np.sin(2*(np.pi)*frequency*(env.unwrapped.dt*step)-3*np.pi/4),
    #           amplitude*np.sin(2*(np.pi)*frequency*(env.unwrapped.dt*step) - np.pi),
    #           amplitude*np.sin(2*(np.pi)*frequency*(env.unwrapped.dt*step) - 5*np.pi/4),
    #           amplitude*np.sin(2*(np.pi)*frequency*(env.unwrapped.dt*step) - 3*np.pi/2),]

    # n_joints = env.action_space.shape[0]          # 7 for N=8, 4 for N=5, works for any body
    # phase_lag = 2 * np.pi / n_joints               # one full wavelength along the body
    # t = env.unwrapped.dt * step
    # action = [amplitude * np.sin(2 * np.pi * frequency * t - i * phase_lag) for i in range(n_joints)]


    


    # reward: +1 for each step the pole stays upright
    # terminated : swimmer will never terminate
    # truncated: True if we hit the time limit (500 steps)
#     if terminated or truncated:
#         print(f"Episode finished! Total reward: {total_reward}")
#         total_reward = 0
#         x_end = env.unwrapped.data.qpos[0]
#         print(f"Distance traveled: {x_end - x_start:.2f}")
#         observation, info = env.reset() 
    
#     time.sleep(0.02)
#     # episode_over = terminated or truncated


# env.close()
x_end = env.unwrapped.data.qpos[0]
print(f"Distance traveled in {SECONDS} s: {x_end - x_start:.2f}")

# Save the GIF, playing back at real speed and looping forever
frame_ms = 1000 * env.unwrapped.dt * FRAME_SKIP
imageio.mimsave("sine_n5.gif", frames, duration=frame_ms, loop=0)
print(f"Saved sine_n5.gif ({len(frames)} frames)")