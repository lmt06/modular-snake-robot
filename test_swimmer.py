# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import time
# Create our training environment - swimmer
env = gym.make('Swimmer-v5', xml_file= './swimmer_n5.xml', render_mode="human", ctrl_cost_weight=0)

# Reset environment to start a new episode
observation, info = env.reset()
# observation: 
    #qpos (3 elements by default): Position values of the robot’s body parts.
    #qvel (5 elements): The velocities of these individual body parts (their derivatives).
    #By default, the observation does not include the x- and y-coordinates of the front tip. These can be included by passing exclude_current_positions_from_observation=False during construction

# info: extra debugging information (usually not needed for basic learning)

#Parameters of snake: 
    # n: number of body parts
    # mi: mass of part i (i ∈ {1…n})
    # li: length of part i (i ∈ {1…n})
    # k: viscous-friction coefficient

print(f"Starting observation: {observation}")
# Example output: [ -0.04890713 -0.04682811 -0.04984891  0.09476142  0.04318762  0.00018765, 0.07390617 -0.07554847]
# [position values of robots body parts (3), velocity of individual body parts (5)]

episode_over = False
total_reward = 0

for step in range(1000):
    # Choose an action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()  #Picks random valid action, here is picking random value for each torque (four values each between -1 and 1) 

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated : swimmer will never terminate
    # truncated: True if we hit the time limit (500 steps)
    if terminated or truncated:
        total_reward = 0
        observation, info = env.reset() #env.reset resets our world, and we set observation and info with it that way our sensors don't see a fallen poll for a frame
    total_reward += reward
    time.sleep(0.02)
    # episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()

#note removing terminated lets you see pole fall because physics keeps moving even after episode has ended