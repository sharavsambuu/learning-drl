import os
import random
import math
import gym
from collections import deque


debug_render  = True
num_episodes  = 100
memory_length = 4000
replay_memory = deque(maxlen=memory_length)


env = gym.make('CartPole-v0')
env.reset()

n_actions = env.action_space.n


global_steps = 0
try:
    for episode in range(num_episodes):
        state = env.reset()
        while True:
            global_steps = global_steps+1
            action       = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            if done:
                new_state = None
            replay_memory.append((state, action, reward, new_state))

            # replay логик энд бичигдэнэ

            state = new_state
            if debug_render:
                env.render()
            if done:
                break
            print(len(replay_memory))
finally:
    env.close()
