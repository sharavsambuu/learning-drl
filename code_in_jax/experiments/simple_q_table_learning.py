import gymnasium as gym
import numpy as np
from time import sleep

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")

print("Total states:", env.observation_space.n)
print("Actions:", env.action_space.n)
print("Starting training...")

Q             = np.zeros([env.observation_space.n, env.action_space.n])
learning_rate = 0.8
gamma         = 0.95
episodes      = 4000
reward_list   = []

for i in range(episodes):
    state        = env.reset()[0]
    total_reward = 0
    done         = False

    while not done:
        # Choose action with exploration-exploitation trade-off
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1.0 / (i + 1)))

        # Take the chosen action and observe the new state and reward
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update Q-table using the Q-learning update rule
        Q[state, action] = Q[state, action] + learning_rate * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

        total_reward += reward
        state         = new_state

        env.render()
        print(f"Episode: {i + 1}")
        print(f"Action: {action}")
        print("Q-table:")
        print(Q)
        #sleep(0.001)

    reward_list.append(total_reward)

print("Training finished.")
print("Average score:", str(sum(reward_list) / episodes))
print("Q-table (final):")
print(Q)