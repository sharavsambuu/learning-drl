import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv
import threading
import time
import numpy as np


class SparseCartPoleEnv(CartPoleEnv):
    def __init__(self, reward_threshold=200, **kwargs):
        super().__init__(**kwargs)
        self.step_count = 0
        self.reward_threshold = reward_threshold

    def step(self, action):
        self.step_count += 1
        obs, _, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        reward = 1 if done and self.step_count >= self.reward_threshold else 0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        obs, info = super().reset(**kwargs)
        return obs, info


def run_episode(thread_id, reward_list, episode_length_list, reward_threshold=200):
    """Runs a single episode of SparseCartPoleEnv in a thread."""
    env = SparseCartPoleEnv(reward_threshold=reward_threshold)  # Create *separate* env instance
    obs, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0
    steps = 0
    while not (terminated or truncated):
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps        += 1
    env.close() 
    reward_list.append(total_reward)
    episode_length_list.append(steps)
    print(f"Thread {thread_id} - Episode finished with reward: {total_reward}, Length: {steps}")

if __name__ == "__main__":
    num_threads         = 4  
    episodes_per_thread = 100 

    all_threads         = []
    all_rewards         = []
    all_episode_lengths = []

    start_time = time.time()

    for _ in range(episodes_per_thread): 

        threads                   = []
        rewards_for_batch         = [] # collect rewards from threads in this batch
        episode_lengths_for_batch = [] # collect episode lengths

        for i in range(num_threads):
            thread = threading.Thread(target=run_episode, args=(i, rewards_for_batch, episode_lengths_for_batch, 200)) 
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join() 

        all_rewards.extend(rewards_for_batch) # Collect rewards
        all_episode_lengths.extend(episode_lengths_for_batch)

    end_time     = time.time()
    elapsed_time = end_time - start_time

    print("\n--- Summary ---")
    print(f"Total Episodes Run (Parallel) : {episodes_per_thread * num_threads}")
    print(f"Total Time                    : {elapsed_time:.2f} seconds")
    print(f"Average Reward                : {np.mean(all_rewards):.2f}")
    print(f"Average Episode Length        : {np.mean(all_episode_lengths):.2f}")

    print("\n--- Sequential Run (for comparison) ---")
    start_time_seq = time.time()
    all_rewards_seq = []
    all_episode_lengths_seq = []
    for _ in range(episodes_per_thread * num_threads):
        env_seq = SparseCartPoleEnv(reward_threshold=200)
        obs_seq, info_seq = env_seq.reset()
        terminated_seq, truncated_seq = False, False
        total_reward_seq = 0
        steps_seq = 0
        while not (terminated_seq or truncated_seq):
            action_seq = env_seq.action_space.sample()
            obs_seq, reward_seq, terminated_seq, truncated_seq, info_seq = env_seq.step(action_seq)
            total_reward_seq += reward_seq
            steps_seq += 1
        env_seq.close()
        all_rewards_seq.append(total_reward_seq)
        all_episode_lengths_seq.append(steps_seq)
    end_time_seq = time.time()
    elapsed_time_seq = end_time_seq - start_time_seq
    print(f"Total Time (Sequential)             : {elapsed_time_seq:.2f} seconds")
    print(f"Average Reward (Sequential)         : {np.mean(all_rewards_seq):.2f}")
    print(f"Average Episode Length (Sequential) : {np.mean(all_episode_lengths_seq):.2f}")

    if elapsed_time < elapsed_time_seq:
        speedup = elapsed_time_seq / elapsed_time
        print(f"\nParallel execution was faster by {speedup:.2f}x")
    else:
        print("\nSequential execution was faster (or no significant speedup).")