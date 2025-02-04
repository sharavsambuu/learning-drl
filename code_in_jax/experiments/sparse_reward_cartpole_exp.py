import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv

class SparseCartPoleEnv(CartPoleEnv):
    def __init__(self, reward_threshold=200, **kwargs):
        super().__init__(**kwargs)
        self.step_count = 0
        self.reward_threshold = reward_threshold

    def step(self, action):
        self.step_count += 1
        obs, _, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        # Sparse reward: reward only if episode ended and lasted at least reward_threshold steps.
        reward = 1 if done and self.step_count >= self.reward_threshold else 0
        # Alternative: Reward based on success/failure
        # if done:
        #     reward = 1 if self.step_count >= self.spec.max_episode_steps else -1
        # else:
        #     reward = 0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        obs, info = super().reset(**kwargs) # unpack the return value of reset()
        return obs, info

if __name__ == "__main__":
    env = SparseCartPoleEnv(render_mode="human")
    episodes = 5

    for ep in range(episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        total_reward = 0
        while not (terminated or truncated):
            action = env.action_space.sample()  # random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        print(f"Episode {ep+1} finished with reward: {total_reward}")  # Print total reward
    env.close()