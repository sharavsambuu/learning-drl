import gymnasium as gym
import gymnasium_robotics  # registers Fetch* envs
import numpy as np


class TransformerObservationWrapper(gym.ObservationWrapper):
    """
    Dict observation -> flat float32 vector.
    Structure: [observation (25)] + [desired_goal (3)] = 28
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low   =-np.inf,
            high  = np.inf,
            shape = (28,),
            dtype = np.float32,
        )

    def observation(self, obs_dict):
        return np.concatenate(
            [obs_dict["observation"], obs_dict["desired_goal"]],
            dtype=np.float32,
        )


class ActionTokenizer:
    """
    Continuous [-1,1] <-> Discrete tokens [0, bins-1]
    """
    def __init__(self, bins=256):
        self.bins = int(bins)

    def encode(self, continuous_action):
        clipped = np.clip(continuous_action, -1.0, 1.0)
        norm    = (clipped + 1.0) / 2.0
        return (norm * (self.bins - 1)).astype(np.int64)

    def decode(self, tokens):
        tokens = np.asarray(tokens, dtype=np.float32)
        norm   = tokens / (self.bins - 1)
        return (norm * 2.0) - 1.0


class TokenActionWrapper(gym.ActionWrapper):
    """
    Exposes a discrete MultiDiscrete action space (token vector),
    decodes to continuous Box action for the underlying env.
    """
    def __init__(self, env, bins=256):
        super().__init__(env)
        self.tokenizer    = ActionTokenizer(bins=bins)
        n                 = int(np.prod(env.action_space.shape))
        self.action_space = gym.spaces.MultiDiscrete([bins] * n)

    def action(self, token_action):
        a = self.tokenizer.decode(token_action).astype(np.float32)
        return a


class SequentialGoalsWrapper(gym.Wrapper):
    """
    Turns FetchPush into a pinpad-like sequential task:
      - sample K goals per episode
      - agent must achieve them in order
      - reward is sparse: +1 only when final goal is achieved, else 0
      - episode ends when final goal achieved or time limit reached

    Extra:
      - Sets obs["desired_goal"] to the CURRENT subgoal each step (not just at reset).
      - Keeps info["distance_threshold"] in sync with goal_threshold for debugging.
    """
    def __init__(self, env, k=3, goal_threshold=0.05, sparse_final_reward=True):
        super().__init__(env)
        self.k                   = int(k)
        self.goal_threshold      = float(goal_threshold)
        self.sparse_final_reward = bool(sparse_final_reward)

        self._goals              = None
        self._goal_idx           = 0

    def _set_current_goal_in_obs(self, obs):
        obs                    = dict(obs)
        obs ["desired_goal"  ] = self._goals[self._goal_idx].copy()
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Sample K goals from unwrapped goal sampler when available.
        goals = []
        if hasattr(self.env.unwrapped, "_sample_goal"):
            for _ in range(self.k):
                g = np.asarray(self.env.unwrapped._sample_goal(), dtype=np.float32)
                goals.append(g)
        else:
            base = np.asarray(obs["desired_goal"], dtype=np.float32)
            for _ in range(self.k):
                goals.append(base + np.random.uniform(-0.05, 0.05, size=base.shape).astype(np.float32))

        self._goals    = goals
        self._goal_idx = 0

        obs  = self._set_current_goal_in_obs(obs)

        info                   = dict(info)
        info["seq_goal_index"] = self._goal_idx
        info["seq_goals"     ] = np.stack(self._goals, axis=0)
        info["seq_reached"   ] = False
        info["seq_dist"      ] = None
        info["distance_threshold"] = self.goal_threshold

        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        achieved     = np.asarray(obs["achieved_goal"], dtype=np.float32)
        current_goal = self._goals[self._goal_idx]
        dist         = float(np.linalg.norm(achieved - current_goal))

        reached      = dist <= self.goal_threshold

        info                       = dict(info)
        info["seq_goal_index"    ] = self._goal_idx
        info["seq_reached"       ] = reached
        info["seq_dist"          ] = dist
        info["distance_threshold"] = self.goal_threshold

        if reached:
            if self._goal_idx < self.k - 1:
                self._goal_idx += 1
                obs = self._set_current_goal_in_obs(obs)

                info["seq_goal_index"] = self._goal_idx
                info["is_success"    ] = 0.0

                reward = 0.0 if self.sparse_final_reward else 0.1
                return obs, reward, False, truncated, info
            else:
                info["is_success"] = 1.0
                reward             = 1.0
                return obs, reward, True, truncated, info

        obs = self._set_current_goal_in_obs(obs)

        info["is_success"] = 0.0
        reward = 0.0 if self.sparse_final_reward else float(base_reward)
        return obs, reward, terminated, truncated, info


def print_env_specs(env):
    print("\n" + "=" * 60)
    print(f"ENV: {env.unwrapped.spec.id if env.unwrapped.spec else 'Unknown'}")
    print("=" * 60)
    print(f"Action Space (tokenized): {env.action_space}")
    o, info = env.reset(seed=0)
    if isinstance(o, dict):
        print("Observation is dict (unexpected after wrapper).")
    else:
        print(f"Observation (flat) shape: {o.shape}, dtype={o.dtype}")
        print(f"Sample obs[:5]: {o[:5]}")
    print("=" * 60 + "\n")


def run_rollout_eps_greedy(env, eps=0.2, seed=None):
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps        = 0

    mid = (env.action_space.nvec // 2).astype(np.int64)

    while True:
        if np.random.rand() < eps:
            action_tokens = env.action_space.sample()
        else:
            action_tokens = mid.copy()

        obs, reward, terminated, truncated, info = env.step(action_tokens)

        total_reward += float(reward)
        steps += 1

        if terminated or truncated:
            print(
                f"steps={steps} total_reward={total_reward:.3f} "
                f"is_success={info.get('is_success', 0.0)} "
                f"seq_goal_index={info.get('seq_goal_index', -1)}"
            )
            return


def run_rollout_macro(env, macro_step_size=20, seed=None):
    """
    Simple macro-step exploration: sample one token-action, hold it for macro_step_size steps.
    Helps exploration vs per-step jitter in sparse, long-horizon tasks.
    """
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps        = 0

    while True:
        action_tokens = env.action_space.sample()

        for _ in range(int(macro_step_size)):
            obs, reward, terminated, truncated, info = env.step(action_tokens)

            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                print(
                    f"[MACRO] steps={steps} total_reward={total_reward:.3f} "
                    f"is_success={info.get('is_success', 0.0)} "
                    f"seq_goal_index={info.get('seq_goal_index', -1)}"
                )
                return


if __name__ == "__main__":
    # Override the baked-in TimeLimit by passing max_episode_steps here if needed.
    raw_env = gym.make("FetchPush-v4", render_mode="human", max_episode_steps=200)

    # Add sequential subgoals (pinpad-like) + sparse final reward
    raw_env = SequentialGoalsWrapper(raw_env, k=3, goal_threshold=0.05, sparse_final_reward=True)

    # Tokenize actions -> MultiDiscrete, then flatten obs -> 28D float32 vector
    env     = TokenActionWrapper(raw_env, bins=256)
    env     = TransformerObservationWrapper(env)

    print_env_specs(env)

    print("Running epsilon-greedy rollouts (random tokens with prob eps)...")
    for i in range(10):
        run_rollout_eps_greedy(env, eps=0.2, seed=i)

    print("Running macro-step rollouts (hold random token-action for K steps)...")
    for i in range(5):
        run_rollout_macro(env, macro_step_size=20, seed=100 + i)

    env.close()
