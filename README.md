# About

    This repo is just my learning journey and may contain a buggy naive implementations.


# Tasks

    - Fix TRPO
    - Fix PPO
    - Apply GRPO to PPO


# Other things I will learn in the near future

    - Align some small LLMs to some preference texts using PPO
    - Try to build minimum viable reasoner, using GRPO of DeepSeek R1


# Algo names and intuition table


| Algorithm Name                     | Intuitive Summary                                                                                                                                                                                                                                                        |
| :--------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **\* Q-Learning**                   | Learns a "treat-value table" for situations and actions, choosing actions with highest long-term treat value.                                                                                                                                |
| **\* Deep Learning (in Deep RL)** |  Gives RL algorithms "powerful eyes and brains" using neural networks to understand complex situations.                                                                                                                             |
| **\* Policy Gradient**             |  Directly adjusts the "paper airplane's folds" (policy) to fly better, based on flight distance (rewards).                                                                                                                               |
| **\* Actor-Critic**                 |  Student (Actor) learns from teacher (Critic) feedback on action quality, improving policies faster.                                                                                                                                   |
| **\* Advantage Actor-Critic (A2C)** |  Actor-Critic with "extra helpful feedback" - Critic tells "how much better/worse" action was than average.                                                                                                                      |
| **\* Soft Actor-Critic (SAC)**      |  Actor-Critic encouraged to be "curious" - entropy bonus rewards diverse actions, making policies robust in uncertainty.                                                                                                                  |
| **\* Proximal Policy Optimization (PPO)** | Actor-Critic learning in "small, careful steps" - prevents "wild leaps" in policy, ensuring stable, reliable progress.                                                                                                                    |
| **\* Deep Q-Learning (DQN)**         | Q-Learning with Deep Learning "brain" - uses neural networks to estimate treat-values in complex situations.                                                                                                                               |
| **\* Prioritized Experience Replay (PER)** | RL agent replays past memories, focusing on "most surprising/important" moments (high TD-error) for efficient learning.                                                                                                                  |
| **\* Dueling DQN**                 | DQN "brain" split - one part for "situation goodness" (Value), another for "action goodness within situation" (Advantage) - for efficient learning.                                                                                             |
| **\* Noisy Networks**              |  Agent's "brain" with "internal randomness" - noise in network encourages natural exploration, replacing epsilon-greedy.                                                                                                                    |
| **\* Noisy Dueling Double DQN**     |  "All-star DQN" - Combines Deep Learning, Dueling, Double DQN, PER, Noisy Nets for a powerful, improved DQN agent.                                                                                                                        |
| **\* Soft Q-Learning (SQL)**        | Q-Learning encouraging "flexible choices" - "soft" values reward actions probabilistically, promoting exploration.                                                                                                                             |
| **\* Distributional DQN (C51)**   |  DQN learning the *distribution* of "treat-values," not just the average - understanding the range of possible outcomes.                                                                                                                       |
| **\* Trust Region Policy Optimization (TRPO)** | Policy Gradient with "trust region" - limits policy change per step, ensuring reliable, monotonic improvement like a cautious climber.                                                                                                 |
| **\* Deep Deterministic Policy Gradient (DDPG)** |  "Deterministic guidance with Critic feedback" - Actor directly controlled by Critic's evaluation in continuous action spaces.                                                                                                         |
| **\* Twin Delayed Deep Deterministic Policy Gradient (TD3)** | "Skeptical Twin Critics, Smoothing, Delayed Guidance" - improved DDPG with twin critics, target smoothing, delayed updates for robustness.                                                                                             |
| **\* Hierarchical DQN (h-DQN)**   | DQN with a "boss and worker" - Meta-Controller sets high-level goals, Controller executes low-level actions to achieve them.                                                                                                                      |
| **\* N-step DQN**                  | DQN using "multi-step learning" - updates Q-values based on rewards over N steps, bridging 1-step TD and Monte Carlo methods.                                                                                                                    |
| **\* QR-DQN (Quantile Regression DQN)** | Distributional DQN with "quantile view" - represents value distribution using flexible quantiles, adapting to distribution shapes.                                                                                                              |
| **\* IQN (Implicit Quantile Networks)** | "Smarter, efficient quantile generator" - learns a function to generate quantiles "on-demand" for any quantile fraction.                                                                                                                     |
| **\* FQF (Fully Parameterized Quantile Function)** | "Ultimate Distributional RL" - learns to model the *entire CDF shape* directly, adaptively choosing key quantiles for data-efficient, powerful representation.                                                                                       |


# Environment

    Python3.12


# Dependencies

```shell
  sudo apt install libsdl2-dev swig python3.12-tk
  sudo apt install cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev wget unzip
```


# Prepare

```shell
  virtualenv -p python3.12 env && source env/bin/activate && pip install -r requirements.txt
```


# Additional Rocket Lander Gym extension

```shell
  git clone https://github.com/Jeetu95/Rocket_Lander_Gym.git

  change CONTINUOUS variable in Rocket_Lander_Gym/rocket_lander_gym/envs/rocket_lander.py to False

  cd Rocket_Lander_Gym && pip install .
```


# Google's Jax and Flax

    https://github.com/google/jax
    https://github.com/google/flax

  Variables can be vary, change those variables according to your machine specs


```shell
	PYTHON_VERSION=cp38  # alternatives: cp36, cp37, cp38
	CUDA_VERSION=cuda101  # alternatives: cuda100, cuda101, cuda102, cuda110
	PLATFORM=manylinux2010_x86_64  # alternatives: manylinux2010_x86_64
	BASE_URL='https://storage.googleapis.com/jax-releases'
	pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.51-$PYTHON_VERSION-none-$PLATFORM.whl
	pip install --upgrade jax  # install jax
	pip install --upgrade flax
```


# When on-deman GPU resource utilization needed

```shell
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
````

# References

    https://github.com/joaogui1/RL-JAX/tree/master/DQN
