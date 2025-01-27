# About
  This repo is just my learning journey and may contain a buggy naive implementations.

# Tasks

  DONE - Implement C51 aka Categorical DQN with Jax
  - Implement QR-DQN which is improvements over C51
  - Implement IQN which is improvements over previous C51 and QR-DQN
  - Implement FQF which is improvements overs C51, QR-DQN and IQN
  - Implement N-step DQN with Jax
  - Implement Rainbow
  ON GOING - Implement continuous Soft Actor Critics with Jax
  ON GOING - Implement discrete Soft Actor Critics with Jax
  - Implement Hierarchical DQN
  - Implement DDPG aka Deep Deterministic Policy Gradient with Jax
  - Implement TD3 aka Twin Delayed Deep Deterministic Policy Gradient with Jax
  - Implement PPO aka Proximal Policy Optimization
  - Implement TRPO aka Trust Region Policy Optimization
  - Implement SimCLRv2 with Jax
  - Implement CURL with and compare results
  DONE - Implement A2C plus entropy bonus with Jax
  DONE - Implement SQL with Jax, aka Soft Q-Learning
  DONE - Implement A3C with Multiprocessing and Jax
  DONE - Implement A3C with Jax
  DONE - Implement online Advantage Actor Critics A2C with Jax
  DONE - Implement episodic Advantage Actor Critics A2C with Jax
  DONE - Implement Policy Gradient with Jax
  DONE - Implement vanilla DQN with Jax
  DONE - Implement vanilla DQN with Jax + PER
  DONE - Implement Double DQN with Jax
  DONE - Implement Double DQN with Jax + PER
  DONE - Implement Dueling DQN with Jax
  DONE - Implement Dueling DQN with Jax + PER
  DONE - Implement Dueling Double DQN with PER in jax
  DONE - Implement eGreedy Noisy Dueling Double DQN + PER


# Other things I will learn in the near future

  - What is SimCLRv2?
  - What is CURL?
  - What is MARL aka Multi-Agent RL?
  - What is Concurrent Experience Replay Trajectories?
  - What is Dec-HDRQN, Decentralized Hysteretic DQN?
  - What is PPO-RNN?
  - What is DQN-RNN?
  - What is General Advantage Estimation (GAE) Buffer?


#


| Algorithm Name                     | Intuitive Summary                                                                                                                                                                                                                                                        |
| :--------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Q-Learning**                   | Learns a "treat-value table" for situations and actions, choosing actions with highest long-term treat value.                                                                                                                                |
| **2. Deep Learning (in Deep RL)** |  Gives RL algorithms "powerful eyes and brains" using neural networks to understand complex situations.                                                                                                                             |
| **3. Policy Gradient**             |  Directly adjusts the "paper airplane's folds" (policy) to fly better, based on flight distance (rewards).                                                                                                                               |
| **4. Actor-Critic**                 |  Student (Actor) learns from teacher (Critic) feedback on action quality, improving policies faster.                                                                                                                                   |
| **5. Advantage Actor-Critic (A2C)** |  Actor-Critic with "extra helpful feedback" - Critic tells "how much better/worse" action was than average.                                                                                                                      |
| **6. Soft Actor-Critic (SAC)**      |  Actor-Critic encouraged to be "curious" - entropy bonus rewards diverse actions, making policies robust in uncertainty.                                                                                                                  |
| **7. Proximal Policy Optimization (PPO)** | Actor-Critic learning in "small, careful steps" - prevents "wild leaps" in policy, ensuring stable, reliable progress.                                                                                                                    |
| **8. Deep Q-Learning (DQN)**         | Q-Learning with Deep Learning "brain" - uses neural networks to estimate treat-values in complex situations.                                                                                                                               |
| **9. Prioritized Experience Replay (PER)** | RL agent replays past memories, focusing on "most surprising/important" moments (high TD-error) for efficient learning.                                                                                                                  |
| **10. Dueling DQN**                 | DQN "brain" split - one part for "situation goodness" (Value), another for "action goodness within situation" (Advantage) - for efficient learning.                                                                                             |
| **11. Noisy Networks**              |  Agent's "brain" with "internal randomness" - noise in network encourages natural exploration, replacing epsilon-greedy.                                                                                                                    |
| **12. Noisy Dueling Double DQN**     |  "All-star DQN" - Combines Deep Learning, Dueling, Double DQN, PER, Noisy Nets for a powerful, improved DQN agent.                                                                                                                        |
| **13. Soft Q-Learning (SQL)**        | Q-Learning encouraging "flexible choices" - "soft" values reward actions probabilistically, promoting exploration.                                                                                                                             |
| **14. Distributional DQN (C51)**   |  DQN learning the *distribution* of "treat-values," not just the average - understanding the range of possible outcomes.                                                                                                                       |
| **15. Trust Region Policy Optimization (TRPO)** | Policy Gradient with "trust region" - limits policy change per step, ensuring reliable, monotonic improvement like a cautious climber.                                                                                                 |
| **16. Deep Deterministic Policy Gradient (DDPG)** |  "Deterministic guidance with Critic feedback" - Actor directly controlled by Critic's evaluation in continuous action spaces.                                                                                                         |
| **17. Twin Delayed Deep Deterministic Policy Gradient (TD3)** | "Skeptical Twin Critics, Smoothing, Delayed Guidance" - improved DDPG with twin critics, target smoothing, delayed updates for robustness.                                                                                             |
| **18. Hierarchical DQN (h-DQN)**   | DQN with a "boss and worker" - Meta-Controller sets high-level goals, Controller executes low-level actions to achieve them.                                                                                                                      |
| **19. N-step DQN**                  | DQN using "multi-step learning" - updates Q-values based on rewards over N steps, bridging 1-step TD and Monte Carlo methods.                                                                                                                    |
| **20. QR-DQN (Quantile Regression DQN)** | Distributional DQN with "quantile view" - represents value distribution using flexible quantiles, adapting to distribution shapes.                                                                                                              |
| **21. IQN (Implicit Quantile Networks)** | "Smarter, efficient quantile generator" - learns a function to generate quantiles "on-demand" for any quantile fraction.                                                                                                                     |
| **22. FQF (Fully Parameterized Quantile Function)** | "Ultimate Distributional RL" - learns to model the *entire CDF shape* directly, adaptively choosing key quantiles for data-efficient, powerful representation.                                                                                       |


# Dependencies

```shell
  sudo apt install libsdl2-dev swig python3-tk
  sudo apt install python-numpy cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev wget unzip
```


# Prepare

```shell
  virtualenv -p python3 env && source env/bin/activate && pip install -r requirements.txt
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
