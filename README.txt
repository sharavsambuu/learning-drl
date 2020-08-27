# About
  This repo is just my learning journey and may contain a buggy naive implementations.

# Tasks

  On Going - Implement continuous Soft Actor Critics with Jax
  On Going - Implement Categorical DQN with Jax
  - Implement discrete Soft Actor Critics with Jax
  - Implement N-step DQN with Jax
  - Implement Hierarchical DQN
  - Implement DDPG aka Deep Deterministic Policy Gradient with Jax
  - Implement TD3 aka Twin Delayed Deep Deterministic Policy Gradient with Jax
  - Implement PPO aka Proximal Policy Optimization
  - Implement TRPO aka Trust Region Policy Optimization
  - Implement SimCLRv2 with Jax
  - Implement CURL with and compare results
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


# Dependencies
  sudo apt install libsdl2-dev swig python3-tk
  sudo apt install python-numpy cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev wget unzip

# Prepare
  virtualenv -p python3 env && source env/bin/activate && pip install -r requirements.txt

# Additional Rocket Lander Gym extension
  git clone https://github.com/Jeetu95/Rocket_Lander_Gym.git

  change CONTINUOUS variable in Rocket_Lander_Gym/rocket_lander_gym/envs/rocket_lander.py to False

  cd Rocket_Lander_Gym && pip install .

# Google's Jax and Flax
  https://github.com/google/jax
  https://github.com/google/flax

  Variables can be vary, change those variables according to your machine specs

`
	PYTHON_VERSION=cp38  # alternatives: cp36, cp37, cp38
	CUDA_VERSION=cuda101  # alternatives: cuda100, cuda101, cuda102, cuda110
	PLATFORM=manylinux2010_x86_64  # alternatives: manylinux2010_x86_64
	BASE_URL='https://storage.googleapis.com/jax-releases'
	pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.51-$PYTHON_VERSION-none-$PLATFORM.whl
	pip install --upgrade jax  # install jax
	pip install --upgrade flax
`

# When on-deman GPU resource utilization needed
`
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
`

# References
  https://github.com/joaogui1/RL-JAX/tree/master/DQN
