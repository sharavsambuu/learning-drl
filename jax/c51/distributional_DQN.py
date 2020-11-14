import os
import random
import math
import gym
from collection import deque
import flax
import jax
from jax import numpy as jnp
import numpy as np

num_episodes  = 500
batch_size    = 64
learning_rate = 0.01
syn_steps     = 100
memory_length = 5000

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99

