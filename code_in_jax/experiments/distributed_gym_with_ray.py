# Spawn multiple environment using ray
# References:
#  - https://bair.berkeley.edu/blog/2018/01/09/ray/
import gym
import ray

ray.init()

@ray.remote
class Simulator(object):
    def __init__(self,):
        self.env = gym.make('CartPole-v1')
        self.env.reset()
    def step(self, action):
        return self.env.step(action)
    def random_step(self,):
        action = self.env.action_space.sample()
        return self.env.step(action)

num_envs   = 4
simulators = [Simulator.remote() for i in range(num_envs)]

num_steps    = 10
observations = []
for _ in range(num_steps):
    for simulator in simulators:
        state = ray.get(simulator.random_step.remote())
        observations.append(state)

print(observations)

