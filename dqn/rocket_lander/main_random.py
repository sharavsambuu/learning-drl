from time import sleep
import gym
import rocket_lander_gym

env = gym.make('RocketLander-v0')

episodes = 5

for i in range(episodes):
	state = env.reset()
	for _ in range(1000):
		action = env.action_space.sample()
		new_state, reward, done, _ = env.step(action)
		print("Action Taken  ", action)
		print("Observation   ", state)
		print("Reward Gained ", reward)
		state = new_state
		env.render()
		if done:
			break

env.close()
