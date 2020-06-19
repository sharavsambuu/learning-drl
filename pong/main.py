from time import sleep
import gym

env = gym.make('Pong-v0')
#env = gym.make('Freeway-v0')


episodes = 5

for i in range(episodes):
	state = env.reset()
	for _ in range(1000):
		action = env.action_space.sample()
		new_state, reward, done, _ = env.step(action)
		state = new_state
		env.render()
		if done:
			break

env.close()
