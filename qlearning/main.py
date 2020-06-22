from os   import system
from time import sleep
import gym
import numpy as np

env = gym.make('FrozenLake8x8-v0')

system("clear")
print("Нийт төлөв :", env.observation_space.n)
print("Үйлдлүүд   :", env.action_space.n     )
print("Удахгүй сургаж эхлэнэ түр хүлээнэ үү!")
sleep(3)
system("clear")

# Q хүснэгтийг бүгдийг тэгээр дүүргэн үүсгэх
Q  = np.zeros([env.observation_space.n, env.action_space.n])
# alpha буюу learning rate
learning_rate = 0.8
# gamma буюу discount factor
gamma         = 0.95
# episode-ийн тоо, хичнээн удаа environment-ийг дахин эхлүүлж явуулах вэ
episodes      = 4000
# episode бүрт цуглуулсан reward оноонуудыг хадгалах жагсаалт
reward_list   = []

for i in range(episodes):
	# episode эхлэж байгаа тул environment тэр чигт нь шинэчлэх
	state        = env.reset()
	total_reward = 0
	done         = False
	while True:
		# action буюу үйлдлийг сонгох, гэхдээ тодорхой хэмжээний noise-тойгоор
		action = np.argmax(Q[state, :]+np.random.randn(1, env.action_space.n)*(1.0/(i+1)))
		# үйлдэл хийж шинэ төлөв болон reward оноог авах
		new_state, new_reward, done, _ = env.step(action)
		# шинэ олж авсан мэдлэгээрээ Q хүснэгтийг шинэчлэх
		Q[state, action] = Q[state, action] + \
				learning_rate*(new_reward + gamma*np.max(Q[new_state, :]) - Q[state, action])
		total_reward     = total_reward + new_reward
		state            = new_state

		#system("clear")
		env.render()

		if done == True:
			break
	reward_list.append(total_reward)

print("Сургаж дууслаа.")

print("Нийт оноо :", str(sum(reward_list)/episodes))
print("Q хүснэгт :")
print(Q)
