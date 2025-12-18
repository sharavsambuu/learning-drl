#
# CRITIC-LESS GRPO (REINFORCE STYLE)
#
# Зорилго
# Энэхүү код нь CartPole тоглоомыг Critic сүлжээ ашиглахгүйгээр зөвхөн Actor сүлжээ 
# болон GRPO аргачлалаар сургах туршилт юм
#
# Архитектур
# Actor Network нь энгийн MLP буюу Multi-Layer Perceptron бүтэцтэй
# Value Function байхгүй тул Advantage ийг туршилтын үр дүнгээс шууд тооцно
#
# Аргачлал
# GRPO буюу Group Relative Policy Optimization
# Нэг ижил seed бүхий орчинд олон удаагийн туршилт буюу Rollout хийж
# тэдгээрийн үр дүнг хооронд нь харьцуулах замаар Advantage тооцоолно
# Critic модель сургах шаардлагагүй тул санах ой болон тооцоололд хэмнэлттэй
#
# Outcome-Based Advantage
# Түүхийн алхам бүрт Reward өгөхгүйгээр зөвхөн тоглоом дууссан үр дүнгээр
# тухайн оролдлогыг бүхэлд нь үнэлэх аргачлал
#
#

import os
import random
import math
import jax
import optax
import numpy      as np
import gymnasium  as gym
import flax.linen as nn
from   jax        import numpy as jnp


# CONFIGURATION

debug_render          = True
debug                 = True
play_frequency        = 30              # Хэдэн episode тутамд render хийж шалгах
num_episodes          = 10000
learning_rate         = 0.001
gamma                 = 0.99
env_name              = "CartPole-v1"
group_size            = 6               # Нэг update хийхэд цуглуулах туршилтын тоо
max_steps             = 500             # CartPole v1 ийн дээд хязгаар

entropy_coefficient   = 0.01            # Моделийн сониуч байдлыг дэмжих


# MODEL ARCHITECTURE

class ActorNetwork(nn.Module):
    n_actions: int
    
    @nn.compact
    def __call__(self, x):
        # State input ийг боловсруулах давхаргууд
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        
        # Action Probability гаргах хэсэг
        x = nn.Dense(features=self.n_actions)(x)
        return nn.softmax(x)


# INITIALIZATION

# GRPO бүлгийн гишүүн бүрт зориулсан тусдаа орчин үүсгэх
env_array   = [gym.make(env_name, render_mode=None) for _ in range(group_size)]
# Render хийх буюу явцыг нүдээр харах орчин
env         = gym.make(env_name, render_mode='human')

state, info = env.reset()
state       = np.array(state, dtype=np.float32)
state_shape = state.shape
n_actions   = env.action_space.n

# Модель болон Optimizer тохиргоо
actor_module           = ActorNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape)
actor_params           = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model_params     = actor_params

actor_optimizer_def    = optax.adam(learning_rate)
actor_optimizer_state  = actor_optimizer_def.init(actor_model_params)


# JAX COMPILATION & UPDATE LOGIC

@jax.jit
def actor_inference(params, x):
    # Action сонгоход ашиглах магадлалыг тооцоолох
    return actor_module.apply({'params': params}, x)

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, props):
    # props[0] -> States     (Batch, StateDim)
    # props[1] -> Actions    (Batch,)
    # props[2] -> Advantages (Batch,)
    
    def loss_fn(params):
        action_probas = actor_module.apply({'params': params}, props[0])
        
        # Сонгосон action ийн магадлалыг олж авах
        probs         = jnp.take_along_axis(action_probas, props[1][:, None], axis=1).squeeze(1)
        probs         = jnp.clip(probs, 1e-8, 1.0)
        logp          = jnp.log(probs)

        # Critic-less GRPO тул Advantage ийг гаднаас шууд оруулж ирнэ
        # Градиент урсгалыг таслах буюу stop_gradient ашиглах
        advantages_sg = jax.lax.stop_gradient(props[2])

        # Entropy Bonus буюу хэт нэг хэвийн шийдвэр гаргахаас сэргийлэх
        entropies     = -jnp.sum(action_probas * jnp.log(action_probas + 1e-8), axis=1)

        # Policy Gradient Loss буюу REINFORCE аргачлал
        # Advantage өндөр байх тусам тухайн action ийг хийх магадлалыг ихэсгэнэ
        pg_loss       = -jnp.mean(logp * advantages_sg)
        ent_bonus     =  jnp.mean(entropies)

        # Нийт Loss функц
        loss          = pg_loss - entropy_coefficient * ent_bonus
        return loss, (pg_loss, ent_bonus)

    # Градиент тооцоолж моделийг шинэчлэх
    (loss, (pg_loss, ent_bonus)), gradients = jax.value_and_grad(loss_fn, has_aux=True)(actor_model_params)
    updates, new_optimizer_state = actor_optimizer_def.update(gradients, optimizer_state, actor_model_params)
    new_params = optax.apply_updates(actor_model_params, updates)
    
    return new_optimizer_state, new_params, loss, pg_loss, ent_bonus


# ROLLOUT & GRPO LOGIC

def rollout_trajectory(group_member_id, actor_model_params, seed):
    """
    Нэг Agent ийн бүтэн тоглолтыг гүйцэтгэх функц
    """
    env_item       = env_array[group_member_id]
    
    # GRPO ийн гол санаа нь бүх Agent нэг ижил нөхцөлөөс эхлэх ёстой
    # Тиймээс бүгд ижил seed ашиглан reset хийнэ
    state, info    = env_item.reset(seed=int(seed))
    state          = np.array(state, dtype=np.float32)

    states         = np.zeros(shape=(max_steps,) + state_shape, dtype=np.float32)
    actions        = np.zeros(shape=(max_steps,), dtype=np.int32)
    rewards        = np.zeros(shape=(max_steps,), dtype=np.float32)

    step           = 0

    for _ in range(max_steps):
        # Action сонгох
        action_probabilities = actor_inference(actor_model_params, jnp.asarray([state]))
        action_probabilities = np.array(action_probabilities[0])
        action               = np.random.choice(n_actions, p=action_probabilities)

        next_state, reward, terminated, truncated, info = env_item.step(int(action))
        done_boundary        = bool(terminated or truncated)

        # Түүхийг хадгалах
        states [step, :]     = state
        actions[step   ]     = int(action)
        rewards[step   ]     = float(reward)

        step      += 1
        state      = np.array(next_state, dtype=np.float32)

        if done_boundary: break

    trajectory_length = step
    
    # Outcome буюу нийт цуглуулсан оноог тооцох
    total_reward      = float(np.sum(rewards[:trajectory_length])) if trajectory_length > 0 else 0.0

    return group_member_id, total_reward, trajectory_length, states, actions


def rollout_group(actor_model_params, seed):
    """
    GRPO Group Rollout хийх буюу олон хувилбарыг зэрэг турших
    """
    group_total_rewards = np.zeros(shape=(group_size,), dtype=np.float32)
    group_lengths       = np.zeros(shape=(group_size,), dtype=np.int32  )

    group_states        = []
    group_actions       = []

    # Бүлгийн гишүүн бүр ижил seed ашиглан тоглоно
    for group_member_id in range(group_size):
        member_id, total_reward, trajectory_length, states, actions = rollout_trajectory(
            group_member_id     = group_member_id,
            actor_model_params  = actor_model_params,
            seed                = seed
        )

        group_total_rewards[member_id] = float(total_reward)
        group_lengths      [member_id] = int  (trajectory_length)

        group_states .append(states [:trajectory_length])
        group_actions.append(actions[:trajectory_length])

    # Group Baseline буюу бүлгийн дундаж оноог олох
    group_mean_reward = float(np.mean(group_total_rewards))

    # Advantage тооцоолол буюу тухайн туршилтын үр дүнг бүлгийн дундажтай харьцуулах
    # DeepSeek style буюу standard deviation д хуваахгүйгээр шууд зөрүүг авна
    group_adv_scalar  = (group_total_rewards - group_mean_reward)

    # Outcome-based Advantage ийг алхам бүрт хуулж өгөх
    # Учир нь Critic байхгүй тул алхам бүрийн чанарыг тус тусад нь үнэлэх боломжгүй
    # Тиймээс эцсийн үр дүнг тэр чигт нь ашиглана
    group_advantages  = []
    for member_id in range(group_size):
        length  = int(group_lengths[member_id])
        adv_trj = np.full(shape=(length,), fill_value=float(group_adv_scalar[member_id]), dtype=np.float32)
        group_advantages.append(adv_trj)

    return group_mean_reward, group_lengths, group_states, group_actions, group_advantages


# TRAINING LOOP

try:
    global_step = 0

    print("\n" + "="*50)
    print(f"  STARTING TRAINING: {env_name}")
    print(f"  Group Size: {group_size} | Max Steps: {max_steps}")
    print("="*50 + "\n")

    for episode in range(num_episodes):

        # Өгөгдөл цуглуулах
        group_mean_reward, group_lengths, group_states, group_actions, group_advantages = rollout_group(
            actor_model_params  = actor_model_params,
            seed                = episode
        )

        # Batch үүсгэх буюу өгөгдлийг нэгтгэх
        flat_states     = np.concatenate(group_states    , axis=0) if len(group_states    ) > 0 else np.zeros((0,) + state_shape, dtype=np.float32)
        flat_actions    = np.concatenate(group_actions   , axis=0) if len(group_actions   ) > 0 else np.zeros((0,), dtype=np.int32)
        flat_advantages = np.concatenate(group_advantages, axis=0) if len(group_advantages) > 0 else np.zeros((0,), dtype=np.float32)

        actor_loss = 0.0
        pg_loss    = 0.0
        ent_bonus  = 0.0

        # Сургалт хийх буюу Update алхам
        if flat_states.shape[0] > 0:
            global_step += int(flat_states.shape[0])

            actor_optimizer_state, actor_model_params, actor_loss, pg_loss, ent_bonus = backpropagate_actor(
                actor_optimizer_state,
                actor_model_params,
                (
                    jnp.asarray(flat_states    , dtype=jnp.float32),
                    jnp.asarray(flat_actions   , dtype=jnp.int32  ),
                    jnp.asarray(flat_advantages, dtype=jnp.float32),
                )
            )

        # Лог хэвлэх
        if debug and (episode % 10 == 0):
            print(f"Ep {episode:6d} | Steps {global_step:9d} | GroupMeanR {group_mean_reward:9.3f} | ActLoss {float(actor_loss):9.4f} | PG {float(pg_loss):9.4f} | Ent {float(ent_bonus):9.4f}")

        # Visual Validation буюу сургалтын үр дүнг нүдээр харах
        if episode % play_frequency == 0 and debug_render == True:
            state, info = env.reset(seed=int(episode))
            state       = np.array(state, dtype=np.float32)
            rewards     = []

            while True:
                action_probabilities  = actor_inference(actor_model_params, jnp.asarray([state]))
                action_probabilities  = np.array(action_probabilities[0])
                action                = np.random.choice(n_actions, p=action_probabilities)

                next_state, reward, terminated, truncated, info = env.step(int(action))
                done       = terminated or truncated
                next_state = np.array(next_state, dtype=np.float32)

                rewards.append(float(reward))
                state = next_state

                env.render()

                if done:
                    print(f"   >> Test Play Episode {episode}: Reward {sum(rewards)}")
                    break

finally:
    # Програмыг хаах үед орчныг цэвэрлэх
    env.close()
    for env_item in env_array:
        env_item.close()