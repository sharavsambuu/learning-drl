#
#
# The goal is to teach a little LLM to write happy sentences.
#
#
# RL "State" => Text context: The "state" in RL terms represents the context of the sentence being generated. 
# It's how much of the sentence has been written so far.
#
# Critic's State => RNN's Hidden State: The Critic network uses the RNN's hidden state as 
# its "state" input to evaluate the sentence's progress.
#
# Actor's Action => Character Choice: The "action" the Toy LLM takes at each 
# step is choosing the next character to write from our vocabulary.
#
# PPO Teaches "Positive Writing": We use PPO to train the Toy LLM to generate 
# sentences that get high "positivity scores" from our simple Reward Model.
#
# Reward Model is like a "positivity meter" for sentences
# It doesn't compare our robot's sentence to some perfect "positive sentence." Instead, 
# it just reads the sentence and gives it a score based on how "positive" it sounds, 
# according to its own simple rules.
#
# Let's say our "positive keywords" are: "good", "happy".
#
#   Generated Sentence 1: "This is a good day."
#   Reward Model: Counts "good" (1 keyword).
#   Reward Score: 1
#
#   Generated Sentence 2: "Today is happy and sunny."
#   Reward Model: Counts "happy" (1 keyword).
#   Reward Score: 1
#
#   Generated Sentence 3: "This is a great and wonderful day!"
#   Reward Model: Counts "great", "wonderful" (2 keywords).
#   Reward Score: 2
#
#   Generated Sentence 4: "The weather is bad."
#   Reward Model: Counts no positive keywords.
#   Reward Score: 0
#
#
#
#   Those examples are count based, current change is edit distance
#
#


import os
import random
import math
import gym
from collections import deque

import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import time

debug_render = False
num_episodes = 50000
learning_rate = 0.001
gamma = 0.99
gae_lambda = 0.95
clip_ratio = 0.2
policy_epochs = 10
batch_size = 32
mini_batch_size = 16
sentence_length = 20
hidden_size = 128
epsilon_exploration = 0.1
positive_example_episodes = 5

vocabulary_characters = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ' ', '.', ',', '!', '?', '<EOS>'
]
char_to_index = {char: index for index, char in enumerate(vocabulary_characters)}
index_to_char = {index: char for index, char in enumerate(vocabulary_characters)}
vocab_size = len(vocabulary_characters)
eos_token_index = char_to_index['<EOS>']

class ToyLLM(nn.Module):
    n_vocab: int

    @nn.compact
    def __call__(self, x):
        x = nn.Embed(num_embeddings=self.n_vocab, features=hidden_size, embedding_init=jax.nn.initializers.uniform())(x)
        x = nn.Dense(features=hidden_size)(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.n_vocab)(x)
        return logits

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Embed(num_embeddings=vocab_size, features=hidden_size, embedding_init=jax.nn.initializers.uniform())(x)
        x = nn.Dense(features=hidden_size)(x)
        x = nn.relu(x)
        value = nn.Dense(features=1)(x)
        return value

rng = jax.random.PRNGKey(0)
toy_llm_module = ToyLLM(n_vocab=vocab_size)
critic_module = CriticNetwork()

dummy_input = jnp.zeros((1,), dtype=jnp.int32)
actor_params = toy_llm_module.init(rng, dummy_input)['params']
critic_params = critic_module.init(rng, dummy_input)['params']

actor_optimizer = optax.adam(learning_rate)
critic_optimizer = optax.adam(learning_rate)

actor_opt_state = actor_optimizer.init(actor_params)
critic_opt_state = critic_optimizer.init(critic_params)

@jax.jit
def actor_inference(actor_params, x):
    logits = toy_llm_module.apply({'params': actor_params}, x)
    action_probabilities = nn.softmax(logits)
    return action_probabilities

@jax.jit
def critic_inference(critic_params, x):
    value = critic_module.apply({'params': critic_params}, x)
    return value

def levenshtein_edit_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def reward_model_edit_distance(sentence, positive_keywords):
    words = sentence.lower().split()
    total_reward = 0
    for word in words:
        min_distance = float('inf')
        for keyword in positive_keywords:
            distance = levenshtein_edit_distance(word, keyword)
            min_distance = min(min_distance, distance)
        if min_distance < float('inf'):
            total_reward += 1.0 / (1 + min_distance)
    return total_reward

def reward_model(sentence):
    positive_keywords = ["good", "great", "wonderful", "amazing", "happy", "joyful", "positive"]
    return reward_model_edit_distance(sentence=sentence, positive_keywords=positive_keywords)

def gae_advantage(rewards, values, last_value, gamma, gae_lambda):
    values_np = np.array(values + [last_value])
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_np[t + 1] - values_np[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
    return advantages

@jax.jit
def train_step(actor_params, actor_opt_state, critic_params, critic_opt_state, batch):
    states, actions, old_log_probs, advantages, returns = batch

    def actor_loss_fn(actor_params):
        logits = toy_llm_module.apply({'params': actor_params}, states)
        action_probabilities = nn.softmax(logits)
        log_probs = jnp.log(action_probabilities[jnp.arange(len(actions)), actions])
        ratio = jnp.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        return actor_loss

    def critic_loss_fn(critic_params):
        values = critic_module.apply({'params': critic_params}, states).reshape(-1)
        critic_loss = jnp.mean((values - returns)**2)
        return critic_loss

    def combined_loss_fn(params):
        actor_loss_val = actor_loss_fn(params[0])
        critic_loss_val = critic_loss_fn(params[1])
        combined_loss = actor_loss_val + critic_loss_val
        return combined_loss, (actor_loss_val, critic_loss_val)

    (combined_loss, auxiliary_losses), grads = jax.value_and_grad(combined_loss_fn, has_aux=True)([actor_params, critic_params])
    actor_loss_val, critic_loss_val = auxiliary_losses
    actor_grads, critic_grads = grads

    actor_updates, actor_opt_state = actor_optimizer.update(actor_grads, actor_opt_state, actor_params)
    critic_updates, critic_opt_state = critic_optimizer.update(critic_grads, critic_opt_state, critic_params)

    actor_params = optax.apply_updates(actor_params, actor_updates)
    critic_params = optax.apply_updates(critic_params, critic_updates)

    return actor_params, actor_opt_state, critic_params, critic_opt_state, combined_loss, actor_loss_val, critic_loss_val

def decode_text(token_indices):
    decoded_text = ""
    for index in token_indices:
        if index == eos_token_index:
            break
        decoded_text += vocabulary_characters[index]
    return decoded_text

def encode_text(text):
    lower_text = text.lower()
    return [char_to_index[char] for char in lower_text] + [eos_token_index]

positive_example_sentences = [
    "this is a good day.",
    "today is happy and sunny!",
    "have a great and wonderful day!",
    "i am feeling joyful.",
    "life is positive and amazing.",
]
positive_example_token_sequences = [encode_text(sentence) for sentence in positive_example_sentences]

rng = jax.random.PRNGKey(0)
try:
    for episode in range(num_episodes):
        episode_rewards = 0
        episode_sentences = []
        episode_states, episode_actions, episode_rewards_list, episode_log_probs, episode_values = [], [], [], [], []

        sentence_tokens = []
        current_state = 0

        positive_example_mode = episode < positive_example_episodes
        example_sentence_tokens = []
        example_sentence_index = 0

        if positive_example_mode:
            example_sentence_tokens = positive_example_token_sequences[episode % len(positive_example_token_sequences)]
            example_sentence_index = 0

        for step in range(sentence_length):
            rng, action_key, value_key = jax.random.split(rng, 3)

            action_probabilities = actor_inference(actor_params, jnp.asarray([current_state]))
            action_probabilities = np.array(action_probabilities[0])

            if positive_example_mode and example_sentence_index < len(example_sentence_tokens):
                action_index = example_sentence_tokens[example_sentence_index]
                example_sentence_index += 1
            else:
                if random.random() < epsilon_exploration:
                    action_index = random.choice(range(vocab_size))
                else:
                    action_index = np.random.choice(vocab_size, p=action_probabilities)

            predicted_value = critic_inference(critic_params, jnp.asarray([current_state]))
            value = np.array(predicted_value[0]).item()
            log_prob = jnp.log(action_probabilities[action_index])

            action = int(action_index)

            episode_states.append(current_state)
            episode_actions.append(action)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            episode_rewards_list.append(0)

            sentence_tokens.append(action)
            current_state = action

            if action_index == eos_token_index:
                break

        sentence_text = decode_text(sentence_tokens)
        reward = reward_model(sentence_text)
        episode_rewards = reward

        advantages = gae_advantage(episode_rewards_list, episode_values, 0, gamma, gae_lambda)
        returns = advantages + np.array(episode_values)

        batch_data = (
            np.array(episode_states, dtype=np.int32),
            np.array(episode_actions, dtype=np.int32),
            np.array(episode_log_probs, dtype=np.float32),
            advantages.astype(np.float32),
            returns.astype(np.float32)
        )

        for _ in range(policy_epochs):
            perm = np.random.permutation(len(episode_states))
            for start_idx in range(0, len(episode_states), mini_batch_size):
                mini_batch_idx = perm[start_idx:start_idx + mini_batch_size]
                mini_batch = tuple(arr[mini_batch_idx] for arr in batch_data)
                actor_params, actor_opt_state, critic_params, critic_opt_state, combined_loss, actor_loss, critic_loss = train_step(
                    actor_params, actor_opt_state, critic_params, critic_opt_state, mini_batch
                )

        print(f"Episode {episode}, Reward: {episode_rewards:.2f}, Sentence: '{sentence_text}', Combined Loss: {combined_loss:.4f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Exploration: {epsilon_exploration if episode >= positive_example_episodes else 'Off (Positive Examples)'}")

finally:
    pass

