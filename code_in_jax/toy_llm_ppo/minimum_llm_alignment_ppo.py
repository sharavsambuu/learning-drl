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

debug_render              = False
num_episodes              = 50000
learning_rate             = 0.001
gamma                     = 0.99
gae_lambda                = 0.95
clip_ratio                = 0.2
policy_epochs             = 10
batch_size                = 32
mini_batch_size           = 16
sentence_length           = 20
hidden_size               = 128
epsilon_exploration       = 0.1
positive_example_episodes = 500


vocabulary_characters = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ' ', '.', ',', '!', '?', '<EOS>'
]
char_to_index       = {char: index for index, char in enumerate(vocabulary_characters)}
index_to_char       = {index: char for index, char in enumerate(vocabulary_characters)}
vocab_size          = len(vocabulary_characters)
eos_token_index     = char_to_index['<EOS>']
padding_token_index = eos_token_index


class ToyLLM(nn.Module):
    n_vocab    : int
    hidden_size: int
    @nn.compact
    def __call__(self, x, hidden_state):
        embed    = nn.Embed(num_embeddings=self.n_vocab, features=self.hidden_size, embedding_init=jax.nn.initializers.uniform())
        gru_cell = nn.GRUCell(features=self.hidden_size, kernel_init=jax.nn.initializers.uniform())
        dense    = nn.Dense(features=self.n_vocab, kernel_init=jax.nn.initializers.uniform())
        carry    = hidden_state
        logits_list = []
        for i in range(x.shape[1]):
            input_embed = embed(x[:, i])
            carry, _    = gru_cell(carry, input_embed)
            logits      = dense(carry)
            logits_list.append(logits)
        logits = jnp.stack(logits_list, axis=1)
        return logits, carry

class CriticNetwork(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x, hidden_state):
        embed    = nn.Embed(num_embeddings=vocab_size, features=self.hidden_size, embedding_init=jax.nn.initializers.uniform())
        gru_cell = nn.GRUCell(features=self.hidden_size, kernel_init=jax.nn.initializers.uniform())
        dense    = nn.Dense(features=1, kernel_init=jax.nn.initializers.uniform())

        carry      = hidden_state
        value_list = []
        for i in range(x.shape[1]):
            input_embed = embed(x[:, i])
            carry, _    = gru_cell(carry, input_embed)
            value       = dense(carry)
            value_list.append(value)
        value = jnp.stack(value_list, axis=1)
        return value, carry

rng = jax.random.PRNGKey(0)
toy_llm_module = ToyLLM(n_vocab=vocab_size, hidden_size=hidden_size)
critic_module  = CriticNetwork(hidden_size=hidden_size)

dummy_input = jnp.zeros((1, 1), dtype=jnp.int32)
initial_hidden_state_actor  = nn.GRUCell(features=hidden_size, kernel_init=jax.nn.initializers.uniform()).initialize_carry(rng, (1,))
initial_hidden_state_critic = nn.GRUCell(features=hidden_size, kernel_init=jax.nn.initializers.uniform()).initialize_carry(rng, (1,))

actor_params     = toy_llm_module.init(rng, dummy_input, initial_hidden_state_actor)['params']
critic_params    = critic_module.init(rng, dummy_input, initial_hidden_state_critic)['params']

actor_optimizer  = optax.adam(learning_rate)
critic_optimizer = optax.adam(learning_rate)

actor_opt_state  = actor_optimizer.init(actor_params)
critic_opt_state = critic_optimizer.init(critic_params)


@jax.jit
def actor_inference(actor_params, x, hidden_state):
    logits, next_hidden_state = toy_llm_module.apply({'params': actor_params}, x, hidden_state)
    action_probabilities = nn.softmax(logits[:, -1, :], axis=-1)
    return action_probabilities, next_hidden_state

@jax.jit
def critic_inference(critic_params, x, hidden_state):
    value, next_hidden_state = critic_module.apply({'params': critic_params}, x, hidden_state)
    return value[:, -1, :], next_hidden_state


def reward_model_word_count(sentence, positive_keywords):
    words = sentence.lower().split()
    positive_word_count = 0
    for word in words:
        if word in positive_keywords:
            positive_word_count += 1
    return positive_word_count

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
    positive_keywords    = ["good", "great", "wonderful", "amazing", "happy", "joyful", "positive"]
    edit_distance_reward = reward_model_edit_distance(sentence=sentence, positive_keywords=positive_keywords)
    word_count_reward    = reward_model_word_count(sentence=sentence, positive_keywords=positive_keywords)
    combined_reward      = edit_distance_reward + 0.3 * word_count_reward  
    return combined_reward


def gae_advantage(rewards, values, last_value, gamma, gae_lambda):
    values_np = np.array(values + [last_value])
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_np[t + 1] - values_np[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
    return advantages

def pad_sequences(sequences, max_length, padding_value):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_length:
            padded_seq = seq[:max_length]
        else:
            padded_seq = seq + [padding_value] * (max_length - len(seq))
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)

@jax.jit
def train_step(actor_params, actor_opt_state, critic_params, critic_opt_state, batch, initial_hidden_state_actor, initial_hidden_state_critic):
    states, actions, old_log_probs, advantages, returns, masks = batch

    def actor_loss_fn(actor_params):
        logits, _ = toy_llm_module.apply({'params': actor_params}, states, initial_hidden_state_actor)
        action_probabilities = nn.softmax(logits, axis=-1)
        log_probs = jnp.log(action_probabilities[jnp.arange(states.shape[0])[:, None], jnp.arange(states.shape[1]), actions]) * masks
        ratio = jnp.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        actor_loss = -jnp.mean(jnp.sum(jnp.minimum(surr1, surr2), axis=1))
        return actor_loss

    def critic_loss_fn(critic_params):
        values, _ = critic_module.apply({'params': critic_params}, states, initial_hidden_state_critic)
        critic_loss = jnp.mean(jnp.sum(((values - returns)**2) * masks, axis=1))
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
        episode_rewards   = 0
        episode_sentences = []
        episode_states_list, episode_actions_list, episode_rewards_list, episode_log_probs_list, episode_values_list = [], [], [], [], []

        sentence_tokens    = []
        current_state      = 0
        states_sequence    = [current_state]
        actions_sequence   = []
        log_probs_sequence = []
        values_sequence    = []
        rewards_sequence   = []

        initial_hidden_state_actor  = nn.GRUCell(features=hidden_size, kernel_init=jax.nn.initializers.uniform()).initialize_carry(rng, (1,))
        initial_hidden_state_critic = nn.GRUCell(features=hidden_size, kernel_init=jax.nn.initializers.uniform()).initialize_carry(rng, (1,))
        hidden_state_actor  = initial_hidden_state_actor
        hidden_state_critic = initial_hidden_state_critic

        positive_example_mode   = episode < positive_example_episodes
        example_sentence_tokens = []
        example_sentence_index  = 0

        if positive_example_mode:
            example_sentence_tokens = positive_example_token_sequences[episode % len(positive_example_token_sequences)]
            example_sentence_index  = 0

        for step in range(sentence_length):
            rng, action_key, value_key = jax.random.split(rng, 3)

            input_state = jnp.asarray([[current_state]])
            action_probabilities, hidden_state_actor = actor_inference(actor_params, input_state, hidden_state_actor)
            action_probabilities = np.array(action_probabilities[0])
            if np.isnan(action_probabilities).any():
                #print("Warning: NaN probabilities detected, using uniform distribution.")
                action_probabilities = np.ones(vocab_size) / vocab_size # Uniform distribution

            if positive_example_mode and example_sentence_index < len(example_sentence_tokens):
                action_index = example_sentence_tokens[example_sentence_index]
                example_sentence_index += 1
            else:
                if random.random() < epsilon_exploration:
                    action_index = random.choice(range(vocab_size))
                else:
                    action_index = np.random.choice(vocab_size, p=action_probabilities)

            predicted_value, hidden_state_critic = critic_inference(critic_params, input_state, hidden_state_critic)
            value = np.array(predicted_value[0]).item()
            log_prob = jnp.log(action_probabilities[action_index])

            action = int(action_index)

            states_sequence   .append(current_state)
            actions_sequence  .append(action)
            values_sequence   .append(value)
            log_probs_sequence.append(log_prob)
            rewards_sequence  .append(0)

            sentence_tokens.append(action)
            current_state = action

            if action_index == eos_token_index:
                break

        sentence_text        = decode_text(sentence_tokens)
        reward               = reward_model(sentence_text)
        episode_rewards      = reward
        rewards_sequence[-1] = reward # Assign reward to the last step

        advantages = gae_advantage(rewards_sequence, values_sequence, 0, gamma, gae_lambda)
        returns    = advantages + np.array(values_sequence)

        episode_states_list   .append(states_sequence[:-1]) # Remove last state because action is taken from previous state
        episode_actions_list  .append(actions_sequence    )
        episode_log_probs_list.append(log_probs_sequence  )
        episode_rewards_list  .append(rewards_sequence    )
        episode_values_list   .append(values_sequence     )

        padded_states     = pad_sequences(episode_states_list   , sentence_length, padding_token_index)
        padded_actions    = pad_sequences(episode_actions_list  , sentence_length, padding_token_index)
        padded_log_probs  = pad_sequences(episode_log_probs_list, sentence_length, 0.0                )
        padded_advantages = pad_sequences([advantages.tolist()] , sentence_length, 0.0                ) # Pad advantages
        padded_returns    = pad_sequences([returns.tolist()   ] , sentence_length, 0.0                ) # Pad returns
        masks             = (padded_states != padding_token_index).astype(np.float32)

        batch_data = (
            padded_states    ,
            padded_actions   ,
            padded_log_probs ,
            padded_advantages,
            padded_returns   ,
            masks
        )

        for _ in range(policy_epochs):
            perm = np.random.permutation(1) # batch size is 1 effectively
            for start_idx in range(0, 1, mini_batch_size): # Iterate once since batch size is 1
                mini_batch_idx = perm[start_idx:start_idx + mini_batch_size]
                mini_batch = tuple(arr[mini_batch_idx] for arr in batch_data)

                actor_params, actor_opt_state, critic_params, critic_opt_state, combined_loss, actor_loss, critic_loss = train_step(
                    actor_params, actor_opt_state, critic_params, critic_opt_state, mini_batch, initial_hidden_state_actor, initial_hidden_state_critic
                )

        print(f"Episode {episode}, Reward: {episode_rewards:.2f}, Sentence: '{sentence_text}', Combined Loss: {combined_loss:.4f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Exploration: {epsilon_exploration if episode >= positive_example_episodes else 'Off (Positive Examples)'}")

finally:
    pass