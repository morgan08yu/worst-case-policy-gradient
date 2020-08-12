import jax.numpy as np
from jax.config import config
from jax import jit, grad, random, lax, nn, ops
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Softmax
from jax.scipy.stats import norm

from typing import Any, Tuple
import gym
from gym import spaces
import random as rand
from collections import deque
import itertools
from tqdm import tqdm
from itertools import product


rng = random.PRNGKey(0)



class FastSlowLaneSelection(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FastSlowLaneSelection, self).__init__()
        self.states = np.eye(7)
        self.state_index = 0

        success_probability = 0.8

        dynamics = []
        for state_index in range(5):
            dynamics.append([])
            for action_index in range(2):
                dynamics[state_index].append([0. for i in range(7)])

                left_prob = success_probability if action_index == 0 else (1 - success_probability)
                right_prob = (1 - left_prob)

                dynamics[state_index][action_index][state_index + 1 + (state_index % 2)] = left_prob
                dynamics[state_index][action_index][state_index + 2 + (state_index % 2)] = right_prob

        self.dynamics = np.array(dynamics)

    def step(self, action, rng):
        action_index = int(np.argmax(action))

        dynamics = self.dynamics[self.state_index][action_index]

        temp, rng = random.split(rng)
        next_state_index = int(random.choice(temp, 7, p=dynamics))
        self.state_index = next_state_index
        next_state = self.states[next_state_index]

        mean = 1. if next_state_index == 0 else 2.
        std = 1. if next_state_index == 0 else 2.

        temp, rng = random.split(rng)
        reward = std * random.normal(temp) + mean

        done = (next_state_index >= 5)

        return next_state, reward, done, {}

    def reset(self):
        self.state_index = 0


class Buffer():
    def __init__(self, maxlen=100):
        self.history = deque(maxlen=maxlen)

    def log(self, observation):
        self.history.append(observation)

    def sample(self, batch_size=1):
        observations = rand.sample(list(self.history), min(batch_size, len(self.history)))
        states, actions, next_states, rewards, alpha = zip(*observations)
        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(alpha)


num_states = 7
num_actions = 2

# ---------------------------------------------------- CRITIC ---------------------------------------------------------

critic_init, critic_forward_block = stax.serial(
    Dense(32), Relu,
    Dense(64), Relu,
    Dense(2)
)

def critic_forward(critic_params, inputs):
    outputs = critic_forward_block(critic_params, inputs).reshape(-1, 2)
    means, variances = np.split(outputs, 2, axis=1)
    variances = nn.softplus(variances)
    return means, variances

critic_lr = 1e-3
_, critic_params = critic_init(rng, (num_states + num_actions + 1,))
critic_opt_init, critic_opt_update, get_critic_params = optimizers.adam(critic_lr)
critic_state = critic_opt_init(critic_params)
fixed_critic_params = critic_params

# --------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------- ACTOR ---------------------------------------------------------

actor_init, actor_forward = stax.serial(
    Dense(32), Relu,
    Dense(64), Relu,
    Dense(num_actions), Softmax
)

actor_lr = 1e-3
_, actor_params = actor_init(rng, (num_states + 1,))
actor_opt_init, actor_opt_update, get_actor_params = optimizers.adam(actor_lr)
actor_state = actor_opt_init(actor_params)
fixed_actor_params = actor_params

# --------------------------------------------------------------------------------------------------------------------

env = FastSlowLaneSelection()
buffer = Buffer(100)

num_episodes = 500
epsilon = 0.15
gamma = 0.9
minibatch_size = 64

actor_count = itertools.count()
critic_count = itertools.count()

states = np.eye(num_states)
actions = np.eye(num_actions)

rampup = 10
update_frequency = 1
swap_training = 10
num_updates = 10
update_critic = True


def is_terminal(next_state):
    return np.argmax(next_state, 1) >= 5


def critic_loss(critic_params, fixed_critic_params, fixed_actor_params, env_dynamics, batch):
    state, action, _, reward, alpha = batch

    inputs = np.concatenate((state, action, alpha), 1)
    q_s, upsilon_s = critic_forward(fixed_critic_params, inputs)

    next_state = np.tile(states, (alpha.shape[0], 1))
    next_alpha = np.repeat(alpha, num_states, axis=0)

    inputs = np.concatenate((next_state, next_alpha), 1)
    next_action = actions[np.argmax(actor_forward(fixed_actor_params, inputs), 1)]

    inputs = np.concatenate((next_state, next_action, next_alpha), 1)
    q_sp, upsilon_sp = critic_forward(fixed_critic_params, inputs)

    state_index = np.argmax(np.repeat(state, num_states, axis=0), 1)
    action_index = np.argmax(np.repeat(action, num_states, axis=0), 1)
    next_state_index = np.argmax(next_state, 1)

    wq_sp = np.expand_dims(env_dynamics[state_index, action_index, next_state_index], 1) * q_sp
    wq_sp = np.expand_dims(np.sum(wq_sp.reshape(num_states, -1), 0), 1)

    wqs_sp = np.expand_dims(env_dynamics[state_index, action_index, next_state_index], 1) * np.square(q_sp)
    wqs_sp = np.expand_dims(np.sum(wqs_sp.reshape(num_states, -1), 0), 1)

    wu_sp = np.expand_dims(env_dynamics[state_index, action_index, next_state_index], 1) * upsilon_sp
    wu_sp = np.expand_dims(np.sum(wu_sp.reshape(num_states, -1), 0), 1)

    mu_1 = q_s
    C_1 = upsilon_s

    mu_2 = lax.stop_gradient(reward + gamma * wq_sp)
    C_2 = lax.stop_gradient(reward ** 2. + 2 * gamma * reward * wq_sp + gamma ** 2. * wu_sp + gamma ** 2. * wqs_sp - q_s ** 2.)

    W_loss = np.square(mu_1 - mu_2) + (C_1 + C_2 - 2 * np.sqrt(C_1 * C_2))
    return W_loss.mean()


@jit
def critic_step(i, critic_state, fixed_critic_params, fixed_actor_params, batch):
    critic_params = get_critic_params(critic_state)
    gradient = grad(critic_loss)(critic_params, fixed_critic_params, fixed_actor_params, env.dynamics, batch)
    return critic_opt_update(i, gradient, critic_state)


def actor_loss(actor_params, fixed_critic_params, env_dynamics, batch):
    state, _, _, _, alpha = batch

    inputs = np.concatenate((state, alpha), 1)
    action = actor_forward(actor_params, inputs)

    inputs = np.concatenate((state, action, alpha), 1)
    q_s, upsilon_s = critic_forward(fixed_critic_params, inputs)

    cvar = q_s - (norm.pdf(alpha) / norm.cdf(alpha)) * np.sqrt(upsilon_s)

    return -cvar.mean()


@jit
def actor_step(i, actor_state, fixed_critic_params, batch):
    actor_params = get_actor_params(actor_state)
    gradient = grad(actor_loss)(actor_params, fixed_critic_params, env.dynamics, batch)
    return actor_opt_update(i, gradient, actor_state)


for episodes in tqdm(range(num_episodes)):
    env.reset()
    state = states[0]

    alpha_rng, rng = random.split(rng)
    alpha = random.uniform(rng, (1,), minval=0.1, maxval=1.0)

    actor_params = get_actor_params(actor_state)

    done = False
    while not done:
        inputs = np.expand_dims(np.concatenate((state, alpha)), 0)
        action_vec = actor_forward(actor_params, inputs)
        action = actions[np.argmax(action_vec)]

        epsilon_rng, rng = random.split(rng)
        if random.uniform(epsilon_rng) < epsilon:
            action_rng, rng = random.split(rng)
            action = actions[random.randint(action_rng, (), minval=0, maxval=actions.shape[0])]

        temp, rng = random.split(rng)
        next_state, reward, done, _ = env.step(action, temp)
        buffer.log((state, action, next_state, reward, alpha))
        state = next_state

    if episodes > rampup and episodes % update_frequency == 0:
        batch = buffer.sample(minibatch_size)

        if update_critic:
            for updates in range(num_updates):
                critic_state = critic_step(next(critic_count), critic_state, fixed_critic_params, fixed_actor_params, batch)
            if episodes % swap_training == 0:
                fixed_critic_params = get_critic_params(critic_state)
                update_critic = not update_critic
        else:
            for updates in range(num_updates):
                actor_state = actor_step(next(actor_count), actor_state, fixed_critic_params, batch)
            if episodes % swap_training == 0:
                fixed_actor_params = get_actor_params(actor_state)
                update_critic = not update_critic


actor_params = get_actor_params(actor_state)
critic_params = get_critic_params(critic_state)


for alpha_value in [0.1, 0.5, 0.8, 1.0]:
    print('Alpha: {}'.format(alpha_value))

    alpha = np.full((states.shape[0], 1), alpha_value)
    inputs = np.concatenate((states, alpha), 1)
    print('Inputs:')
    print(inputs)
    print('Actor:')
    print(actor_forward(actor_params, inputs))


    inputs = np.array([np.concatenate((state, action)) for state, action in product(states, actions)])
    alpha = np.full((inputs.shape[0], 1), alpha_value)
    inputs = np.concatenate((inputs, alpha), 1)
    means, variances = critic_forward(critic_params, inputs)
    print('Inputs:')
    print(inputs)
    print('Means:')
    print(means)
    print('Variances:')
    print(variances)

    print('-' * 50)

