import time, random
from collections import deque

import numpy as np

import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# result location
result_location = "/tmp/cartpole-keras-1"

# set random seed
np.random.seed(100)


class CartPoleController(object):

    def __init__(self, env, n_hidden=50, gamma=1.0, epsilon=1.0,
                 n_episodes=500, batch_size=100, benchmark=150):
        self.env = env
        self.n_input = env.observation_space.shape
        self.n_hidden = n_hidden
        self.n_output = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.benchmark = benchmark
        self.memory = deque(maxlen=10000)

        # action neural network
        self.action_model = Sequential()
        self.action_model.add(
            Dense(
                self.n_hidden, input_shape=self.n_input, activation="tanh",
                )
            )
        self.action_model.add(Dense(self.n_output, activation="linear"))
        self.action_model.compile(
            loss="mse", optimizer=Adam(lr=0.01, decay=0.01)
            )

    def preprocess_state(self, state):
        return np.reshape(state, [1, self.n_input[0]])

    def epsilon_greedy_action(self, state, epsilon):
        if np.random.rand(1) < epsilon:
            return self.env.action_space.sample()
        else:
            Q = self.action_model.predict(state)
            return np.argmax(Q)

    def replay(self):
        size = min(len(self.memory), self.batch_size)
        minibatch = random.sample(self.memory, k=size)
        x_batch, y_batch = list(), list()
        for state, action, reward, next_state, done in minibatch:
            y_target = self.action_model.predict([state])
            y_target[0, action] = reward + done * self.gamma * np.max(
                self.action_model.predict(next_state)
                )
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        self.action_model.fit(
            np.array(x_batch), np.array(y_batch), batch_size=len(x_batch),
            verbose=False)

    def run(self):
        target = 200.
        scores = deque(maxlen=100)
        for episode in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            intra_episode_total_reward = 0
            while not done:
                action = self.epsilon_greedy_action(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                intra_episode_total_reward += reward
            scores.append(intra_episode_total_reward)
            if self.epsilon > 0.01 and intra_episode_total_reward > np.mean(scores):
                self.epsilon *= 1. - intra_episode_total_reward / target
            print("episode {} total score {}".format(episode, intra_episode_total_reward))
            if np.mean(scores) >= self.benchmark and len(scores) > 50:
                print("Learned how to play after {} episodes".format(episode))
            self.replay()


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
#     env = gym.wrappers.Monitor(env, result_location, force=True)
    cart_pole_controller = CartPoleController(env=env, n_episodes=2000)
    cart_pole_controller.run()
