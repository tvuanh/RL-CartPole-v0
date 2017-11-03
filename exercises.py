#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import gym
from gym import wrappers

import numpy as np


def qlearn_cartpole():
    Q = train()
    env = gym.make("CartPole-v0")
    for _ in range(100):
        observation = env.reset()
        for t in range(1000):
            env.render()


def train(Nepisodes=20, gamma=0.8):
    Q = np.zeros((10, 2))
    env = gym.make("CartPole-v0")
    for _ in range(Nepisodes):
        observation = env.reset()
        current_state = observation_to_state(observation)
        for t in range(100):
            env.render()
            action = np.random.choice([0, 1])
            observation, reward, done, info = env.step(action)
            next_state = observation_to_state(observation)
            maxQ = np.max(Q[next_state, :])
            Q[current_state, action] = reward + gamma * maxQ
    return Q


def observation_to_state(observation):
    _, _, angle, _ = observation
    low, hi = -0.15, 0.15
    step = (hi - low) / 10
    return int((angle - low) / step)


def cartpole():
    env = gym.make("CartPole-v0")
    env = wrappers.Monitor(env, "/tmp/cartpole-experiment-1", force=True)

    total_reward = 0.

    for _ in range(100):
        observation = env.reset() # [cart_position, cart_velocity, pole_angle, pole_rotation_rate]
        for t in range(1000):
            env.render()
            # if t % 50 == 0:
            #     print(observation)

            print("t = {}".format(t))
            print("    Initial observation {}".format(observation))
            # action = cart_action_from_base_model(observation)
            action = cart_action(observation)
            observation, reward, done, info = env.step(action)
            print("    Chosen action {}, subsequent observation {}".format(action, observation))
            total_reward += reward

            if done:
                print(observation)
                print("Episode finishes after {} steps.".format(t+1))
                break

    print("average reward after 100 episodes {}".format(total_reward / 100.))


def cart_action(observation):
    cart_position, cart_speed, pole_angle, pole_rotation = observation
    action = None
    if pole_angle + pole_rotation > 0.15:
        action = 1
    elif pole_angle + pole_rotation < -0.15:
        action = 0
    else:
        if np.abs(cart_position + cart_speed) >= 2.4:
            if cart_position > 0:
                action = 0
            else:
                action = 1
        else:
            if pole_rotation > 0:
                action = 1
            else:
                action = 0
    
    return action


def cart_action_from_base_model(observation):
    action = None
    if observation[3] > 0:
        action = 1
    elif observation[3] < 0:
        action = 0
    if abs(observation[3]) < 0.15 and abs(observation[1]) > 0.5:
        action = 0 if (action == 1) else 1
    return action


if __name__ == "__main__":
    cartpole()
