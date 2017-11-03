import random

import numpy as np

import gym
# import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)

#
# enter the game
#
env = gym.make("CartPole-v0")
# env = gym.wrappers.Monitor(env, "/tmp/cartpole-tensorflow-1", force=True)
gamma = 0.8
# epsilon = 0.2
epsilon = 1.0
num_episodes = 2000
# lists containing total rewards and total number of steps per episode
stepsList, rewardsList = list(), list()

with tf.Session() as sess:
    n_hidden_nodes = 50
    # establish the feed-forward part of the network used to choose actions
    input = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    W1 = tf.Variable(tf.random_normal([4, n_hidden_nodes], stddev=0.5))
    b1 = tf.Variable(tf.zeros([n_hidden_nodes]))
    hidden = tf.nn.sigmoid(tf.matmul(input, W1) + b1)
    W2 = tf.Variable(tf.random_normal([n_hidden_nodes, 2], stddev=0.5))
    b2 = tf.Variable(tf.zeros([2]))
    estimatedQ = tf.matmul(hidden, W2) + b2
    greedyAction = tf.argmax(estimatedQ, axis=1)

    # train the network using the quadratic loss function
    nextQ = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - estimatedQ))
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    target = 200.

    for episode in range(num_episodes):
        current_state = env.reset()
        rewardAll = 0.
        steps = 0
        done = False
        # the Q-network
        while steps < 400 and not done:
            steps += 1
            # greedy action and current Q vectors
            action, targetQ = sess.run(
                [greedyAction, estimatedQ], feed_dict={input: [current_state]}
                )
            # make greedy action epsilon-greedy
            if np.random.rand(1) < epsilon:
                action[0] = env.action_space.sample()
            # get the new state
            next_state, reward, done, _ = env.step(action[0])
            if not done:
                # obtain the next Q values by feeding the state through the network
                nextEstimatedQ = sess.run(estimatedQ, feed_dict={input: [next_state]})
                targetQ[0, action[0]] = reward + gamma * np.max(nextEstimatedQ)
            else:
#                 targetQ[0, action[0]] = reward
                targetQ[0, action[0]] = -1
            # train the networkd using the target and predicted Q values
            sess.run(
                optimiser, feed_dict={
                    input: [current_state],
                    nextQ: targetQ
                    }
                )
            rewardAll += reward
            current_state = next_state
        stepsList.append(steps)
        rewardsList.append(rewardAll)
        print("episode {} total score {}".format(episode, rewardAll))
        if epsilon > 0.01 and np.mean(rewardsList) > 10:
            epsilon *= 0.99 * (1.0 - np.mean(rewardsList) / target)

print(
    "Mean rewards of all the successful episodes: {}".format(np.mean(rewardsList))
    )
print(
    "Mean rewards of the last 1000 successful episodes: {}".format(
        np.mean(rewardsList[num_episodes - 1000:])
        )
    )
