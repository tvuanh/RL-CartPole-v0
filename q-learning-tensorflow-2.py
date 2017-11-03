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
gamma = 0.9
epsilon = 0.1
num_episodes = 2000
# lists containing total rewards and total number of steps per episode
stepsList, rewardsList = list(), list()
# experience replay
replay_memory = list()
BATCH_SIZE = 50
MAX_LEN_REPLAY_MEMORY = 30000
MIN_FRAMES_FOR_LEARNING = 100

with tf.Session() as sess:
    # establish the feed-forward part of the network used to choose actions
    input = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    W1 = tf.Variable(tf.random_normal([4, 32], stddev=0.5))
    b1 = tf.Variable(tf.zeros([32]))
    hidden = tf.nn.sigmoid(tf.matmul(input, W1) + b1)
    W2 = tf.Variable(tf.random_normal([32, 2], stddev=0.5))
    b2 = tf.Variable(tf.zeros([2]))
    estimatedQ = tf.matmul(hidden, W2) + b2
    greedyAction = tf.argmax(estimatedQ, axis=1)

    # train the network using the quadratic loss function
    nextQ = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - estimatedQ))
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

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
            replay_memory.append((current_state, action[0], reward, done, next_state))
            if len(replay_memory) > MIN_FRAMES_FOR_LEARNING:
                experiences = random.sample(replay_memory, BATCH_SIZE)
                train_actions, targetQs = sess.run(
                    [greedyAction, estimatedQ], feed_dict={
                        input: [exp[4] for exp in experiences]
                        }
                    )
                train_states = list()
                for index in range(BATCH_SIZE):
                    state, _, r, terminal, _ = experiences[index]
                    targetQs[index, train_actions[index]] = r + terminal * np.max(targetQs[index])
                    train_states.append(state)
                sess.run(
                    optimiser, feed_dict={
                        input: train_states,
                        nextQ: targetQs
                        }
                    )
            if len(replay_memory) > MAX_LEN_REPLAY_MEMORY:
                replay_memory = replay_memory[1:]

            rewardAll += reward
            current_state = next_state
        stepsList.append(steps)
        rewardsList.append(rewardAll)

print(
    "Mean rewards of all the successful episodes: {}".format(np.mean(rewardsList))
    )
print(
    "Mean rewards of the last 1000 successful episodes: {}".format(
        np.mean(rewardsList[num_episodes - 1000:])
        )
    )
