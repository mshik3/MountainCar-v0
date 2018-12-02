import argparse
import os

import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from time import time
import keras as K
envName = "MountainCar-v0"
env = gym.make(envName)

def play_model(actor):
    state = env.reset()
    score = 0
    done = False
    images = []
    R = 0
    t = 0
    while not done:
        state = np.reshape(state, [-1, env.observation_space.shape[0]])
        action = actor.predict(state)
        nextState, reward, done, _ = env.step(np.argmax(action))
        state = nextState
        score += reward
        if done:
            return score
    return 0

model = "save/MountainCar-v0_target_model_1543419126.88.h5"
totalIters = 100
expectedReward = -110

#Test
testScores = []
actor = K.models.load_model('{}'.format(model))
print("Saved model loaded from '{}'".format(model))
print("Starting testing.. Expecting reward to be {} over {} iterations".format(
    expectedReward, totalIters))
for itr in range(1, totalIters + 1):
    score = play_model(actor)
    testScores.append(score)
    print("Iteration: {}\tScore: {}".format(itr, score))
avg_reward = np.mean(testScores)
print("Total Avg. Score over {} consecutive iterations : {}".format(totalIters, avg_reward))
if avg_reward >= expectedReward:
    print("Agent finished test within expected reward boundary! Environment is solved.")
else:
    print("Agent has failed this test. Average score observed was {}".format(avg_reward))
