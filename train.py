import argparse
import os

import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from time import time
import keras as K
import random
from collections import namedtuple, deque
import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display

env_name = None
initial_timestamp = 0.0
np.random.seed(1024)

class DQNetwork:

    def __init__(self, numStates, numActions, maxAction=1.0, minAction=0.0, layerSizes=(64, 64),
                 batchNormPerLayer=(True, True), dropoutPerLayer=(0, 0), learningRate=0.0001):
        self.numStates = numStates
        self.numActions = numActions
        self.maxAction = maxAction
        self.minAction = minAction
        self.layerSizes = layerSizes
        self.batchNormPerLayer = batchNormPerLayer
        self.dropoutPerLayer = dropoutPerLayer
        self.learningRate = learningRate

        self.build_model()

    def build_model(self):
        states = K.layers.Input(shape=(self.numStates,), name='states')
        neuralNet = states
        # hidden layers

        for i in range(len(self.layerSizes)):
            neuralNet = K.layers.Dense(units=self.layerSizes[i])(neuralNet)
            neuralNet = K.layers.Activation('relu')(neuralNet)
            if self.batchNormPerLayer[i]:
                neuralNet = K.layers.BatchNormalization()(neuralNet)
            neuralNet = K.layers.Dropout(self.dropoutPerLayer[i])(neuralNet)

        actions = K.layers.Dense(units=self.numActions, activation='linear',
                                 name='rawActions')(neuralNet)

        self.model = K.models.Model(inputs=states, outputs=actions)

        self.optimizer = K.optimizers.Adam(lr=self.learningRate)
        self.model.compile(loss='mse', optimizer=self.optimizer)

class DDQNAgent:

    def __init__(self, env, bufferSize=int(1e5), batchSize=64, gamma=0.99, tau=1e-3, lr=5e-4, callbacks=()):
        self.env = env
        self.env.seed(1024)
        self.batchSize = batchSize
        self.gamma = gamma
        self.tau = tau
        self.qTargets = 0.0
        self.numStates = env.observation_space.shape[0]
        self.numActions = env.action_space.n
        self.callbacks = callbacks

        layerSizes = [256, 256]
        batchNormPerLayer = [False, False]
        dropoutPerLayer = [0, 0]

        print("Initialising DDQN Agent with params : {}".format(self.__dict__))

        # Make local & target model
        self.localNetwork = DQNetwork(self.numStates, self.numActions,
                                       layerSizes=layerSizes,
                                       batchNormPerLayer=batchNormPerLayer,
                                       dropoutPerLayer=dropoutPerLayer,
                                       learningRate=lr)
        print("Finished initializing local network.")
        self.targetNetwork = DQNetwork(self.numStates, self.numActions,
                                        layerSizes=layerSizes,
                                        batchNormPerLayer=batchNormPerLayer,
                                        dropoutPerLayer=dropoutPerLayer,
                                        learningRate=lr)
        print("Finished initializing target network")
        self.memory = ReplayBuffer(bufferSize=bufferSize, batchSize=batchSize)

    def resetEpisode(self):
        state = self.env.reset()
        self.prevState = state
        return state

    def step(self, action, reward, nextState, done):
        self.memory.add(self.prevState, action, reward, nextState, done)

        if len(self.memory) > self.batchSize:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

        self.prevState = nextState

    def act(self, state, eps=0.):
        state = np.reshape(state, [-1, self.numStates])
        action = self.localNetwork.model.predict(state)

        if random.random() > eps:
            return np.argmax(action)
        else:
            return random.choice(np.arange(self.numActions))

    def learn(self, experiences, gamma):
        states, actions, rewards, nextStates, dones = experiences

        for itr in range(len(states)):
            state, action, reward, nextState, done = states[itr], actions[itr], rewards[itr], nextStates[itr], dones[
                itr]
            state = np.reshape(state, [-1, self.numStates])
            nextState = np.reshape(nextState, [-1, self.numStates])

            self.qTargets = self.localNetwork.model.predict(state)
            if done:
                self.qTargets[0][action] = reward
            else:
                nextQ = self.targetNetwork.model.predict(nextState)[0]
                self.qTargets[0][action] = (reward + gamma * np.max(nextQ))

            self.localNetwork.model.fit(state, self.qTargets, epochs=1, verbose=0, callbacks=self.callbacks)

    def updateTargetModel(self):
        self.targetNetwork.model.set_weights(self.localNetwork.model.get_weights())


class ReplayBuffer:

    def __init__(self, bufferSize, batchSize):
        self.memory = deque(maxlen=bufferSize)  # internal memory (deque)
        self.batchSize = batchSize
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "nextState", "done"])

    def add(self, state, action, reward, nextState, done):
        e = self.experience(state, action, reward, nextState, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batchSize)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        nextStates = np.vstack([e.nextState for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, nextStates, dones)

    def __len__(self):
        return len(self.memory)

def train(numEpisodes=2000, startEpsilon=1.0, endEpsilon=0.001, epsDecayRate=0.9, targetReward=1000):
    scores = []
    scoresWindow = deque(maxlen=100)
    eps = startEpsilon
    print("Starting model training for {} episodes.".format(numEpisodes))
    avgScoreGreaterThanTargetCounter = 0
    for episode in range(1, numEpisodes + 1):
        initialTime = time()
        state = agent.resetEpisode()
        score = 0
        done = False
        while not done:
            action = agent.act(state, eps)
            nextState, reward, done, _ = env.step(action)
            agent.step(action, reward, nextState, done)
            state = nextState
            score += reward
            if done:
                agent.updateTargetModel()
                break
        timeTaken = time() - initialTime
        scoresWindow.append(score)
        scores.append(score)
        eps = max(endEpsilon, epsDecayRate * eps)
        print('Episode {}\tTime Taken: {:.2f} sec\tScore: {:.2f}\tState: {}\tAverage Q-Target: {:.4f}'
                     '\tEpsilon: {:.3f}\tAverage Score: {:.2f}\t'.format(
            episode, timeTaken, score, state[0], np.mean(agent.qTargets), eps, np.mean(scoresWindow)))
        if episode % 100 == 0:
            print(
                'Episode {}\tTime Taken: {:.2f} sec\tScore: {:.2f}\tState: {}\tAverage Q-Target: {:.4f}\tAverage Score: {:.2f}'.format(
                    episode, timeTaken, score, state[0], np.mean(agent.qTargets), np.mean(scoresWindow)))
            agent.localNetwork.model.save('save/{}_local_model_{}.h5'.format(envName, initialTime))
            agent.targetNetwork.model.save('save/{}_target_model_{}.h5'.format(envName, initialTime))
        if np.mean(scoresWindow) >= targetReward:
            avgScoreGreaterThanTargetCounter += 1
            if avgScoreGreaterThanTargetCounter >= 5:
                print("Model training finished! \nAverage Score over last 100 episodes: {}\tNumber of Episodes: {}".format(
                    np.mean(scoresWindow), episode))
                return scores
        else:
            avgScoreGreaterThanTargetCounter = 0
    print("Model training finished! \nAverage Score over last 100 episodes: {}\tNumber of Episodes: {}".format(
        np.mean(scoresWindow), numEpisodes))
    return scores


def play_model(actor, renderEnv=False, shouldReturnImages=False):
    state = env.reset()
    score = 0
    done = False
    images = []
    R = 0
    t = 0
    while not done:
        if renderEnv:
            if shouldReturnImages:
                images.append(env.render("rgb_array"))
            else:
                env.render()
        state = np.reshape(state, [-1, env.observation_space.shape[0]])
        action = actor.predict(state)
        nextState, reward, done, _ = env.step(np.argmax(action))
        state = nextState
        score += reward
        if done:
            return score, images
    return 0, images

def displayFramesAsGif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))

#train
envName = "MountainCar-v0"
env = gym.make(envName)
agent = DDQNAgent(env, bufferSize=100000, gamma=0.99, batchSize=64, lr=0.0001, callbacks=[])
scores = train(numEpisodes=2000, targetReward=-110, epsDecayRate=0.9)
