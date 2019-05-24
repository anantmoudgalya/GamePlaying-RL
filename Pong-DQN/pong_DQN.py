import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

ENV_NAME = "Pong-ramDeterministic-v4"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 10

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.02
EXPLORATION_DECAY = 0.995


def some_random_games_first():
    env = gym.make(ENV_NAME)
    scores = []
    for episode in range(5):
        step_count = 0
        observations = []
        env.reset()
        scores.append(0)
        for t in range(10000):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            scores[episode] += reward
            step_count += 1
            observations.append(observation)
            if done:
                break
        print("Score for", episode+1, "episode:", scores[episode])
        print("Steps for", episode+1, "episode: ", step_count)
        #print("Final State: ", observations[len(observations)-1], "\n")
    print("Action Space: ", env.action_space.n)
    print("Observation Space: ", env.observation_space.shape[0])
    print("Reward Range: ", env.reward_range)

some_random_games_first()

def game_memory(state, reward, next_state, ):
    memory = []
    memory    