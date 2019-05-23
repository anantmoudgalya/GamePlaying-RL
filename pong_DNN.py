import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import fully_connected, input_data, dropout
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('Pong-ram-v0')
env.reset()
goal_steps = 1000
score_requirement = -20.0


initial_games = 1000

def some_random_games_first():
    scores = []
    for episode in range(5):
        env.reset()
        scores.append(0)
        for t in range(goal_steps):
            #env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            scores[episode] += reward
            if done:
                break
        print("Score for", episode+1, "episode:", scores[episode])
    print("Action Space: ", env.action_space.n)
    print("Observation Space: ", env.observation_space.shape[0])
    print("Reward Range: ", env.reward_range)

#some_random_games_first()

def initial_population():
    training_data = []
    # all scores:
    scores = []
    count = 0
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            action = env.action_space.sample()
            # do it!
            observation, reward, done, info = env.step(action)
            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
                #print("Action: ", action)
                #print("Game mem")

            prev_observation = observation
            score+=reward
            if done: 
                break
        if score >= score_requirement:
            #print("Accepted score")
            accepted_scores.append(score)
            print("Accepted: ", score)

        env.reset()
        scores.append(score)    
    

    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return accepted_scores

accepted_scores = initial_population()
print(accepted_scores[0])
