import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import fully_connected, input_data, dropout
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('Boxing-v0')
env.reset()
goal_steps = 200
score_requirement = 0.0
initial_games = 500

def some_random_games_first():
    valid_actions = []
    rtot = 0
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break
            rtot += reward
            if reward == 1.0:
                '''print("Reward", reward)
                print("Info", info)
                print("Valid Action", action)'''
                valid_actions.append(action)
    print("Valid: ", set(valid_actions)) 
    print("Total Reward: ", rtot)
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
            for data in game_memory:
                #print("Data: ", type(data), "Length", len(data), "Action", data[1])
                # convert to one-hot (this is the output layer for our neural network)
                for i in [data[1],]:
                    output = [0]*18
                    output[i] = 1
                    #print("Action: ", data[1], "Output: ", output)
               
                training_data.append([data[0], output])

        env.reset()
        scores.append(score)    
    
    '''training_data_save = np.array(training_data)
    np.save('boxing_save.npy', training_data_save)'''

    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

def neural_network_model(input_size):
    
    network = input_data(shape=[None,input_size,1], name="input")

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)





'''training_data = initial_population()
training_data = np.array(training_data)
print(training_data.shape)'''