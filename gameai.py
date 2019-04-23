import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import fully_connected, input_data, dropout
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('Atlantis-ram-v0')
env.reset()
goal_steps = 200
score_requirement = 500.0
initial_games = 1000

def some_random_games_first():
    valid_actions = []
    actions = []
    rtot = 0
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            actions.append(action)
            if done:
                break
            rtot += reward
            if reward or (not reward) :
                print("Observation: ", observation)
                print("Reward: ", reward)
                print("Info: ", info)
                print("Valid Action: ", action)
                valid_actions.append(action)
    print("Valid: ", set(valid_actions)) 
    print("Total Reward: ", rtot)
    print("Total actions: ", set(actions))
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
                if data[1] == 1:
                    output = [0,1,0,0]
                elif data[1] == 0:
                    output = [1,0,0,0]
                elif data[1] == 2:
                    output = [0,0,1,0]
                elif data[1] == 3:
                    output = [0,0,0,1]

                print("Action: ", data[1], "Output: ", output)
                training_data.append([data[0], output])

        env.reset()
        scores.append(score)    
    
    training_data_save = np.array(training_data)
    np.save('atlantis_save.npy', training_data_save)

    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

#training_data = initial_population()


def neural_network_model(input_size):
    
    network = input_data(shape=[None,input_size,1], name="input")

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 4, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=LR, 
    loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model = False):
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]
    
    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({'input': X}, {'targets' : y}, n_epoch = 10, show_metric = True, 
    run_id='atlantis')

    return model

training_data = np.load("atlantis_save.npy")
print("Train_shape: ", training_data.shape)
model = train_model(training_data)
print("Model Done")