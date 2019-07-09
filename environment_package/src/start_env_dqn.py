#!/usr/bin/env python2

import sys
# if sys.version_info[0] < 3:
#         raise Exception("Must be using Python 3 on ROS")

# import tensorflow as tf
import gym
import numpy as np
import time
# import qlearn
import random
from gym import wrappers
from gym.envs.registration import register
# ROS packages required
import rospy
import rospkg

#REINFORCEMENT LEARNING:
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# import our training environment
# from openai_ros.task_envs.iiwa_tasks import iiwa_move
# from openai_ros.task_envs.hopper import hopper_stay_up
# import pickle, os

# from baselines import PPO2
# from run_algo import start_learning
import subprocess

# For the launch 
import roslaunch
import os

import git
import sys
# save part
import pickle
import matplotlib.pyplot as plt
import termios, tty # for keyboard
# import environment_package.env_class.robot_gazebo_env
# from openai_ros.task_envs.fetch import fetch_test_task
# from home.roboticlab14.catkin_ws.src.envirenement_reinforcement_learning.environment_package.env_class.robot_gazebo_env import RobotGazeboEnv
# from environment_package import robot_gazebo_env
from classes.robot_gazebo_env import RobotGazeboEnv

# Global variables:
here = os.path.dirname(os.path.abspath(__file__))

def init_env():
    '''
    Init the environment
    '''
    # Cheating with the registration 
    # (shortcut: look at openai_ros_common.py and task_envs_list.py)
    timestep_limit_per_episode = 10000000
    register(
        id="RobotGazeboEnv-v0",
        entry_point = 'classes.robot_gazebo_env:RobotGazeboEnv',
        max_episode_steps=timestep_limit_per_episode,
    )

    # Create the Gym environment
    env = gym.make('RobotGazeboEnv-v0')
    print("Gym environment done: RobotGazeboEnv-v0")
    return env

def discrete_action(action):
    '''
    Transform the asking action to a discretize action to simplify the problem 
    0 = + step_size * x
    1 = - step_size * x
    2 = + step_size * y
    3 = + step_size * y
    4 = + step_size * z
    5 = - step_size * z
    '''
    a = [0, 0, 0, 0, 0, 0]
    step_size = 0.1
    if action == 0:
        a[action] = step_size
    elif action == 1:
        a[0] = -step_size
    elif action == 2:
        a[1] = step_size
    elif action == 3:
        a[1] = -step_size
    elif action == 4:
        a[2] = step_size
    elif action == 5:
        a[2] = -step_size
    return a

# save the data
def save(list_theta, epoch):
    saved = False
    try:

        name = "/home/roboticlab14/catkin_ws/src/envirenement_reinforcement_learning/environment_package/src/saves/pickles/list_of_reward_" + str(epoch) + ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(list_theta, f, protocol=pickle.HIGHEST_PROTOCOL)
        saved = True
    except:
        print("Couldn't save the file .pkl")
    return saved

def choose_action(model, state, action_space, exploration_rate):
    '''
    Choose an action using Epsilon-Greedy exploration/exploitation
    '''
    if np.random.rand() < exploration_rate:
        return random.randrange(action_space)
    q_values = model.predict(state)
    return np.argmax(q_values[0])

def save_forReplay(memory, state, action, reward, new_state, done):
    '''
    Experience saving in a table
    '''
    memory.append((state, action, reward, new_state, done))
    # print(memory)
    return memory

def experience_replay(model, memory, 
      BATCH_SIZE, exploration_rate, EXPLORATION_DECAY, EXPLORATION_MIN, GAMMA):
    '''
    Train the NN
    '''
    if len(memory) < BATCH_SIZE:
        return
    # print("***********************************************")
    # print(memory)
    batch = random.sample(memory, BATCH_SIZE)
    # print("***********************************************")
    # print(batch)
    # print("***********************************************")
    # print(batch)
    for state, action, reward, new_state, done in batch:
        # print("Inside replay")
        if not done:
            # print(new_state)
            # print(model.predict(new_state))
            q_target = (reward + GAMMA * np.amax(model.predict(new_state)))
        else:
            q_target = reward

        print(q_target)
        q_values = model.predict(state)
        print(q_values)
        print("action: ", action)
        q_values[0][action] = q_target
        print("Training the model...")
        model.fit(state, q_values, verbose=0)#, callbacks=[tensorboard])

    exploration_rate *= EXPLORATION_DECAY
    exploration_rate = max(EXPLORATION_MIN, exploration_rate)
    print(exploration_rate)
    # return model

# Neural network with 2 hidden layer using Keras + experience replay
def DDN_learning_keras_memoryReplay(env):
    '''
    function which is a neural net using Keras with memory replay
    '''
    EPISODE_MAX = 600
    MAX_STEPS = 15
    #PARAMS
    GAMMA = 0.95
    LEARNING_RATE = 0.001
    MEMORY_SIZE = EPISODE_MAX
    BATCH_SIZE = 20
    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.995

    # Env params
    observation_space = 7
    # observation_space = env.observation_space
    action_space = 4

    exploration_rate = EXPLORATION_MAX

    # Model params
    '''
    NN: 7 inputs, 4 outputs
         ...
    s1 - ... - q_1
    s2 - ... - q_2
    s3 - ... - q_3
         ... - q_4
         ...

    '''
    model = Sequential()
    model.add(Dense(16, input_shape=(observation_space,), activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(action_space, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    # Experience replay:
    memory = deque(maxlen=MEMORY_SIZE)
    list_total_reward = []
    episode_max = EPISODE_MAX
    done = False
    for i in range(episode_max):
        total_reward = 0
        j = 0
        state = env.reset()
        # Becarefull with the state (np and list)
        np_state = (np.array(state, dtype=np.float32),)
        np_state = np.reshape(np_state, [1, observation_space])
        # np_state = np.identity(7)[np_state:np_state+1]
        # print(np_state)
        done = False
        while j < MAX_STEPS and not done: # step inside an episode
            j+=1
            # print("[INFO]: episode: ", i, ", step: ", j)

            # BECAREFUL with the action!
            action = choose_action(model, state, action_space, exploration_rate)
            disc_action = discrete_action(action)
            # print(action)
            new_state, reward, done, info = env.step(disc_action)
            print("*********************************************")
            print("Observation: ", new_state)
            # print("State: ", state)
            print("Reward: ", reward)
            print("Total rewards: ", total_reward)
            print("Done: ", done)
            # print("# dones: ", done_increment)
            # print("Info: ", info)
            # print("Action: ",  action)
            print("Episode: ", i)
            print("Step: ", j)
            print("*********************************************")
            np_new_state = (np.array(new_state, dtype=np.float32),)
            np_new_state = np.reshape(np_new_state, [1, observation_space])
            # np_new_state = np.identity(7)[np_new_state:np_new_state+1]

            # Momory replay
            # memory = save_forReplay(memory, state, action, reward, new_state, done)
            save_forReplay(memory, np_state, action, reward, np_new_state, done)
            experience_replay(model, memory, 
                BATCH_SIZE, exploration_rate, EXPLORATION_DECAY, EXPLORATION_MIN, GAMMA)

            np_state = np_new_state
            total_reward += reward
        
        list_total_reward.append(total_reward)
        print("Saving...", save(list_total_reward, i))
        
        if i%20:
            # Save the model
            print("Saving...")
            model.save('/home/roboticlab14/catkin_ws/src/envirenement_reinforcement_learning/environment_package/src/saves/model/try_1.h5')
            # Save datas
            print("Saving...", save(list_total_reward, i))

    
    #Test algo
    # print("***********Prediction***************")
    # prediction = True
    # if prediction == True:
    #     done = False
    #     t = 0
    #     state = env.reset()
    #     state = np.identity(16)[state:state+1]
    #     while ((not done) and t < 15):
    #         #ACTION TO SET
    #         action = model.predict(state)
    #         new_state, reward, done, _ = take_action(np.argmax(action), env)
    #         new_state = np.identity(16)[new_state:new_state+1]         
    #         env.render()
    #         state = new_state
    #         t+=1
        # print(W1)
    #end function


# load_weights only sets the weights of your network. You still need to define its architecture before calling load_weights:
def create_model():
    '''
    Create a model with defined parameters
    '''

    EPISODE_MAX = 600
    MAX_STEPS = 15
    #PARAMS
    GAMMA = 0.95
    LEARNING_RATE = 0.001
    MEMORY_SIZE = EPISODE_MAX
    BATCH_SIZE = 20
    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.995

    # Env params
    observation_space = 7
    # observation_space = env.observation_space
    action_space = 4

    exploration_rate = EXPLORATION_MAX

    model = Sequential()
    model.add(Dense(16, input_shape=(observation_space,), activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(action_space, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    return model

def load_trained_model(weights_path):
    '''
    Return the model pretrained

    '''
    model = create_model()
    model.load_weights(weights_path)

    return model


def main():
    rospy.init_node('training_node', anonymous=True, log_level=rospy.WARN)
    print("Start")
    # env = init_env()
    # env.reset()
    env = init_env()
    # env.reset()
    # qlearning(env)
    # catch_object(env)
    DDN_learning_keras_memoryReplay(env)
    print("end")

if __name__ == '__main__':
    main()


def qlearning(env):
    # Parameters:
    MAX_STEPS = 50
    MAX_EPOCHS = 25
    
    # while True:
    for epoch in range(0, MAX_EPOCHS):
        
        last_obs = env.reset()
        for step in range(0, MAX_STEPS):
            # discrete_act = env.action_space.sample()
            discrete_act = 1
            action = discrete_action(discrete_act)
            # action = [0, 0.0, 0, 0.01, 0, 0]
            obs, reward, done, info = env.step(action)
            print("*********************************************")
            print("Observation: ", obs)
            # print("State: ", state)
            print("Reward: ", reward)
            print("Done: ", done)
            # print("# dones: ", done_increment)
            print("Info: ", info)
            # print("Action: ",  action)
            print("Episode: ", epoch)
            print("Step: ", step)
            print("*********************************************")

def catch_object(env):
    env.reset()
    env.set_endEffector_pose([0.5, 0.0, 0.1, 3.1457, 0.0, 0.0])
    rospy.sleep(15)
    env.set_endEffector_pose([0.5, 0.0, 0.3, 3.1457, 0.0, 0.0])
    env.set_endEffector_pose([0.0, 0.5, 0.30, 3.1457, 0.0, 0.0])




# load_weights only sets the weights of your network. You still need to define its architecture before calling load_weights:

# def create_model():
#    model = Sequential()
#    model.add(Dense(64, input_dim=14, init='uniform'))
#    model.add(LeakyReLU(alpha=0.3))
#    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
#    model.add(Dropout(0.5)) 
#    model.add(Dense(64, init='uniform'))
#    model.add(LeakyReLU(alpha=0.3))
#    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
#    model.add(Dropout(0.5))
#    model.add(Dense(2, init='uniform'))
#    model.add(Activation('softmax'))
#    return model

# def train():
#    model = create_model()
#    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(loss='binary_crossentropy', optimizer=sgd)

#    checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
#    model.fit(X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True, validation_split=0.2, verbose=2, callbacks=[checkpointer])

# def load_trained_model(weights_path):
#    model = create_model()
#    model.load_weights(weights_path)



