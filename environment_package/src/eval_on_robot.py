#!/usr/bin/env python2

import utils
import datetime
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
from keras.callbacks import TensorBoard

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
import termios, tty # for keyboard
# import environment_package.env_class.robot_gazebo_env
# from openai_ros.task_envs.fetch import fetch_test_task
# from home.roboticlab14.catkin_ws.src.envirenement_reinforcement_learning.environment_package.env_class.robot_gazebo_env import RobotGazeboEnv
# from environment_package import robot_gazebo_env
from classes.robot_gazebo_env import RobotGazeboEnv
from tf.transformations import *
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

def discrete_action(action, step_size = 0.01):
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

def action_to_string(action):
    '''
    Transform an action to a string to be more explicite about what is the direction
    '''
    string_action = ""
    if action == 0:
        string_action = "down"
    elif action == 1:
        string_action = "up"
    elif action == 2:
        string_action = "right"
    elif action == 3:
        string_action = "left"
    elif action == 4:
        string_action = "up in Z"
    elif action == 5:
        string_action = "down in Z"
    return string_action

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
        return model, exploration_rate
    batch = random.sample(memory, BATCH_SIZE)
    # list_q_
    for state, action, reward, new_state, done in batch:
        if not done:
            q_target = (reward + GAMMA * np.amax(model.predict(new_state)))
        else:
            q_target = reward

        q_values = model.predict(state)
        q_values[0][action] = q_target
        model.fit(state, q_values, verbose=0)#, callbacks=[tbCallBack])

    exploration_rate *= EXPLORATION_DECAY
    # print("Exploration rate: ", exploration_rate)
    exploration_rate = max(EXPLORATION_MIN, exploration_rate)
    return model, exploration_rate

def experience_replay_v2(model, memory, 
      BATCH_SIZE, exploration_rate, EXPLORATION_DECAY, EXPLORATION_MIN, GAMMA):
    '''
    Train the NN
    '''
    if len(memory) < BATCH_SIZE:
        return model, exploration_rate, 0
    batch = random.sample(memory, BATCH_SIZE)
    np_list_states = np.zeros((BATCH_SIZE, 10))
    np_list_q_values = np.zeros((BATCH_SIZE, 4))
    i = 0
    for state, action, reward, new_state, done in batch:
        if not done:
            q_target = (reward + GAMMA * np.amax(model.predict(new_state)))
        else:
            q_target = reward

        q_values = model.predict(state)
        q_values[0][action] = q_target
        np_list_states[i] = state
        np_list_q_values[i] = q_values[0]
        i+=1
    history = model.fit(np_list_states, np_list_q_values, epochs=2,  batch_size=BATCH_SIZE, verbose=0)#, callbacks=[tbCallBack])
    # print("Loss is: ", history.history)
    # print(history.history['loss'])

    #FIXE THE EXPLORATION RATE!!!!
    # exploration_rate *= EXPLORATION_DECAY
    # print("Exploration rate: ", exploration_rate)
    # exploration_rate = max(EXPLORATION_MIN, exploration_rate)
    exploration_rate = 0.1
    return model, exploration_rate, history.history['loss']

def experience_evaluating(model, state, action, reward, new_state, done, GAMMA):

    if not done:
        q_target = (reward + GAMMA * np.amax(model.predict(new_state)))
    else:
        q_target = reward

    q_values = model.predict(state)
    q_values[0][action] = q_target
    
    history = model.evaluate(state, q_values, verbose=0)

    print("Loss is: ", history)
    return history

def use_model(env, model, MAX_STEPS, observation_space, step_size, GAMMA):
    '''
    Use a trained model 
    '''
    # Parameters;
    env.set_mode(mode="evaluating")
    list_target_points = []
    list_target_points.append([0.5, -0.3, 0.1])
    list_target_points.append([0.6, -0.0, 0.1])
    list_target_points.append([0.4, 0.3, 0.1])
    # list_target_points.append([0.4, -0.4, 0.1])
    # list_target_points.append([0.6, 0.1, 0.1])
    # list_target_points.append([0.6, -0.1, 0.1])
    # list_target_points.append([0.5, -0.2, 0.1])
    np_target_points = np.array(list_target_points)
    epidodes_max = len(list_target_points)
    done = False
    list_loss = [] # to save for compute the mean of this model
    for i in range(epidodes_max):
        j = 0
        env.set_for_evaluation(np_target_points[i])
        rospy.sleep(0.05)
        state = env.reset()
        rospy.sleep(5.0)
        np_state = (np.array(state, dtype=np.float32),)
        np_state = np.reshape(np_state, [1, observation_space])
        
        while j < MAX_STEPS and not done:
            q_values = model.predict(np_state)
            action = np.argmax(q_values[0])
            disc_action = discrete_action(action, step_size)
            new_state, reward, done, info = env.step(disc_action)
            print("*********************************************")
            print("Observation: ", new_state)
            print("Action: ", action_to_string(action))
            print("Reward: ", reward)
            print("Done: ", done)
            print("Episode: ", i)
            print("Step: ", j)
            print("*********************************************")
            np_new_state = (np.array(new_state, dtype=np.float32),)
            np_new_state = np.reshape(np_new_state, [1, observation_space])

            loss = experience_evaluating(model, np_state, action, reward, np_new_state, done, GAMMA)
            list_loss.append(loss)
            np_state = np_new_state
            state = new_state
            j+=1
        i+=1
    return sum(list_loss)/len(list_loss)

if __name__ == '__main__':
    main()