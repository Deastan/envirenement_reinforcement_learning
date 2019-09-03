#!/usr/bin/env python2

import sys
import datetime
# if sys.version_info[0] < 3:
#         raise Exception("Must be using Python 3 on ROS")

import matplotlib.pyplot as plt
import math
import numpy as np
import time
import random
import os
import subprocess


#REINFORCEMENT LEARNING:
import gym
from gym import wrappers
from gym.envs.registration import register

# ROS packages required
import rospy
import rospkg
import roslaunch

# Machine learning
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

# save part
import pickle
from classes.robot_gazebo_env import RobotGazeboEnv
from tf.transformations import *

# Global variables:
here = os.path.dirname(os.path.abspath(__file__))

# Callback function
# The goal was to have a callback to see in real time some parameter of the NN
# from tensorflow import keras
tbCallBack = TensorBoard(log_dir='/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/learn_to_go_position/tensorboard', histogram_freq=0, write_graph=True, write_images=True)


def create_model(inputs=10, outputs=4, hidden_layers=2, neurons=64, LEARNING_RATE = 0.001):
    '''
    Create a model with defined parameters
    
    Inputs: 
        input is the observation space
        output is the action space
        number of layer: fix here
        neurons fix per layers
    Outputs:
    # Model params
    NN: 10 inputs, 4 outputs and 2 hidden layers
         ...
    s1 - ... - q_1
    s2 - ... - q_2
    s3 - ... - q_3
         ... - q_4
         ...

    '''

    model = Sequential()
    print("[ INFO]: Hidden layer 1 is created!")
    model.add(Dense(neurons, input_shape=(inputs,), activation="relu"))
    for i in range(hidden_layers-1):
        print("[ INFO]: Hidden layer ", str(i+2), " is created!")
        model.add(Dense(neurons, activation="relu"))
    model.add(Dense(outputs, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    return model

def load_trained_model(weights_path):
    '''
    Return the chosen trained model before from a defined weights file

    Output: model
    '''
    model = create_model(inputs=10, outputs=4, hidden_layers=2, neurons=64, LEARNING_RATE = 0.001)
    model.load_weights(weights_path)

    return model

def compute_exploration_decay(EXPLORATION_MAX, EXPLORATION_MIN, EPISODE_MAX, MAX_STEPS):
    '''
    Compute the exploration decay in function of Episode max

    Output: EXPLORATION_DECAY

    Calculation:
    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.999790 #0.9993 # Over 500 =>0.9908, for 2500 =>0.998107
    # Use exploration_rate = exploration_rate*EXPLORATION_DECAY
    # exploration decay = 10^(log(0.01)/EPISODE_MAX)
    '''
    return 10**(math.log10(EXPLORATION_MIN)/(EPISODE_MAX*MAX_STEPS/4*3))

# save the data
# TODO: Addapt the saved data function
#       Create a function to save every data using one function for training, evaluation, and application phase
def save(list_theta, episode, arg_path= 
        "/home/roboticlab14/catkin_ws/src/envirenement_reinforcement_learning/environment_package/src/saves/pickles/", 
        arg_name = "list_of_reward_"):
    '''
    Save a pickle file of a list that you want to save on the SSD (hard disk)

    input:  list_data list that you want to save
            episode current episode of the training
            arg_name is the name of the file
            arg_path is where you want to save.
            => it the path start with / that implies absolute path, and without mean relative from the utils.py is locatated.
    Output: bool, true if the saving worked false otherwise
            pickle file written on the hard dick


    '''
    saved = False
    try:
        path = arg_path
        name = arg_name
        full_path = path + name + str(episode) + ".pkl"
        with open(full_path, 'wb') as f:
            pickle.dump(list_theta, f, protocol=pickle.HIGHEST_PROTOCOL)
        saved = True
    except:
        print("Couldn't save the file .pkl")
    return saved

# TODO: Add the possibility to set the path
def init_folders(task="position_learning"):
    '''
    Create the sub directory: done, losses, memory, model, reward, trajectory

    return True if the subfolder are created
    '''
    now = datetime.datetime.now()
    path = "/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/"
    path = path + task + "/"
    
    # Add a Zeros to have every time the same type of folder title
    # Months:
    if int(now.month) < 10:   
        str_month = str(0) + str(now.month)
    else:
        str_month = str(now.month)
    # Days:
    if int(now.day) < 10:   
        str_day = str(0) + str(now.day)
    else:
        str_day = str(now.day)
    # Hour
    if int(now.hour) < 10:   
        str_hour = str(0) + str(now.hour)
    else:
        str_hour = str(now.hour)
    # Min
    if int(now.minute) < 10:   
        str_min = str(0) + str(now.minute)
    else:
        str_min = str(now.minute)
    # Sec
    if int(now.second) < 10:   
        str_sec = str(0) + str(now.second)
    else:
        str_sec = str(now.second)
    prefix_folder_name = str(now.year) + str_month + str_day + "_" + str_hour + str_min + str_sec + "_"
    folder_name = prefix_folder_name + task + "/" #"learn_to_go_position/"

    list_sub_folder_names = ["done", "losses_from_demos", "losses", "memory", "model_from_demos", "model", "reward", "trajectory", "mean_loss", "evaluation"]
    try:
        for names in list_sub_folder_names:
            path_sub_folders = path + folder_name + names
            os.makedirs(path_sub_folders)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        return False, path + folder_name
    else:
        print ("Successfully created the directories!")

    return True, path + folder_name

# TODO: add the missing parameters...
def create_summary(folder_path, task, EPISODE_MAX, MAX_STEPS, GAMMA, MEMORY_SIZE, BATCH_SIZE, EXPLORATION_MAX, EXPLORATION_MIN, EXPLORATION_DECAY, observation_space, action_space, hidden_layers, neurons, LEARNING_RATE, step_size):
    '''
    Create a summary in a text file of the used parameters for the learning 
    '''
    
    name = "summary.txt"
    full_path = folder_path + name
    
    try:
        f = open(full_path,"w+")
        f.write("####################################\n")
        f.write("############## Summary #############\n")
        f.write("####################################\n")
        f.write("\n")
        f.write("The trained task is " + task + "!\n")
        f.write("####################################\n")
        f.write("############# Parameters ###########\n")
        f.write("####################################\n")
        f.write("")
        f.write("Environment: \n")
        f.write("   Actions space: " + str(action_space) + "\n")
        f.write("   Observations space: " + str(observation_space) + "\n")
        f.write("\n")
        f.write("Reinforcement learning:\n")
        f.write("   Episodes max: " + str(EPISODE_MAX) + "\n")
        f.write("   Maximum step per episode: " + str(MAX_STEPS) + "\n")
        f.write("   End effector step size: " + str(step_size) + "\n")
        f.write("Q-learning: \n")
        f.write("   Gamma: " + str(GAMMA) + "\n")
        f.write("   Exploration min: " + str(EXPLORATION_MIN) + "\n")
        f.write("   Exploration max: " + str(EXPLORATION_MAX) + "\n")
        f.write("   Exploration decay: " + str(EXPLORATION_DECAY) + "\n")
        f.write("   Memory size: " + str(MEMORY_SIZE) + "\n")
        f.write("")
        f.write("Neural Network: \n")
        f.write("   Input: " + str(observation_space) + "\n")
        f.write("   Output: " + str(action_space) + "\n")
        f.write("   Hidden layers: " + str(hidden_layers) + "\n")
        f.write("   Neurons: " + str(neurons) + "\n")
        f.write("   Learning rate: " + str(LEARNING_RATE) + "\n")
        f.write("   Batch size: " + str(BATCH_SIZE) + "\n")
        
        f.close()
    except:
        print("[ ERROR: Cannot write the summary!]")
        return False
    return True