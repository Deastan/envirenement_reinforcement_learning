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

# Callback function
# from tensorflow import keras
tbCallBack = TensorBoard(log_dir='/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/learn_to_go_position/tensorboard', histogram_freq=0, write_graph=True, write_images=True)

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
def save(list_theta, episode, arg_path= 
        "/home/roboticlab14/catkin_ws/src/envirenement_reinforcement_learning/environment_package/src/saves/pickles/", 
        arg_name = "list_of_reward_"):
    '''
    Save a pickle file of a list that you want to save.

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
        model.fit(state, q_values, verbose=0, callbacks=[tbCallBack])

    exploration_rate *= EXPLORATION_DECAY
    print("Exploration rate: ", exploration_rate)
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
    history = model.fit(np_list_states, np_list_q_values,  batch_size=BATCH_SIZE, verbose=0)#, callbacks=[tbCallBack])
    # print("Loss is: ", history.history)
    # print(history.history['loss'])
    exploration_rate *= EXPLORATION_DECAY
    print("Exploration rate: ", exploration_rate)
    exploration_rate = max(EXPLORATION_MIN, exploration_rate)
    return model, exploration_rate, history.history['loss']

def training_from_demo(model, memory_from_demo, GAMMA):
    for state, action, reward, new_state, done in memory_from_demo:
        if not done:
            q_target = (reward + GAMMA * np.amax(model.predict(new_state)))
        else:
            q_target = reward

        q_values = model.predict(state)
        q_values[0][action] = q_target
        model.fit(state, q_values, verbose=0)
    return model

def create_demo(env):
    # constant_z = 0.1
    action = [1, 3, 3, 3, 3, 3, 3]
    demo_epoch = 20
    demo_step = len(action)
    observation_space = 7
    deck_size = 2*demo_step*demo_epoch
    # q_interm = quaternion_from_euler(3.14, 0.0, 0.0)
    memory_from_demo = deque(maxlen=deck_size)

    #PARAMS
    GAMMA = 0.95
    LEARNING_RATE = 0.001
    BATCH_SIZE = 20
    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.995
    # state = [0.6, 0.1, constant_z, q_interm[0], q_interm[1], q_interm[2], q_interm[3]]
    # np_state = (np.array(state, dtype=np.float32),)
    # np_state = np.reshape(np_state, [1, observation_space])
    # '''
    # 0 = + step_size * x droite
    # 1 = - step_size * x gauche
    # 2 = + step_size * y haut
    # 3 = + step_size * y bas
    # in summury: links, down down down down
    # '''
    # action = 1
    # done = env._is_done(state)
    # reward = env._compute_reward(state, done)
    
    for i in range(0, demo_epoch):
        j = 0
        state = env.reset()
        np_state = (np.array(state, dtype=np.float32),)
        np_state = np.reshape(np_state, [1, observation_space])
        for j in range(0, demo_step):

            
            
            disc_action = discrete_action(action[j])
            new_state, reward, done, info = env.step(disc_action)
            print("*********************************************")
            print("Observation: ", new_state)
            print("Reward: ", reward)
            print("Done: ", done)
            print("Episode: ", i)
            print("Step: ", j)
            print("*********************************************")
            np_new_state = (np.array(new_state, dtype=np.float32),)
            np_new_state = np.reshape(np_new_state, [1, observation_space])
            memory_from_demo.append((np_state, action, reward, np_new_state, done))
            np_state = np_new_state
            state = new_state
            j+=1
        i+=1
    action = [3, 3, 3, 3, 3, 3, 1]
    i = 0

    for i in range(0, demo_epoch):
        j = 0
        state = env.reset()
        np_state = (np.array(state, dtype=np.float32),)
        np_state = np.reshape(np_state, [1, observation_space])
        for j in range(0, demo_step):

            
            
            disc_action = discrete_action(action[j])
            new_state, reward, done, info = env.step(disc_action)
            print("*********************************************")
            print("Observation: ", new_state)
            print("Reward: ", reward)
            print("Done: ", done)
            print("Episode: ", i)
            print("Step: ", j)
            print("*********************************************")
            np_new_state = (np.array(new_state, dtype=np.float32),)
            np_new_state = np.reshape(np_new_state, [1, observation_space])
            memory_from_demo.append((np_state, action, reward, np_new_state, done))
            np_state = np_new_state
            state = new_state
            j+=1
        i+=1
    
    return memory_from_demo

# Neural network with 2 hidden layer using Keras + experience replay
def dqn_learning_keras_memoryReplay(env, model):
    '''
    function which is a neural net using Keras with memory replay
    '''

    # Time optimzation:
    
    EPISODE_MAX = 501
    MAX_STEPS = 15
    #PARAMS
    GAMMA = 0.95
    MEMORY_SIZE = EPISODE_MAX
    BATCH_SIZE = 128
    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.9993 # Over 500 =>0.9908, for 2500 =>0.998107
    # Use exploration_rate = exploration_rate*EXPLORATION_DECAY
    # exploration decay = 10^(log(0.01)/EPISODE_MAX)

    # Env params
    observation_space = 10
    action_space = 4

    exploration_rate = EXPLORATION_MAX

    # Experience replay:
    memory = deque(maxlen=MEMORY_SIZE)

    # Create list for plot and saving
    list_memory_history = []
    list_total_reward = []
    list_loss = []
    episode_max = EPISODE_MAX

    done = False

    for i in range(episode_max):
        time_start_epoch = time.time()
        total_reward = 0
        j = 0
        state = env.reset()
        rospy.sleep(2.0)
        # Becarefull with the state (np and list)
        np_state = (np.array(state, dtype=np.float32),)
        np_state = np.reshape(np_state, [1, observation_space])
        # np_state = np.identity(7)[np_state:np_state+1]
        # print(np_state)
        done = False
        list_memory = []
        
        while j < MAX_STEPS and not done: # step inside an episode
            time_start_step = time.time()
            
            # print("[INFO]: episode: ", i, ", step: ", j)

            # BECAREFUL with the action!
            action = choose_action(model, np_state, action_space, exploration_rate)
            disc_action = discrete_action(action)
            # print(action)
            time_start_action = time.time()
            print("################INSIDE ENV################")
            new_state, reward, done, info = env.step(disc_action)
            print("################INSIDE ENV################")
            print("[ INFO] Time for the action ", j, ": ", time.time()-time_start_action)
            time_start_print = time.time()
            print("*********************************************")
            print("Observation: ", new_state)
            # print("State: ", state)
            print("Reward: ", reward)
            print("Total rewards: ", total_reward)
            print("Done: ", done)
            print("Episode: ", i)
            print("Step: ", j)
            print("*********************************************")
            print("[ INFO] Time for the print info ", j, ": ", time.time()-time_start_print)
            np_new_state = (np.array(new_state, dtype=np.float32),)
            np_new_state = np.reshape(np_new_state, [1, observation_space])
            # np_new_state = np.identity(7)[np_new_state:np_new_state+1]

            # Momory replay
            # memory = save_forReplay(memory, state, action, reward, new_state, done)
            time_start_save_for_replay = time.time()
            save_forReplay(memory, np_state, action, reward, np_new_state, done)
            print("[ INFO] Time for the saving of experience replay ", j, ": ", time.time()-time_start_save_for_replay)
            time_start_experience_replay = time.time()
            model, exploration_rate_new, loss = experience_replay_v2(model, memory, 
                BATCH_SIZE, exploration_rate, EXPLORATION_DECAY, EXPLORATION_MIN, GAMMA)
            print("[ INFO] Time for the experience replay ", j, ": ", time.time()-time_start_experience_replay)
            
            list_memory.append([np_state, action, reward, np_new_state, done])
            # Save the lost function into a list
            if len(memory) > BATCH_SIZE:
                list_loss.append(loss[0])
            # print("list_loss: ", list_loss)
            np_state = np_new_state
            state = new_state
            total_reward += reward
            exploration_rate = exploration_rate_new
            
            print("[ INFO] Time for the step ", j, ": ", time.time()-time_start_step)
            j+=1

        # Save the trajectory and reward
        list_memory_history.append(list_memory)
        list_total_reward.append(total_reward)
        # print(list)
        print("*********************************************")
        print("*********************************************")  
        print("[ INFO] Time for the epoch ", i, ": ", time.time()-time_start_epoch)
        print("*********************************************")
        print("*********************************************")

        if i%20 == 0:
            # Save the model
            print("Saving...")
            # model.save('/home/roboticlab14/catkin_ws/src/envirenement_reinforcement_learning/environment_package/src/saves/model/try_1.h5')
            model.save('/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/learn_to_go_position/model/try_1.h5')
            # Save datas
            print("Saving list_total_reward: ", save(list_total_reward, i, 
                arg_path = "/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/learn_to_go_position/reward/", 
                arg_name="list_total_reward_"))
            print("Saving list_memory_history: ", save(list_memory_history, 
                i, arg_path = "/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/learn_to_go_position/trajectory/", 
                arg_name = "list_memory_history_"))
            print("Saving list_loss: ", save(list_loss, 
                i, arg_path = "/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/learn_to_go_position/losses/", 
                arg_name = "list_loss_"))


# load_weights only sets the weights of your network. You still need to define its architecture before calling load_weights:
def create_model():
    '''
    Create a model with defined parameters
       # Model params
    
    NN: 10 inputs, 4 outputs and 2 hidden layers
         ...
    s1 - ... - q_1
    s2 - ... - q_2
    s3 - ... - q_3
         ... - q_4
         ...

    '''

    # Leanring params
    LEARNING_RATE = 0.001

    # Env params
    observation_space = 10
    action_space = 4

    model = Sequential()
    model.add(Dense(64, input_shape=(observation_space,), activation="relu"))
    model.add(Dense(64, activation="relu"))
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

def use_model(env, model):
    '''
    Use a trained model 
    '''
    epidodes_max = 5
    demo_step = 10
    observation_space = 10
    done = False
    for i in range(0, epidodes_max):
        j = 0
        state = env.reset()
        np_state = (np.array(state, dtype=np.float32),)
        np_state = np.reshape(np_state, [1, observation_space])
        # for j in range(0, demo_step):
        while j < demo_step and not done:
            q_values = model.predict(np_state)
            disc_action = discrete_action(np.argmax(q_values[0]))
            new_state, reward, done, info = env.step(disc_action)
            print("*********************************************")
            print("Observation: ", new_state)
            print("Reward: ", reward)
            print("Done: ", done)
            print("Episode: ", i)
            print("Step: ", j)
            print("*********************************************")
            np_new_state = (np.array(new_state, dtype=np.float32),)
            np_new_state = np.reshape(np_new_state, [1, observation_space])
            np_state = np_new_state
            state = new_state
            j+=1
        i+=1

def catch_object(env):
    env.reset()
    env.set_endEffector_pose([0.5, 0.0, 0.1, 3.1457, 0.0, 0.0])
    rospy.sleep(15)
    env.set_endEffector_pose([0.5, 0.0, 0.3, 3.1457, 0.0, 0.0])
    env.set_endEffector_pose([0.0, 0.5, 0.30, 3.1457, 0.0, 0.0])

def slide_object(env):
    env.reset()
    env.set_endEffector_pose([0.4, 0.3, 0.1, 3.1457, 0.0, 0.0])
    # rospy.sleep(15)
    env.set_endEffector_pose([0.4, 0.1, 0.1, 3.1457, 0.0, 0.0])
    env.set_endEffector_pose([0.4, -0.1, 0.1, 3.1457, 0.0, 0.0])
    env.set_endEffector_pose([0.4, -0.3, 0.1, 3.1457, 0.0, 0.0])

def main():
    rospy.init_node('training_node', anonymous=True, log_level=rospy.WARN)
    print("Start")
    # env = init_env()
    # env.reset()

    training = True
    if training:
        # Training using demos
        model = create_model()
        env = init_env()
        # memory = create_demo(env)
        # GAMMA = 0.95
        # model = training_from_demo(model, memory, GAMMA)
        dqn_learning_keras_memoryReplay(env, model)
    else:
        # Predict model
        model = load_trained_model("/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/20190716_095200_learn_to_go_position/model/try_1.h5")
        env = init_env()
        use_model(env, model)

    print("End")

if __name__ == '__main__':
    main()

