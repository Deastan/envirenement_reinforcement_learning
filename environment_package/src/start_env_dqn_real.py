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
        model.fit(state, q_values, verbose=0, callbacks=[tbCallBack])

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

            disc_action = discrete_action(action[j],step_size = 0.05)
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
            
            disc_action = discrete_action(action[j], step_size = 0.05)
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
def dqn_learning_keras_memoryReplay(env, model, folder_path, EPISODE_MAX, MAX_STEPS, GAMMA,MEMORY_SIZE, BATCH_SIZE, EXPLORATION_MAX, EXPLORATION_MIN, EXPLORATION_DECAY, observation_space, action_space, step_size):
    '''
    function which is a neural net using Keras with memory replay
    '''

    # Initializations:
    exploration_rate = EXPLORATION_MAX
    # Experience replay:
    memory = deque(maxlen=MEMORY_SIZE)

    # Create list for plot and saving
    list_memory_history = []
    list_total_reward = []
    list_loss = []
    episode_max = EPISODE_MAX
    list_done = []

    done = False

    for i in range(episode_max):
        time_start_epoch = time.time()
        total_reward = 0
        j = 0
        state = env.reset()
        rospy.sleep(5.0)
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
            disc_action = discrete_action(action, step_size)
            # print(action)
            time_start_action = time.time()
            print("################INSIDE ENV################")
            new_state, reward, done, _ = env.step(disc_action)
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

            # Memory replay
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
        list_done.append(done)
        # Save the trajectory and reward
        list_memory_history.append(list_memory)
        list_total_reward.append(total_reward)
        # print(list)
        print("*********************************************")
        print("*********************************************")  
        print("[ INFO] Time for the epoch ", i, ": ", time.time()-time_start_epoch)
        print("*********************************************")
        print("*********************************************")

        if i%5 == 0:
            # Save the model
            print("Saving...")
            
            # model.save('/home/roboticlab14/catkin_ws/src/envirenement_reinforcement_learning/environment_package/src/saves/model/try_1.h5')
            path = folder_path + 'model/'
            name = 'model_'
            total_model_path = path + name + str(i) + '.h5' 
            # model.save('/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/learn_to_go_position/model/try_1.h5')
            model.save(total_model_path)
            # Save datas
            print("Saving list_total_reward: ", utils.save(list_total_reward, i, 
                arg_path = folder_path + "reward/", 
                arg_name="list_total_reward_"))
            print("Saving list_memory_history: ", utils.save(list_memory_history, 
                i, arg_path = folder_path + "trajectory/", 
                arg_name = "list_memory_history_"))
            print("Saving list_done: ", utils.save(list_done, 
                i, arg_path = folder_path + "done/", 
                arg_name = "list_done_"))
            print("Saving list_loss: ", utils.save(list_loss, 
                i, arg_path = folder_path + "losses/", 
                arg_name = "list_loss_"))
            # Save the memory!
            print("Saving memory: ", utils.save(memory, 
                i, arg_path = folder_path + "memory/", 
                arg_name = "memory_"))
    

# Neural network with 2 hidden layer using Keras + experience replay
def dqn_learning_keras_memoryReplay_v2(env, model, folder_path, EPISODE_MAX, MAX_STEPS, GAMMA,MEMORY_SIZE, BATCH_SIZE, EXPLORATION_MAX, EXPLORATION_MIN, EXPLORATION_DECAY, observation_space, action_space, step_size):
    '''
    function which is a neural net using Keras with memory replay
    '''

    # Initializations:
    exploration_rate = EXPLORATION_MAX
    # Experience replay:
    memory = deque(maxlen=MEMORY_SIZE)

    # Create list for plot and saving
    list_memory_history = []
    list_total_reward = []
    list_loss = []
    episode_max = EPISODE_MAX
    list_done = []

    done = False

    for i in range(episode_max):
        # time_start_epoch = time.time()
        total_reward = 0
        j = 0
        state = env.reset()
        rospy.sleep(4.0)
        # Becarefull with the state (np and list)
        np_state = (np.array(state, dtype=np.float32),)
        np_state = np.reshape(np_state, [1, observation_space])
        # np_state = np.identity(7)[np_state:np_state+1]
        # print(np_state)
        done = False
        list_memory = []
        print("*********************************************")
        print("Episode: ", i)
        print("*********************************************")
        while j < MAX_STEPS and not done: # step inside an episode
            # time_start_step = time.time()
            
            # print("[INFO]: episode: ", i, ", step: ", j)

            # BECAREFUL with the action!
            action = choose_action(model, np_state, action_space, exploration_rate)
            disc_action = discrete_action(action, step_size)
            # print(action)
            # time_start_action = time.time()
            # print("################INSIDE ENV################")
            new_state, reward, done, _ = env.step(disc_action)
            # print("################INSIDE ENV################")
            # print("[ INFO] Time for the action ", j, ": ", time.time()-time_start_action)
            # time_start_print = time.time()
            # print("*********************************************")
            # print("Observation: ", new_state)
            # print("State: ", state)
            # print("Reward: ", reward)
            # print("Total rewards: ", total_reward)
            # print("Done: ", done)
            # print("Episode: ", i)
            # print("Step: ", j)
            # print("*********************************************")
            # print("[ INFO] Time for the print info ", j, ": ", time.time()-time_start_print)
            np_new_state = (np.array(new_state, dtype=np.float32),)
            np_new_state = np.reshape(np_new_state, [1, observation_space])
            # np_new_state = np.identity(7)[np_new_state:np_new_state+1]

            # Memory replay
            # memory = save_forReplay(memory, state, action, reward, new_state, done)
            # time_start_save_for_replay = time.time()
            save_forReplay(memory, np_state, action, reward, np_new_state, done)
            # print("[ INFO] Time for the saving of experience replay ", j, ": ", time.time()-time_start_save_for_replay)
            # time_start_experience_replay = time.time()
            
            # print("[ INFO] Time for the experience replay ", j, ": ", time.time()-time_start_experience_replay)
            
            list_memory.append([np_state, action, reward, np_new_state, done])
            
            # print("list_loss: ", list_loss)
            np_state = np_new_state
            state = new_state
            total_reward += reward
            
            
            # print("[ INFO] Time for the step ", j, ": ", time.time()-time_start_step)
            j+=1
        model, exploration_rate_new, loss = experience_replay_v2(model, memory, 
                BATCH_SIZE, exploration_rate, EXPLORATION_DECAY, EXPLORATION_MIN, GAMMA)
        exploration_rate = exploration_rate_new
        # Save the lost function into a list
        if len(memory) > BATCH_SIZE:
            list_loss.append(loss[0])
        list_done.append(done)
        # Save the trajectory and reward
        list_memory_history.append(list_memory)
        list_total_reward.append(total_reward)
        # print(list)
        # print("*********************************************")
        # print("*********************************************")  
        # print("[ INFO] Time for the epoch ", i, ": ", time.time()-time_start_epoch)
        # print("*********************************************")
        # print("*********************************************")

        if i%500 == 0:
            # Save the model
            print("Saving...")
            
            # model.save('/home/roboticlab14/catkin_ws/src/envirenement_reinforcement_learning/environment_package/src/saves/model/try_1.h5')
            path = folder_path + 'model/'
            name = 'model_'
            total_model_path = path + name + str(i) + '.h5' 
            # model.save('/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/learn_to_go_position/model/try_1.h5')
            model.save(total_model_path)
            # Save datas
            print("Saving list_total_reward: ", utils.save(list_total_reward, i, 
                arg_path = folder_path + "reward/", 
                arg_name="list_total_reward_"))
            print("Saving list_memory_history: ", utils.save(list_memory_history, 
                i, arg_path = folder_path + "trajectory/", 
                arg_name = "list_memory_history_"))
            print("Saving list_done: ", utils.save(list_done, 
                i, arg_path = folder_path + "done/", 
                arg_name = "list_done_"))
            print("Saving list_loss: ", utils.save(list_loss, 
                i, arg_path = folder_path + "losses/", 
                arg_name = "list_loss_"))
            # Save the memory!
            print("Saving memory: ", utils.save(memory, 
                i, arg_path = folder_path + "memory/", 
                arg_name = "memory_"))


def use_model(env, model, MAX_STEPS, observation_space, step_size, GAMMA):
    '''
    Use a trained model 
    '''
    # Parameters;
    env.set_mode(mode="evaluating")
    # list_target_points = []
    # list_target_points.append([0.5, -0.3, 0.3])
    # list_target_points.append([0.6, -0.0, 0.3])
    # list_target_points.append([0.4, 0.3, 0.3])
    # # list_target_points.append([0.4, -0.4, 0.1])
    # # list_target_points.append([0.6, 0.1, 0.1])
    # # list_target_points.append([0.6, -0.1, 0.1])
    # # list_target_points.append([0.5, -0.2, 0.1])
    # np_target_points = np.array(list_target_points)
    # epidodes_max = len(list_target_points)
    epidodes_max = 100
    done = False
    list_loss = [] # to save for compute the mean of this model
    list_total_reward = []
    for i in range(epidodes_max):
        j = 0
        # env.set_for_evaluation(np_target_points[i])
        rospy.sleep(0.05)
        state = env.reset()
        rospy.sleep(5.0)
        np_state = (np.array(state, dtype=np.float32),)
        np_state = np.reshape(np_state, [1, observation_space])
        total_reward = 0
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
            total_reward += reward
            np_state = np_new_state
            state = new_state
            j+=1
        list_total_reward.append(total_reward)
        i+=1
    return sum(list_loss)/len(list_loss)

def use_model_v2(env, model, MAX_STEPS, observation_space, step_size, GAMMA):
    '''
    Use a trained model 
    '''
    # Parameters;
    env.set_mode(mode="predicting")
    
    
    epidodes_max = 1
    done = False
    list_loss = [] # to save for compute the mean of this model
    for i in range(epidodes_max):
        j = 0
        # env.set_for_evaluation(np_target_points[i])
        # rospy.sleep(0.05)
        state = env.reset()
        rospy.sleep(2.0)
        np_state = (np.array(state, dtype=np.float32),)
        np_state = np.reshape(np_state, [1, observation_space])
        
        while j < MAX_STEPS and not done:
            # state = env.new_observation
            # np_state = (np.array(state, dtype=np.float32),)
            # np_state = np.reshape(np_state, [1, observation_space])
            q_values = model.predict(np_state)
            print("state for chossing: ", np_state)
            action = np.argmax(q_values[0])
            disc_action = discrete_action(action, step_size)
            # rospy.sleep(6.0)
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
    #return sum(list_loss)/len(list_loss)


def compute_exploration_decay(EXPLORATION_MAX, EXPLORATION_MIN, EPISODE_MAX, MAX_STEPS):
    '''
    Compute the exploration decay in function of Episode max

    Output: EXPLORATION_DECAY

    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.999790 #0.9993 # Over 500 =>0.9908, for 2500 =>0.998107
    # Use exploration_rate = exploration_rate*EXPLORATION_DECAY
    # exploration decay = 10^(log(0.01)/EPISODE_MAX)
    '''
    return 10**(math.log10(EXPLORATION_MIN)/(EPISODE_MAX*MAX_STEPS/4*3))

def main():
    rospy.init_node('training_node', anonymous=True, log_level=rospy.WARN)
    print("Start")
    
    # Parameters:
    EPISODE_MAX = 4000
    MAX_STEPS = 50
    #PARAMS
    GAMMA = 0.95
    MEMORY_SIZE = EPISODE_MAX*10
    BATCH_SIZE = 256
    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = compute_exploration_decay(EXPLORATION_MAX, EXPLORATION_MIN, EPISODE_MAX, MAX_STEPS)

    # Env params
    observation_space = 10
    action_space = 4
    step_size = 0.025
    task = "position_learning"

    # Neural Net:
    hidden_layers = 2
    neurons = 64
    LEARNING_RATE = 0.001

    training = False
    bool_evaluate = False
    if training:
        print("training")
        _, folder_path = utils.init_folders(task=task)
        utils.create_summary(folder_path, task, EPISODE_MAX, MAX_STEPS, GAMMA,MEMORY_SIZE, BATCH_SIZE, EXPLORATION_MAX, EXPLORATION_MIN, EXPLORATION_DECAY, observation_space, action_space, hidden_layers, neurons, LEARNING_RATE, step_size)

        # Training using demos
        model = utils.create_model(inputs=observation_space, outputs=action_space, hidden_layers=hidden_layers, neurons=neurons, LEARNING_RATE = LEARNING_RATE)
        env = init_env()
        # memory = create_demo(env)
        # GAMMA = 0.95
        # model = training_from_demo(model, memory, GAMMA)
        dqn_learning_keras_memoryReplay_v2(env, model, folder_path, EPISODE_MAX, MAX_STEPS, GAMMA,MEMORY_SIZE, BATCH_SIZE, EXPLORATION_MAX, EXPLORATION_MIN, EXPLORATION_DECAY, observation_space, action_space, step_size)
    elif bool_evaluate:
        print("evaluating")
        env = init_env()
        folder_path = "/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/position_learning/"
        folder_name = "model_for_kuka_position_learning/"
        folder_path = folder_path + folder_name
        
        model_path = folder_path +"model/"
        model_name = "model_"
        total_model_path = model_path + model_name + str(3500) + ".h5"
        model = utils.load_trained_model(total_model_path)
        use_model(env, model, MAX_STEPS, observation_space, step_size, GAMMA)
    else:
        print("Predicting")
        env = init_env()
        folder_path = "/media/roboticlab14/DocumentsToShare/Reinforcement_learning/Datas/position_learning/"
        folder_name = "model_for_kuka_position_learning/"
        folder_path = folder_path + folder_name
        
        model_path = folder_path +"model/"
        model_name = "model_"
        total_model_path = model_path + model_name + str(3500) + ".h5"
        model = utils.load_trained_model(total_model_path)
        use_model_v2(env, model, MAX_STEPS, observation_space, step_size, GAMMA)
    print("End")

if __name__ == '__main__':
    main()

# def catch_object(env):
#     env.reset()
#     env.set_endEffector_pose([0.5, 0.0, 0.1, 3.1457, 0.0, 0.0])
#     rospy.sleep(15)
#     env.set_endEffector_pose([0.5, 0.0, 0.3, 3.1457, 0.0, 0.0])
#     env.set_endEffector_pose([0.0, 0.5, 0.30, 3.1457, 0.0, 0.0])

# def slide_object(env):
#     env.reset()
#     env.set_endEffector_pose([0.4, 0.3, 0.1, 3.1457, 0.0, 0.0])
#     # rospy.sleep(15)
#     env.set_endEffector_pose([0.4, 0.1, 0.1, 3.1457, 0.0, 0.0])
#     env.set_endEffector_pose([0.4, -0.1, 0.1, 3.1457, 0.0, 0.0])
#     env.set_endEffector_pose([0.4, -0.3, 0.1, 3.1457, 0.0, 0.0])