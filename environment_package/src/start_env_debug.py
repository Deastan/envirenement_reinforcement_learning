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
def save(list_theta, epoch, arg_path= 
        "/home/roboticlab14/catkin_ws/src/envirenement_reinforcement_learning/environment_package/src/saves/pickles/", 
        arg_name = "list_of_reward_"):
    '''
    Save a pickle file of a list that you want to save.

    '''
    saved = False
    try:
        path = arg_path
        name = arg_name
        full_path = path + name + str(epoch) + ".pkl"
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
        return model
    batch = random.sample(memory, BATCH_SIZE)
    # list_q_
    for state, action, reward, new_state, done in batch:
        if not done:
            q_target = (reward + GAMMA * np.amax(model.predict(new_state)))
        else:
            q_target = reward

        q_values = model.predict(state)
        q_values[0][action] = q_target
        model.fit(state, q_values, verbose=0)#, callbacks=[tensorboard])

    exploration_rate *= EXPLORATION_DECAY
    exploration_rate = max(EXPLORATION_MIN, exploration_rate)
    return model

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
    demo_epoch = 1
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
            j+=1
        i+=1
    
    return memory_from_demo

# Neural network with 2 hidden layer using Keras + experience replay
def dqn_learning_keras_memoryReplay(env, model):
    '''
    function which is a neural net using Keras with memory replay
    '''

    # Time optimzation:
    

    EPISODE_MAX = 4000
    MAX_STEPS = 15
    #PARAMS
    GAMMA = 0.95
    LEARNING_RATE = 0.001
    MEMORY_SIZE = EPISODE_MAX
    BATCH_SIZE = 2
    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.995

    # Env params
    observation_space = 7
    # observation_space = env.observation_space
    action_space = 4

    exploration_rate = 0.4#EXPLORATION_MAX

    # Create the model !!!
    # model = create_model()

    # Experience replay:
    memory = deque(maxlen=MEMORY_SIZE)

    # Create list for plot and saving
    list_memory_history = []
    list_total_reward = []
    
    # Plot
    # fig = plt.figure()
    # fig.suptitle("Training in progress", fontsize=22)

    # # Figure 1
    # ax = fig.add_subplot(211)
    # [line] = ax.plot(list_total_reward, label='Reward')
    # ax.title.set_text('Reward')
    # ax.set_xlabel('Epochs [-]')
    # ax.set_ylabel('Reward [-]')
    # # plt.xlim(0, EPISODE_MAX)
    # # plt.ylim(-150, 100)
    # ax.legend()
    # plt.show(False)
    
    
    episode_max = EPISODE_MAX

    done = False

    for i in range(episode_max):
        time_start_epoch = time.time()
        total_reward = 0
        j = 0
        state = env.reset()
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
            model = experience_replay(model, memory, 
                BATCH_SIZE, exploration_rate, EXPLORATION_DECAY, EXPLORATION_MIN, GAMMA)
            print("[ INFO] Time for the experience replay ", j, ": ", time.time()-time_start_experience_replay)
            
            list_memory.append([np_state, action, reward, np_new_state, done])

            np_state = np_new_state
            total_reward += reward
            
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

        # # Live plot:
        # # Plot 2
        # ax.clear()
        # string = str('Reward at ' + str(i) + ' epoch')
        # [line] = ax.plot(list_total_reward, label=string)
        
        # ax.title.set_text('Reward')
        # ax.set_xlabel('Epochs [-]')
        # ax.set_ylabel('Reward [-]')
        # # plt.xlim(0, j)
        # # plt.ylim(-150, 100)
        # ax.legend()
        # # plt.pause(0.05)
        # fig.canvas.draw_idle()
        # # fig.canvas.draw()
        # # plt.close(fig)
        # # plt.show(False)

        if i%50 == 0:
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

    
    # plt.show()
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
       # Model params
    
    NN: 7 inputs, 4 outputs
         ...
    s1 - ... - q_1
    s2 - ... - q_2
    s3 - ... - q_3
         ... - q_4
         ...


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

def catch_object(env):
    env.reset()

    rospy.sleep(15)
    env.set_endEffector_pose([0.5, 0.0, 0.1, 3.1457, 0.0, 0.0])
    
    env.set_endEffector_pose([0.5, 0.0, 0.3, 3.1457, 0.0, 0.0])
    env.set_endEffector_pose([0.0, 0.5, 0.30, 3.1457, 0.0, 0.0])

def slide_object(env):
    env.reset()
    # rospy.sleep(20)
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

    # Create the model
    model = create_model()
    env = init_env()
    # env.reset()
    slide_object(env)
    slide_object(env)
    slide_object(env)
    slide_object(env)

    # print("end before sleep!")
    # rospy.sleep(150)
    # memory = create_demo(env)
    # GAMMA = 0.95
    # model = training_from_demo(model, memory, GAMMA)
    # qlearning(env)
    # catch_object(env)
    # slide_object(env)
    # dqn_learning_keras_memoryReplay(env, model)
    # create_demo(env)
    print("end")

if __name__ == '__main__':
    main()




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



