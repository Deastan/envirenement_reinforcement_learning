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

def main():
    rospy.init_node('training_node', anonymous=True, log_level=rospy.WARN)
    print("Start")
    # Parameters:
    MAX_STEPS = 50
    MAX_EPOCHS = 3
    env = init_env()
    print(env.reset())
    # while True:
    for epoch in range(0, MAX_EPOCHS):
        
        last_obs = env.reset()
        for step in range(0, MAX_STEPS):
            discrete_act = env.action_space.sample()
            action = discrete_action(discrete_act)
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


    print("end")

if __name__ == '__main__':
    main()
   