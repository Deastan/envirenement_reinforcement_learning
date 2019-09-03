#!/usr/bin/env python2

from math import pi
import numpy as np
import os

# Reinforcement learning
import gym
from gym.utils import seeding
from gym import spaces

# For the ROS 
import roslaunch
import rospy
import rospkg
import git
import sys
import time
import random
import tf.transformations as tft

# Moveit
import moveit_commander
from tf.transformations import *

# Msg
import geometry_msgs.msg
from std_msgs.msg import String
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import moveit_msgs.msg
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import TrajectoryConstraints, Constraints, JointConstraint, RobotState, WorkspaceParameters, MotionPlanRequest
from controller_manager_msgs.srv import SwitchController
from controller_manager_msgs.srv import LoadController
from std_srvs.srv import Empty
from std_msgs.msg import Empty as EmptyMsg
from geometry_msgs.msg import *
# Spawner
import roslib
# from gazebo.srv import *

# Ros Services
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import SpawnModelRequest
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import DeleteModelRequest

# https://github.com/openai/gym/blob/master/gym/core.py
class RobotGazeboEnv(gym.Env):

    def __init__(self):
        self.status = "environment created"
        
        #Gazebo:
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_world_or_sim = "NO_RESET" # SIMULATION WORLD NO_RESET
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        # Init position:
        self.bool_init_joint_or_pose = False
        self.constant_z = 0.3

        # Creation of moveit variables
        # self.moveit_object = MoveIiwa()
        self.pub_cartesianPose = rospy.Publisher('/iiwa/moveToCartesianPose', Pose, queue_size=1)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        
        self.group_name = "manipulator"
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        self.group.set_planner_id("RRTkConfigDefault")
        
        # For GYM
        # Action space
        # +x, -x, +y, -y, +z. z 
        # self.n_actions = 6 # real one
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)

        # Observations space
        low = np.array([-0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        # quat of (6.28, 6.28, 6.28) => (0.99534, 0.05761, 0.05761, 0.05162)
        high = np.array([0.8, 0.8, 1.266, 1.0, 1.0, 1.0, 1.0])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Observations:
        # Last position
        self.last_pose = [] #this is a pose so x, y, z R, P, Y
        self.last_observation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #this is an obersvations: x, y, z R, P, Y, (maybe gripper: open/close), etc...
        # Current position
        self.current_pose = []
        self.current_observation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # TARGET
        # TODO: Separate target_x... in a yaml file
        target_x = 0.5#0.0#0.5
        target_y = -0.5#0.5
        target_z = self.constant_z#0.5
        self.target_position = []
        self.target_position.append(target_x)
        self.target_position.append(target_y)
        self.target_position.append(target_z)

        # eePositon:
        # TODO: Separate ee_x... in a yaml file
        ee_x = 0.5#0.0#0.5
        ee_y = +0.2#0.5
        ee_z = self.constant_z#0.5
        self.ee_position = []
        self.ee_position.append(ee_x)
        self.ee_position.append(ee_y)
        self.ee_position.append(ee_z)

        # MODE
        # Default is training
        # Training, Predicting, Evaluating
        self.env_mode = "training"
        self.continuous = False # waiting on a result

        # Reward
        self.out_workspace = False

        # REAL ROBOT
        self.bool_real_robot = False
        self.pub_real_robot = rospy.Publisher('/iiwa/command/CartesianPose', PoseStamped, queue_size=1)
        self.listener = rospy.Subscriber("/iiwa/state/CartesianPose", PoseStamped, self.callback_real_robot, queue_size=5)
        
        # self.rospy.s
        # rospy.spin()
        # TODO: create an env without GAZEBO FO THE REAL ROBOT
        # Real robot position init
        self.current_robot_pose = []
        # Position
        self.current_robot_pose.append(self.ee_position[0])
        self.current_robot_pose.append(self.ee_position[1])
        self.current_robot_pose.append(self.ee_position[2])
        # Orientation
        self.current_robot_pose.append(0.0)
        self.current_robot_pose.append(0.0)
        self.current_robot_pose.append(0.0)
        self.current_robot_pose.append(1.0)

        self.current_observation = self.current_robot_pose
        self.current_observation.append(self.target_position[0])
        self.current_observation.append(self.target_position[1])
        self.current_observation.append(self.target_position[2])
        self.new_observation = self.current_observation

    # Env methods
    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        # return
        raise NotImplementedError

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.set_endEffector_actionToPose(action)
        observation = self._get_obs()
        done = self._is_done(observation)
        reward = self._compute_reward(observation, done)
        info = "No information yet"
        
        return observation, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns: 
            observation (object): the initial observation.
        """
        if self.env_mode == "training":
            # self._init_env_variables()
            result = self.random_target_position()
            self.random_ee_position()
        elif self.env_mode == "predicting":
            result = self.random_target_position()
            self.random_ee_position()
        elif self.env_mode == "evaluating":
            
            result = self.random_target_position()
            self.random_ee_position()

        self.load_stop_controller()
        self.start_controller()
        self.pause()
        self._reset_sim()
        self.unpause()
        # self.load_start_controller()
        # self.start_controller()
        # self.launch.start() # add the node to close the hand
        
        
        self._init_pose()

        #DIRTY!
        init_action = [self.ee_position[0], self.ee_position[1], self.ee_position[2], 3.1457, 0.0, 0.0]
        self.set_endEffector_pose(init_action)
        rospy.sleep(3.0)
        return self._get_obs()

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    # We will do nothing yet inside this one
    def render(self, mode='human'):

        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                        # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError
    #*****************************************************************************
    #   Method extern as gym but intern at the reinforcement learning
    #*****************************************************************************


    def _init_pose(self):
        '''
        Initialize the position of the robot in the environment between two episodes!

        '''
        if(self.bool_init_joint_or_pose == True):
            init_joint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # init_joint = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]
            self.set_joints_execute(init_joint)
        else:
            # init_action = [0.6, 0.1, self.constant_z, 3.1457, 0.0, 0.0]
            init_action = [self.ee_position[0], self.ee_position[1], self.ee_position[2], 3.1457, 0.0, 0.0]
            self.set_endEffector_pose(init_action)
        self._init_env_variables()
        return True

    def _init_env_variables(self):
        '''
        Initialize all needed variables of the environment between two episodes
        '''
        # print("Var = 0")
        # self.step_circle = 0
        if not self.bool_real_robot:
            self.last_pose = self.get_endEffector_pose().pose
            self.current_pose = self.last_pose
        elif self.bool_real_robot:
            self.last_pose = Pose()
            self.last_pose.position.x = self.current_robot_pose[0]
            self.last_pose.position.y = self.current_robot_pose[1]
            self.last_pose.position.z = self.current_robot_pose[2]
            self.last_pose.orientation.x = self.current_robot_pose[3]
            self.last_pose.orientation.y = self.current_robot_pose[4]
            self.last_pose.orientation.z = self.current_robot_pose[5]
            self.last_pose.orientation.w = self.current_robot_pose[6]

            self.current_pose = self.last_pose
        
        return True

    def _reset_sim(self):
        """
        Resets a simulation
        """

        if self.reset_world_or_sim == "SIMULATION":
            # reset also the physic environment!
            rospy.wait_for_service('/gazebo/reset_simulation')
            self.reset_simulation_proxy()
        elif self.reset_world_or_sim == "WORLD":
            rospy.wait_for_service('/gazebo/reset_world')
            self.reset_world_proxy()
        elif self.reset_world_or_sim == "NO_RESET":
            print("Reset Gazebo without doing anything")
        else:
            print("ERROR: CANNOT RESET GAZEBO!")
        
        # # Add and delete OBJECT 
        # self.remove_object()
        # # rospy.sleep(0.1)
        # self.spawn_object()
        # Add and delete OBJECT 
        
        # rospy.sleep(0.1)
        # self.spawn_object()

        # Robot REMOVE AND SPAWN ********************************
        # self.remove_robot()
        # # rospy.sleep(0.05)
        # self.spawn_robot()


    def _set_action(self, action):
        """
        It sets the joints of monoped based on the action integer given
        based on the action number given.
        :param action: The action integer that sets what movement to do next.
        """
        print("NO ACTION SET IN : _set_action")

    
    # TODO: remove get pose from moveit but take the one from tf
    def _get_obs(self):
        """
        Observations:
        Pose of the hand effector
        Position of the target 
        
        """
        
        if not self.bool_real_robot:
            # rospy.logdebug("Start Get Observation ==>")
            self.last_pose = self.current_pose
            # print(self.current_pose.pose)

            last_observation = []     
            last_observation.append(self.current_pose.position.x)
            last_observation.append(self.current_pose.position.y)
            last_observation.append(self.current_pose.position.z)
            last_observation.append(self.current_pose.orientation.x)
            last_observation.append(self.current_pose.orientation.y)
            last_observation.append(self.current_pose.orientation.z)
            last_observation.append(self.current_pose.orientation.w)
            # Add the target position to the observation:
            last_observation.append(self.target_position[0])
            last_observation.append(self.target_position[1])
            last_observation.append(self.target_position[2])

            self.last_observation = last_observation
            
            self.current_pose = self.get_endEffector_pose().pose
            observation = []
            
            observation.append(self.current_pose.position.x)
            observation.append(self.current_pose.position.y)
            observation.append(self.current_pose.position.z)
            observation.append(self.current_pose.orientation.x)
            observation.append(self.current_pose.orientation.y)
            observation.append(self.current_pose.orientation.z)
            observation.append(self.current_pose.orientation.w)

            # Add the target position to the observation:
            observation.append(self.target_position[0])
            observation.append(self.target_position[1])
            observation.append(self.target_position[2])

            # Update observation 
            self.current_observation = observation
        elif self.bool_real_robot:
            # print("Current position")
            self.last_pose = self.current_pose
            
            # self.last_observation = last_observation
            self.last_observation = self.current_observation
            
            # self.current_pose = self.get_endEffector_pose().pose
            rospy.sleep(1.0)
            # self.listener
            current_robot_pose_interm = self.current_robot_pose
            observation = []
            
            observation.append(self.current_robot_pose[0])
            observation.append(self.current_robot_pose[1])
            observation.append(current_robot_pose_interm[2])
            q_interm = quaternion_from_euler(3.1457, 0.0, 0.0)

            # DIRTY!
            # robot : quaternion
            # 0.9999978912139245, 0.0, -0.0, -0.00205367176151744,
            # simulation: quaternion
            # -0.9999978144262414, -0.00021456720756557568, 0.000276704450621678, 0.0020611982681980317
            observation.append(-0.9999978144262414)#(q_interm[0]) # ACHTUUUUUNGGGGG
            observation.append(q_interm[1])
            observation.append(q_interm[2])
            observation.append(q_interm[3])
            # observation.append(current_robot_pose_interm[3])
            # observation.append(current_robot_pose_interm[4])
            # observation.append(current_robot_pose_interm[5])
            # observation.append(current_robot_pose_interm[6])
            
            # Add the target position to the observation:
            observation.append(self.target_position[0])
            observation.append(self.target_position[1])
            observation.append(self.target_position[2])

            # Update observation 
            self.current_observation = observation
        
        return observation

    def _compute_reward(self, observations, done):
        """
        Compute and return the reward for a step (single action)
        The reward is defined here by the user!
        
        """
        
        # The sign depend on its function.
        total_reward = 0
        
        # create and update from last position
        last_position = []
        last_position.append(self.last_pose.position.x)
        last_position.append(self.last_pose.position.y)
        last_position.append(self.last_pose.position.z)

        # create and update current position
        current_position = []
        current_position.append(self.current_pose.position.x)
        current_position.append(self.current_pose.position.y)
        current_position.append(self.current_pose.position.z)

        # create the distance btw the two last vector
        distance_before_move = self.distance_between_vectors(last_position, self.target_position)
        distance_after_move = self.distance_between_vectors(current_position, self.target_position)
        distance_btw_start_goal = self.distance_between_vectors(self.ee_position, self.target_position)

        alpha = 1
        # Give the reward
        if self.out_workspace:
            total_reward -= 20
        else:
            if done:
                total_reward += 20
            else:
                # if(distance_after_move - distance_before_move < 0): # before change... >
                #     # print("right direction")
                #     if distance_btw_start_goal != 0:
                #         total_reward += 1.0/distance_btw_start_goal
                #     else:
                #         total_reward += 1.0

                # else:
                    # total_reward -= 2.0 # 1.0
                    # check division by 0: if distance_btw_goal_init != 0
                if distance_btw_start_goal != 0:
                    total_reward -= alpha*distance_after_move/distance_btw_start_goal # 1.0
                else:
                    total_reward -= alpha*distance_after_move

        # Time punishment
        total_reward -= 2.0

        return total_reward
    
    # Check if the goal is reached or not
    def _is_done(self, observations):
        """
        
        """

        done = False
        vector_observ_pose = []
        vector_observ_pose.append(observations[0])
        vector_observ_pose.append(observations[1])
        vector_observ_pose.append(observations[2])

        # Check if the hand effector is close to the target in m!
        if self.distance_between_vectors(vector_observ_pose, self.target_position) < 0.11:
            done = True

        return done
    
    #*****************************************************************************
    #   Method extern as gym
    #*****************************************************************************
    
    def check_workspace(self, pose):
        '''
        Check the workspace if we can reach the position or not...
        Assumptions: Never go under z = 0.2 (offset_z)

        Output: 
                True if we can reach the position, False otherwise.
        '''
        
        
        result = False
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
        
        # Sphere parameters:
        z0 = 0.34
        offset_ee = 0.13
        # offset_z = 0.25
        offset_z = 0.05 # for testing the hands
        
        if (((x*x + y*y + (z-z0)*(z-z0)) < (0.8+offset_ee)*(0.8+offset_ee)) 
                and ((x*x + y*y + (z-z0)*(z-z0)) > 0.4*0.4) 
                and z > offset_z):
            self.out_workspace = False
            result = True
        else:
            print("Check_workspace: Cannot go at this pose")
            self.out_workspace = True
        
        return result

    def get_endEffector_pose(self):
        
        endEffector_pose = self.group.get_current_pose()
        return endEffector_pose

    def set_endEffector_actionToPose(self, action):
        '''
        Vector action is the delta position (m) and angle (rad)
        (x, y, z, R, P, Y)    
        '''
        if not self.bool_real_robot:

            # Current pose of the hand effector
            current_pose = geometry_msgs.msg.Pose()
            current_pose = self.get_endEffector_pose().pose
            # print("***************************************************")
            # print("current orientation")
            # print(current_pose.orientation)

            # Normal way when the orientaiton of the end effector is not fixed!
            # q_interm = quaternion_from_euler(action[3], action[4], action[5])
            # q_interm[0] = current_pose.orientation.x
            # q_interm[1] = current_pose.orientation.y
            # q_interm[2] = current_pose.orientation.z
            # q_interm[3] = current_pose.orientation.w

            # Fix end effector 
            q_interm = quaternion_from_euler(3.1457, 0.0, 0.0)

            # euler to quaternion  to apply the rotation to the end effector
            # R, P, Y = action[3], action[4], action[5]
            # q_rot = quaternion_from_euler(action[3], action[4], action[5])
            # print("desired rotation")
            # print(q_rot)
            # q_interm = quaternion_multiply(q_rot, q_interm)*************************")
    

            # create pose msg
            pose_goal = geometry_msgs.msg.Pose()
            
            pose_goal.position.x = current_pose.position.x + action[0]
            pose_goal.position.y = current_pose.position.y + action[1]
            pose_goal.position.z = self.constant_z #current_pose.position.z + action[2]

            # q_interm = quaternion_multiply(q_rot, q_interm)
            pose_goal.orientation.x = q_interm[0]
            pose_goal.orientation.y = q_interm[1]
            pose_goal.orientation.z = q_interm[2]
            pose_goal.orientation.w = q_interm[3]
            # pose_goal.orientation =  q_mult(q_rot, current_pose.orientation)

        elif self.bool_real_robot:
            goal = PoseStamped()

            goal.header.seq = 1
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = "world"
            
            goal.pose.position.x = self.current_robot_pose[0] + action[0]
            goal.pose.position.y = self.current_robot_pose[1] + action[1]
            goal.pose.position.z = self.constant_z

            quaternion = tf.transformations.quaternion_from_euler(3.14, 0.0, -3.14)#1.5707963
            # quaternion = quaternion_from_euler(3.1457, 0.0, 0.0)
            goal.pose.orientation.x = quaternion[0]
            goal.pose.orientation.y = quaternion[1]
            goal.pose.orientation.z = quaternion[2]
            goal.pose.orientation.w = quaternion[3]

            pose_goal = goal.pose
            # rospy.sleep(1.0)
            self.pub_real_robot.publish(goal)
            return True

        if self.check_workspace(pose_goal) == True:
            result = self.execute_endEffector_pose(pose_goal)
            result = True
        else:
            result = False
        return result
        
    def set_endEffector_pose(self, action):
        '''
        Vector action is a pose!
        (x, y, z, R, P, Y)    
        '''
        q_interm = quaternion_from_euler(action[3], action[4], action[5])

        # create pose msg
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = action[0]
        pose_goal.position.y = action[1]
        pose_goal.position.z = action[2]#current_pose.position.z + action[2]
        # q_interm = quaternion_multiply(q_rot, q_interm)
        pose_goal.orientation.x = q_interm[0]
        pose_goal.orientation.y = q_interm[1]
        pose_goal.orientation.z = q_interm[2]
        pose_goal.orientation.w = q_interm[3]

        if self.bool_real_robot:
            goal = PoseStamped()

            goal.header.seq = 1
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = "world"
            goal.pose.position.x = action[0]
            goal.pose.position.y = action[1]
            goal.pose.position.z = pose_goal.position.z

            quaternion = tf.transformations.quaternion_from_euler(3.14, 0.0, -3.14)#1.5707963
            goal.pose.orientation.x = quaternion[0]
            goal.pose.orientation.y = quaternion[1]
            goal.pose.orientation.z = quaternion[2]
            goal.pose.orientation.w = quaternion[3]

            pose_goal = goal.pose
            self.pub_real_robot.publish(goal)
            return True

        # Check if point is in the workspace:
        # bool_check_workspace = self.check_workspace(pose_goal)
        if self.check_workspace(pose_goal) == True:
            # self.pub_cartesianPose.publish(pose_goal)
            if not self.bool_real_robot:
                result = self.execute_endEffector_pose(pose_goal)
            elif self.bool_real_robot:
                # result = self.execute_endEffector_pose(pose_goal)
                self.pub_real_robot.publish(goal)
            # Shortcut the result from moveit 
            result = True
        else:
            result = False
            
        return result


    # Plan and Execute the trajectory planed for a given cartesian pose
    def execute_endEffector_pose(self, pose):
        '''
        Send to the "controller" the positions (trajectory) where it want to go
        '''
        
        result = False
        
        self.group.set_pose_target(pose)
        
        result = self.group.go(wait=True)
        

        self.group.stop()
        
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.group.clear_pose_targets()
        return result

    def set_joints_execute(self, joints_angle):
        '''
        Execute the trajectory to go to the desired joints angle
        
        '''
        
        # self.group.set_joint_value_target(joint_goal)
        self.group.set_joint_value_target(joints_angle)
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        
        self.plan = self.group.plan()
        result = self.group.go(wait=True)
        
        # Calling ``stop()`` ensures that there is no residual movement
        self.group.stop()
        return result

    # Calculates the distance btw two position vectors
    def distance_between_vectors(self, v1, v2):
        """
        distance between two vectors
        input list
        transfrom the list to np array and they compute the norm 2
        return the distance
        """
        # time_start_distance_between_vectors = time.time()
        # self.group.set_joint_value_target(joint_goal)
        dist = np.linalg.norm(np.array(v1) - np.array(v2))
        return dist

    # TODO: ADD a random position of the object
    #       add the location of the object : https://answers.ros.org/question/237862/rosnode-kill/
    #           and to run as a subprocesses: https://answers.ros.org/question/41848/start-a-node-from-python-code-rospy-equivalence-rosrun-rosgui-qt/
    def spawn_object(self):
        '''
        Spawn an object at a defined position
        https://github.com/ipa320/srs_public/blob/master/srs_user_tests/ros/scripts/spawn_object.py
        '''
        # print("Spawn object")
        


        # Create position of the object
        name = "object_to_push"
        # convert rpy to quaternion for Pose message
		# orientation = rospy.get_param("/objects/%s/orientation" % name)
        orientation = [0.0, 0.0, 0.0]
        quaternion = tft.quaternion_from_euler(orientation[0], orientation[1], orientation[2])
        object_pose = Pose()
        object_pose.position.x = float(self.target_position[0])
        object_pose.position.y = float(self.target_position[1])
        object_pose.position.z = float(0.5)
        object_pose.orientation.x = quaternion[0]
        object_pose.orientation.y = quaternion[1]
        object_pose.orientation.z = quaternion[2]
        object_pose.orientation.w = quaternion[3]
        # Create object
        file_localition = roslib.packages.get_pkg_dir('reflex_description') + '/urdf/object_to_catch.urdf'
        rospy.wait_for_service("/gazebo/spawn_urdf_model")
        srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        file_xml = open(file_localition)
        xml_string=file_xml.read()
        # spawn new model
        req = SpawnModelRequest()
        req.model_name = name # model name from command line input
        req.model_xml = xml_string
        req.initial_pose = object_pose
        
        res = srv_spawn_model(req)

    def remove_object(self):
        '''
        Remove an object
        https://github.com/ipa320/srs_public/blob/master/srs_user_tests/ros/scripts/spawn_object.py
        '''
        name = "object_to_push"
        rospy.wait_for_service("/gazebo/delete_model")
        srv_delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        req = DeleteModelRequest()
        req.model_name = name
        # res = srv_delete_model(name)
        try:
            res = srv_delete_model(name)
        except rospy.ServiceException, e:
            print("ERROR: CANNOT REMOVE Robot: ", name)

    def spawn_robot(self):
        '''
        Spawn an object at a defined position
        https://github.com/ipa320/srs_public/blob/master/srs_user_tests/ros/scripts/spawn_object.py
        '''
        
        # Create position of the object
        name = "iiwa"
        # convert rpy to quaternion for Pose message
		# orientation = rospy.get_param("/objects/%s/orientation" % name)
        orientation = [0.0, 0.0, 0.0]
        quaternion = tft.quaternion_from_euler(orientation[0], orientation[1], orientation[2])
        object_pose = Pose()
        object_pose.position.x = float(0.0)
        object_pose.position.y = float(0.0)
        object_pose.position.z = float(0.0)
        object_pose.orientation.x = quaternion[0]
        object_pose.orientation.y = quaternion[1]
        object_pose.orientation.z = quaternion[2]
        object_pose.orientation.w = quaternion[3]
        # Create object
        file_localition = roslib.packages.get_pkg_dir('reflex_description') + '/urdf/reflex_takktile_2.xacro'
        p = os.popen("rosrun xacro xacro.py " + file_localition)
        xml_string = p.read()
        p.close()
        srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)

        # rospy.wait_for_service("/gazebo/spawn_urdf_model")
        # srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        # file_xml = open(file_localition)
        # xml_string=file_xml.read()
        # spawn new model
        req = SpawnModelRequest()
        req.model_name = name # model name from command line input
        req.model_xml = xml_string
        req.initial_pose = object_pose
        
        res = srv_spawn_model(req)

    def remove_robot(self):
        '''
        Remove an object
        https://github.com/ipa320/srs_public/blob/master/srs_user_tests/ros/scripts/spawn_object.py
        '''
        # print("Remove object")
        name = "iiwa"
        rospy.wait_for_service("/gazebo/delete_model")
        srv_delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        req = DeleteModelRequest()
        req.model_name = name
        # res = srv_delete_model(name)
        try:
			res = srv_delete_model(name)
        except rospy.ServiceException, e:
            print("ERROR: CANNOT REMOVE Robot: ", name)

    def load_start_controller(self):
        '''
        Call the service to spawn controller

        EX:
            # IIwa
            rosservice call /iiwa/controller_manager/load_controller "name: 'PositionJointInterface_trajectory_controller'"
            rosservice call /iiwa/controller_manager/switch_controller "{start_controllers: ['PositionJointInterface_trajectory_controller'], stop_controllers: [], strictness: 1}"

            Reflex:
            rosservice call /reflex_takktile_2/controller_manager/load_controller "name: 'joint_state_controller'"
            rosservice call /reflex_takktile_2/controller_manager/load_controller "name: 'preshape_1_position_controller'"
            rosservice call /reflex_takktile_2/controller_manager/load_controller "name: 'preshape_2_position_controller'"
            rosservice call /reflex_takktile_2/controller_manager/load_controller "name: 'proximal_joint_1_position_controller'"
            rosservice call /reflex_takktile_2/controller_manager/load_controller "name: 'proximal_joint_2_position_controller'"
            rosservice call /reflex_takktile_2/controller_manager/load_controller "name: 'proximal_joint_3_position_controller'"
            rosservice call /reflex_takktile_2/controller_manager/load_controller "name: 'distal_joint_1_position_controller'"
            rosservice call /reflex_takktile_2/controller_manager/load_controller "name: 'distal_joint_2_position_controller'"
            rosservice call /reflex_takktile_2/controller_manager/load_controller "name: 'distal_joint_3_position_controller'"
            rosservice call /reflex_takktile_2/controller_manager/switch_controller "{start_controllers: ['joint_state_controller', preshape_1_position_controller', 'preshape_2_position_controller', 'proximal_joint_1_position_controller', 'proximal_joint_2_position_controller', 'proximal_joint_3_position_controller', 'distal_joint_1_position_controller', 'distal_joint_2_position_controller', 'distal_joint_3_position_controller'], stop_controllers: [], strictness: 2}"
            rosservice call /reflex_takktile_2/controller_manager/switch_controller "{start_controllers: [], stop_controllers: ['joint_state_controller', preshape_1_position_controller', 'preshape_2_position_controller', 'proximal_joint_1_position_controller', 'proximal_joint_2_position_controller', 'proximal_joint_3_position_controller', 'distal_joint_1_position_controller', 'distal_joint_2_position_controller', 'distal_joint_3_position_controller'], strictness: 2}"
            
            rosservice call /reflex_takktile_2/controller_manager/switch_controller "{start_controllers: [preshape_1_position_controller', 'preshape_2_position_controller', 'proximal_joint_1_position_controller', 'proximal_joint_2_position_controller', 'proximal_joint_3_position_controller', 'distal_joint_1_position_controller', 'distal_joint_2_position_controller', 'distal_joint_3_position_controller'], stop_controllers: [], strictness: 2}"
            rosservice call /reflex_takktile_2/controller_manager/switch_controller "{start_controllers: [], stop_controllers: [preshape_1_position_controller', 'preshape_2_position_controller', 'proximal_joint_1_position_controller', 'proximal_joint_2_position_controller', 'proximal_joint_3_position_controller', 'distal_joint_1_position_controller', 'distal_joint_2_position_controller', 'distal_joint_3_position_controller'], strictness: 9}"
        
        '''
        # iiwa
        srv_load_model_iiwa = rospy.ServiceProxy('/iiwa/controller_manager/load_controller', LoadController)
        rospy.wait_for_service("/iiwa/controller_manager/load_controller")
        req_iiwa_load = LoadController()
        req_iiwa_load.name = "PositionJointInterface_trajectory_controller"
        srv_load_model_iiwa(req_iiwa_load.name)
        srv_switch_model_iiwa = rospy.ServiceProxy('/iiwa/controller_manager/switch_controller', SwitchController)
        rospy.wait_for_service("/iiwa/controller_manager/switch_controller")
        srv_switch_model_iiwa(['PositionJointInterface_trajectory_controller'], [], 1)
        
        # rosservice call /reflex_takktile_2/controller_manager/load_controller "name: 'joint_state_controller'"
        srv_load_model_reflex = rospy.ServiceProxy('/reflex_takktile_2/controller_manager/load_controller', LoadController)
        rospy.wait_for_service("/reflex_takktile_2/controller_manager/load_controller")
        req_load_model_reflex_joint= LoadController()
        req_load_model_reflex_joint.name  = "joint_state_controller"
        srv_load_model_reflex(req_load_model_reflex_joint.name)

        # Other controller...
        req_load_model_reflex_joint.name  = "preshape_1_position_controller"
        srv_load_model_reflex(req_load_model_reflex_joint.name)
        req_load_model_reflex_joint.name  = "preshape_2_position_controller"
        srv_load_model_reflex(req_load_model_reflex_joint.name)
        req_load_model_reflex_joint.name  = "proximal_joint_1_position_controller"
        srv_load_model_reflex(req_load_model_reflex_joint.name)
        req_load_model_reflex_joint.name  = "proximal_joint_2_position_controller"
        srv_load_model_reflex(req_load_model_reflex_joint.name)
        req_load_model_reflex_joint.name  = "proximal_joint_3_position_controller"
        srv_load_model_reflex(req_load_model_reflex_joint.name)
        req_load_model_reflex_joint.name  = "distal_joint_1_position_controller"
        srv_load_model_reflex(req_load_model_reflex_joint.name)
        req_load_model_reflex_joint.name  = "distal_joint_2_position_controller"
        srv_load_model_reflex(req_load_model_reflex_joint.name)
        req_load_model_reflex_joint.name  = "distal_joint_3_position_controller"
        srv_load_model_reflex(req_load_model_reflex_joint.name)
        srv_switch_model_reflex = rospy.ServiceProxy('/reflex_takktile_2/controller_manager/switch_controller', SwitchController)
        rospy.wait_for_service("/reflex_takktile_2/controller_manager/switch_controller")
        srv_switch_model_reflex(['joint_state_controller', 'preshape_1_position_controller', 'preshape_2_position_controller', 'proximal_joint_1_position_controller', 'proximal_joint_2_position_controller', 'proximal_joint_3_position_controller', 'distal_joint_1_position_controller', 'distal_joint_2_position_controller', 'distal_joint_3_position_controller'], [], 1)

    def start_controller(self):
        '''
        Call the service to start controller
        '''
        # iiwa
        srv_switch_model_iiwa = rospy.ServiceProxy('/iiwa/controller_manager/switch_controller', SwitchController)
        rospy.wait_for_service("/iiwa/controller_manager/switch_controller")
        srv_switch_model_iiwa(['PositionJointInterface_trajectory_controller'], [], 1)
        print("Loaded controller: iiwa")

        # Reflex
        srv_switch_model_reflex = rospy.ServiceProxy('/reflex_takktile_2/controller_manager/switch_controller', SwitchController)
        rospy.wait_for_service("/reflex_takktile_2/controller_manager/switch_controller")
        srv_switch_model_reflex(['joint_state_controller', 'preshape_1_position_controller', 'preshape_2_position_controller', 'proximal_joint_1_position_controller', 'proximal_joint_2_position_controller', 'proximal_joint_3_position_controller', 'distal_joint_1_position_controller', 'distal_joint_2_position_controller', 'distal_joint_3_position_controller'], [], 1)

    def load_stop_controller(self):
        '''
        Call the service to stop controller
        '''
        # IIWA
        rospy.wait_for_service("/iiwa/controller_manager/switch_controller")
        srv_switch_model_iiwa = rospy.ServiceProxy('/iiwa/controller_manager/switch_controller', SwitchController)
        # service message
        srv_switch_model_iiwa([], ['PositionJointInterface_trajectory_controller'], 1)
        

        # Robotics hand
        rospy.wait_for_service("/reflex_takktile_2/controller_manager/switch_controller")
        srv_switch_model_reflex = rospy.ServiceProxy('/reflex_takktile_2/controller_manager/switch_controller', SwitchController)

        # Call
        srv_switch_model_reflex([], ['joint_state_controller', 'preshape_1_position_controller', 'preshape_2_position_controller', 'proximal_joint_1_position_controller', 'proximal_joint_2_position_controller', 'proximal_joint_3_position_controller', 'distal_joint_1_position_controller', 'distal_joint_2_position_controller', 'distal_joint_3_position_controller'], 1)

    def random_target_position(self):
        '''
        set a random target
        '''

        # Workspace
        # self.check_workspace(pose)
        result = False
        object_pose = Pose()
        while result==False:
            object_pose.position.x = random.uniform(0.4, 0.6)
            object_pose.position.y = random.uniform(-0.45, 0.45)
            object_pose.position.z = self.constant_z
            result = True
            # result = self.check_workspace(object_pose)
        self.target_position[0] = object_pose.position.x
        self.target_position[1] = object_pose.position.y
        self.target_position[2] = object_pose.position.z

        return result

    def random_ee_position(self):
        '''
        set a random start end effector position
        '''
        # Workspace
        # self.check_workspace(pose)
        result = False
        object_pose = Pose()
        while result==False:
            object_pose.position.x = random.uniform(0.4, 0.7)
            object_pose.position.y = random.uniform(-0.45, 0.45)
            object_pose.position.z = self.constant_z
            result = self.check_workspace(object_pose)
        self.ee_position[0] = object_pose.position.x
        self.ee_position[1] = object_pose.position.y
        self.ee_position[2] = object_pose.position.z
        return result


    def set_for_evaluation(self, target_point):
        '''
        Set the target for the end effector allowing to evaluate our model

        input: vector 3D 

        '''
        for i in range(3):
            self.target_position[i] = target_point[i]

    def set_mode(self, mode="training"):
        '''
        Change the environment in training mode, predicting, evaluating
        (Change the initatilisation part mainly)
        Default: training
        training 
        predicting
        evaluating


        return the new setting
        '''
        if mode == "training" or mode == "predicting" or mode == "evaluating":
            self.env_mode = mode
        else:
            self.env_mode = "training"

        return self.env_mode


    def callback_real_robot(self, msg):
        '''
        Save the new position of the real end effector of the kuka
        '''
        # Issue when it is filling the matrix and an other function try to get the values.
        # A possible failure is when the vector is filled and the values will be not get for the right time!
        try:
            
            self.current_robot_pose[0] = msg.pose.position.x
            self.current_robot_pose[1] = msg.pose.position.y
            self.current_robot_pose[2] = msg.pose.position.z
            self.current_robot_pose[3] = msg.pose.orientation.x
            self.current_robot_pose[4] = msg.pose.orientation.y
            self.current_robot_pose[5] = msg.pose.orientation.z
            self.current_robot_pose[6] = msg.pose.orientation.w

            self.new_observation = self.current_robot_pose
            self.new_observation.append(self.target_position[0])
            self.new_observation.append(self.target_position[1])
            self.new_observation.append(self.target_position[2])
        except:
            print("[ ERROR]: Don't have the position of the object!")
