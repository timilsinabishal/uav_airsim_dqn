#! /usr/bin/env python
"""Environment for Microsoft AirSim Unity Quadrotor using AirSim python API
"""
import csv
import math
import pprint
import time

import torch
from PIL import Image

import numpy as np

import airsim
#import setup_path

from detector import HSVDetector

MOVEMENT_INTERVAL = 1

class DroneEnv(object):
    """Drone environment class using AirSim python API"""

    def __init__(self, useDepth=False):
        self.client = airsim.MultirotorClient()
        self.useDepth = useDepth
        self.detector = HSVDetector()
        state, image = self.get_obs()
        self.last_dist = self.get_distance(self.client.getMultirotorState().kinematics_estimated.position, image)
        self.quad_offset = (0, 0, 0)

    def step(self, action):
        """Step"""
        #print("new step ------------------------------")

        self.quad_offset = self.interpret_action(action)
        #print("quad_offset: ", self.quad_offset)

        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.moveByVelocityAsync(
            quad_vel.x_val + self.quad_offset[0],
            quad_vel.y_val + self.quad_offset[1],
            quad_vel.z_val + self.quad_offset[2],
            MOVEMENT_INTERVAL
        ).join()
        collision = self.client.simGetCollisionInfo().has_collided

        time.sleep(0.5)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        if quad_state.z_val < - 7.3:
            self.client.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -7, 1).join()

        state, image = self.get_obs()
        result, done = self.compute_reward(quad_state, quad_vel, collision, image)

        return state, result, done, image

    def reset(self):
        self.client.reset()
        state, image = self.get_obs()
        self.last_dist = self.get_distance(self.client.getMultirotorState().kinematics_estimated.position, image)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        self.client.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -7, 1).join()

        obs, image = self.get_obs()

        return obs, image

    def get_obs(self):
        if self.useDepth:
            # get depth image
            responses = self.client.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, pixels_as_float=True)])
            response = responses[0]
            img1d = np.array(response.image_data_float, dtype=np.float)
            img1d = img1d * 3.5 + 30
            img1d[img1d > 255] = 255
            image = np.reshape(img1d, (responses[0].height, responses[0].width))
            image_array = Image.fromarray(image).resize((84, 84)).convert("L")
        else:
            # Get rgb image
            responses = self.client.simGetImages(
                [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
            )
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            image = img1d.reshape(response.height, response.width, 3)
            image_array = Image.fromarray(image).resize((84, 84)).convert("L")

        obs = np.array(image_array)

        return obs, image

    def get_distance(self, quad_state, frame):
        """Get distance between current state and goal state"""
        x, y, frame = self.detector.process(frame)
        pts = np.array([3, -76, -7])
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - pts)
        return dist

    def compute_reward(self, quad_state, quad_vel, collision, image):
        """Compute reward"""

        reward = 0

        if collision:
            reward = -50
        else:
            dist = self.get_distance(quad_state, image)
            diff = self.last_dist - dist

            if dist < 10:
                reward = -20
            if dist > 30:
                reward = -20
            else:
                reward = 500


            self.last_dist = dist

        done = 0
        if reward <= -10:
            done = 1
            time.sleep(1)
        elif reward > 499:
            done = 1
            time.sleep(1)

        return reward, done


    def interpret_action(self, action):
        """Interprete action"""
        scaling_factor = 3

        if action == 0:
            self.quad_offset = (scaling_factor, 0, 0)
        elif action == 1:
            self.quad_offset = (-scaling_factor, 0, 0)
        elif action == 2:
            self.quad_offset = (0, scaling_factor, 0)
        elif action == 3:
            self.quad_offset = (0, -scaling_factor, 0)

        return self.quad_offset
