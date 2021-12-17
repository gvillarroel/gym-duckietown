#!/usr/bin/env python3

import time
import sys
import argparse
import math
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
from random import random, randint

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--no-pause",action="store_true", help="do not pause on failure")
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(map_name=args.map_name, domain_rand=False,max_steps=500)
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

recompense = 0

p_0 = env.cur_pos 
distance = 0

params=(0.5, 19, 8, 0 ,0)
best = 0

velocity = params[0]

k_p = params[1]
k_d = params[2]

while True:
    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

   

    vel_ang = (
        k_p * distance_to_road_center + k_d * angle_from_straight_in_rads
    )
    p_1 = env.cur_pos
    obs, rec, fin, info = env.step([velocity, vel_ang])
    recompense+=rec
    p_2 = env.cur_pos

    distance += ((p_2[0]-p_1[0])**2+(p_2[1]-p_1[1])**2+(p_2[2]-p_1[2])**2)**(1/2) 

    env.render()

    if fin:
      if recompense*distance>best:
        params = (velocity,k_p,k_d, distance, recompense)
        best = recompense*distance
      if env.step_count <= env.max_steps:
        velocity = params[0]
        k_p = params[1]
        k_d = params[2]
      velocity+=randint(-100,100)/1000
      k_p += randint(-100,100)/1000
      k_d += randint(-100,100)/1000
      
      recompense = 0
      distance = 0
      print("Usando velocidad=%.3f, kp=%s, kd=%s" %(velocity,k_p,k_d))
      print(params)
      obs = env.reset()
      

