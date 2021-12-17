#!/usr/bin/env python

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Duckietown-udem1-v1")
parser.add_argument('--map-name', default='patos')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# Parametros para el detector de patos
filtro_1 = np.array([0,0.0,0.68]) 
filtro_2 = np.array([70,1.0,1.0]) 

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Handler para reiniciar el ambiente
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

# Registrar el handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):
    """
    Funcion que se llama en step.
    """


    # Aquí se controla el duckiebot
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action[0]+=0.44
    if key_handler[key.DOWN]:
        action[0]-=0.44
    if key_handler[key.LEFT]:
        action[1]+=1
    if key_handler[key.RIGHT]:
        action[1]-=1
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5


    # aquí se obtienen las observaciones y se setea la acción
    # obs consiste en un imagen de 640 x 480 x 3
    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if done:
        print('done!')
        env.reset()
        env.render()


    # Detección de patos
    # El objetivo de hoy es detectar los patos ajustando los valores del detector

    obs = obs/255.0
    obs = obs.astype(np.float32)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()
    
    # Filtro por color https://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html
    
    #Cambiar tipo de color de BGR a HSV
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    # Filtrar colores de la imagen en el rango utilizando 
    mask = cv2.inRange(converted, filtro_1, filtro_2)

    # Bitwise-AND mask and original 
    segment_image = cv2.bitwise_and(converted,converted, mask= mask)
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    
    kernel = np.ones((5,5),np.uint8)

    # Esto corresponde a hacer un Opening
    # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    #Operacion morfologica erode
    image_out = cv2.erode(image, kernel, iterations = 5)    
    #Operacion morfologica dilate
    image_out = cv2.dilate(image, kernel, iterations = 10)

    # https://docs.opencv.org/trunk/d3/d05/tutorial_py_table_of_contents_contours.html
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    

    cv2.imshow("patos", image)
    cv2.waitKey(1)



    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()