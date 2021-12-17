#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse

import pyglet
from pyglet.window import key
import time

import numpy as np

import gym
import gym_duckietown

from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

import cv2
import numpy as np


"""
Defina aquí variables globales. 
Tenga en cuenta lo siguiente, las variables que defina aquí podrán ser usadas en las funciones que defina más abajo pero solo para leer y utilizar su contenido
No sobreescribirlo. 

"""


###########################################


#  fx, fy, cx, cy
# CAMERA_PARAMS = [305.5718893575089, 308.8338858195428,  303.0797142544728, 231.8845403702499]
CAMERA_PARAMS = [220.2460277141687,  238.6758484095299, 301.8668918355899, 227.0880056118307]

# K = [CAMERA_PARAMS[0], 0, CAMERA_PARAMS[2], 0, CAMERA_PARAMS[1], CAMERA_PARAMS[3], 0, 0, 1]
K = CAMERA_PARAMS

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480



# 305.5718893575089, 0, 303.0797142544728,
# 0, 308.8338858195428, 231.8845403702499,
# 0, 0, 1,


################################################

env = DuckietownEnv(
    seed = 1,
    #map_name = "simple",
    map_name = "small_loop_pedestrians",
    
    draw_curve = False,
    draw_bbox = False,
    domain_rand = False,
    frame_skip = 1,
    distortion = False,
)

# env.user_tile_start = [2, 1]
# env.start_pose = [[env.road_tile_size * 5, 0, env.road_tile_size * 1.33], np.pi]

obstacles = [{'kind': 'duckie',
             'pos': [4.3, 2.3],
             'rotate': 200,
             'height': 0.05,
             'static': False},
             
             {'kind': 'duckie',
             'pos': [4.2, 2.2],
             'rotate': 90,
             'height': 0.08,
             'static': False}
             ]


# env._load_objects({'objects': obstacles})

# Enter main event loop
env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()

    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif symbol == key.PAGEDOWN:
        env.unwrapped.cam_angle[0] = 1

    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)


    if symbol == key.RETURN:
        from PIL import Image
        im = Image.fromarray(obs)
        im.save('screen.png')



key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


KEEP_GOIN = True


def manual_control():
    """
    Debe reemplazar este código para que el duckiebot lo maneje su 
    """
    action = np.array([0.0, 0.0])

    if key_handler[key.UP] and KEEP_GOIN:
        action = np.array([0.44, 0.0])

    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])

    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])

    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])

    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    return action


def seguidor_linea(blancas, amarillas):
    """
    """
    action = np.array([0.0, 0.0])

    """
    inserte aquí el código para que su duckiebot pueda seguir líneas

    """
    velocity = 1
    y= 0
    b= 0.2
    if len(blancas) > 0:
        #y = (1-blancas[0][2])*2
        d = -1 if blancas[0][2] < 0 else 1
        y = (blancas[0][2]*-8)+(b*d)
    action = np.array([0.4, y])

    print(blancas)

    return action


def detector_objetos(mask, image, min_area):
    detections = []
    kernel = np.ones((5,5),np.uint8)
    image_out = cv2.erode(mask, kernel, iterations = 2)    
    #Operacion morfologica dilate
    image_out = cv2.dilate(image_out, kernel, iterations = 10)
    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
            #Obtener rectangulo
        x, y, w, h = cv2.boundingRect(cnt)
        
        #Filtrar por area minima
        if w*h > min_area:
            x2 = x + w  # obtener el otro extremo
            y2 = y + h
            #Dibujar un rectangulo en la imagen
            cv2.rectangle(image, (int(x), int(y)), (int(x2),int(y2)), (255,0,0), 3)
            detections.append((x, y, w, h, ))
    return np.array(detections)

def detector_lineas(mask, image, min_area):
    detections= []
    edges = cv2.Canny(mask, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    #cv2.HoughLinesP() 
    if lines is None or len(lines) == 0:
        return np.array([])
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image, (x1,y1), (x2,y2), (0, 0, 255), 1, cv2.LINE_AA)
        #detections.append((x1, y1, x2, y2, [rho, theta, a, b]))
        detections.append((rho, theta, a, b, ))

    return np.array(detections)

def detector_patos(image):
    detections = []
    #########################################################
    # b) Bounding Boxes
    MIN_AREA = 500
    kernel = np.ones((5,5),np.uint8)
    #Operacion morfologica erode
    image_out = cv2.erode(image, kernel, iterations = 2)    
    #Operacion morfologica dilate
    image_out = cv2.dilate(image_out, kernel, iterations = 10)
    global KEEP_GOIN
    KEEP_GOIN = True
    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
            #Obtener rectangulo
        x, y, w, h = cv2.boundingRect(cnt)
        
        #Filtrar por area minima
        if w*h > MIN_AREA:
            x2 = x + w  # obtener el otro extremo
            y2 = y + h
            #Dibujar un rectangulo en la imagen
            cv2.rectangle(image2, (int(x), int(y)), (int(x2),int(y2)), (255,0,0), 3)
            detections.append((x, y, w, h, ))
        #########################################################
        # c)  Freno de Emergencia
        if h > CAMERA_HEIGHT * 0.33:
            KEEP_GOIN = False
    """
    inserte aquí el código para que su duckiebot pueda detectar patos (filtro de color)
    """
    return detections


run = True
action = [0., 0.]


# MODE 0: Control Manual
# MODE 1: Automático

mode = 1

pos = []

while run:

    ###########################################

    if key_handler[key.ENTER]:
        #
        print("change mode")
        mode = (mode + 1) % 2


    ########################################################

    obs, reward, done, info = env.step(action)
    real_pos = env.unwrapped.cur_pos

    # obs = obs/255.0
    obs = obs.astype(np.uint8)
    image = cv2.UMat(obs[:, :, [2, 1, 0]]).get()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    #########################################################
    # a) Filtro de color
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    # Filtrar colores de la imagen en el rango utilizando 
    mask = cv2.inRange(converted, np.array([10,50,150]) , np.array([25,255,255]))
    segment_image = cv2.bitwise_and(converted,converted, mask= mask)
    image2 = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)

    #detector_patos(mask)
    #cv2.imshow("patos", image2)
    #cv2.waitKey(1)
    
    #########################################################
    # Detectando la blanca
    blanca_mask = cv2.inRange(converted, np.array([0,0,150]) , np.array([180,25,255]))
    blanca_segment_image = cv2.bitwise_and(converted,converted, mask= blanca_mask)
    blanca = cv2.cvtColor(blanca_segment_image, cv2.COLOR_HSV2BGR)
    lineas_blancas = detector_lineas(blanca_mask, blanca, 100)
    cv2.imshow("blancas", blanca)
    cv2.waitKey(1)
    #########################################################
    # Detectando la amarillas
    amarilla_mask = cv2.inRange(converted, np.array([25,80,10]) , np.array([50,255,255]))
    amarilla_segment_image = cv2.bitwise_and(converted,converted, mask= amarilla_mask)
    amarilla = cv2.cvtColor(amarilla_segment_image, cv2.COLOR_HSV2BGR)
    lineas_amarillas = detector_lineas(amarilla_mask, amarilla, 100)
    cv2.imshow("amarillas", amarilla)
    cv2.waitKey(1)
    #########################################################

    if mode == 0:
        action = manual_control()
    
    if mode == 1:
        action = seguidor_linea(lineas_blancas, lineas_amarillas)

    #########################################################


    # print("Comparación localizador:")
    # print(f"Posición real {env.unwrapped.cur_pos} - Posición estimada {pos}")



    if done:
        print('done!')
        env.reset()


    #env.render(mode="top_down")
    env.render()
    # env.render()
    time.sleep(1.0 / env.unwrapped.frame_rate)



env.close()
