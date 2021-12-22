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
from gym_duckietown.simulator import WINDOW_HEIGHT, WINDOW_WIDTH
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

CENTER_WIDTH = CAMERA_WIDTH/2
CENTER_HEIGHT = CAMERA_HEIGHT/2


# 305.5718893575089, 0, 303.0797142544728,
# 0, 308.8338858195428, 231.8845403702499,
# 0, 0, 1,


################################################

env = DuckietownEnv(
    seed = 1,
    #map_name = "regress_4way_adam",
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
last_direction_blanca = 1
def seguidor_linea(blancas, amarillas, patos):
    """
    """
    action = np.array([0.0, 0.0])

    """
    inserte aquí el código para que su duckiebot pueda seguir líneas

    """
    y= 0
    x = 0.4
    b= 0.2
    m = 4
    y_blancas, y_amarillas, y_patos = (0,0,0,)
    global last_direction_blanca
    if len(blancas) > 0:
        ro = blancas[:,1].mean()
        theta = blancas[:,1].mean()
        a = blancas[:,2].mean()
        b = blancas[:,3].mean()
        d = 1 if a <= 0 else -1
        x00 = blancas[:,4].mean()
        x01 = blancas[:,5].mean()
        last_direction_blanca = d
        avg = np.abs(a) 
        es_derecha = x01 >= WINDOW_WIDTH/2 

        if es_derecha:
            y_blancas = np.abs(avg*m)
        else:
            y_blancas = np.abs(avg*m*8)*-1
        print(f"a:{a:.2f},b:{b:.2f}, theta:{theta:.2f}, x00:{x00:.2f}, x01:{x01:.2f}")
        
    if len(patos) > 0 and patos[:,3].max() > 0.3*CAMERA_HEIGHT:
        ARGMAX = patos[:,3].argmax()
        # Freno de emergencia
        # x = 0
        d = last_direction_blanca
        pos = d*(patos[ARGMAX,3]*m / CAMERA_HEIGHT)+(d*b)
        y_patos = pos
    else:
        if len(amarillas) > 0:
            avg = amarillas[:,2].mean()
            d = 1 if avg < 0 else -1
            avg = np.abs(1-avg)
            x01 = amarillas[:,5].mean()
            es_derecha = x01 >= WINDOW_WIDTH/2 
            if es_derecha:
                y_amarillas = (m*8*-1)+(b*-1)
            else:
                y_amarillas = (m)+(b)
    y = y_blancas + y_amarillas + y_patos
    #print(f"({y_blancas:.2f}, {y_amarillas:.2f},{y_patos:.2f})")
    action = np.array([x, y])

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
SIZE_IMAGE = (WINDOW_HEIGHT**2 + WINDOW_WIDTH**2)**.5
def detector_lineas(mask, image, threshold):
    detections= []
    mask[0:int(0.5*CAMERA_HEIGHT)] = 0
    image[0:int(0.5*CAMERA_HEIGHT)] = 0
    edges = cv2.Canny(mask, 50, 150, apertureSize = 3)
    """
    'HoughLines(image, rho, theta, threshold[
        , lines[, srn[, stn[, min_theta[, max_theta]]]]
        ]) -> lines\n.   @brief Finds lines in a binary image using the standard Hough transform.\n.   \n.   The function implements the standard or standard multi-scale Hough transform algorithm for line\n.   detection. See <http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm> for a good explanation of Hough\n.   transform.\n.   \n.   @param image 8-bit, single-channel binary source image. The image may be modified by the function.\n.   @param lines Output vector of lines. Each line is represented by a 2 or 3 element vector\n.   \\f$(\\rho, \\theta)\\f$ or \\f$(\\rho, \\theta, \\textrm{votes})\\f$ . \\f$\\rho\\f$ is the distance from the coordinate origin \\f$(0,0)\\f$ (top-left corner of\n.   the image). \\f$\\theta\\f$ is the line rotation angle in radians (\n.   \\f$0 \\sim \\textrm{vertical line}, \\pi/2 \\sim \\textrm{horizontal line}\\f$ ).\n.   \\f$\\textrm{votes}\\f$ is the value of accumulator.\n.   @param rho Distance resolution of the accumulator in pixels.\n.   @param theta Angle resolution of the accumulator in radians.\n.   @param threshold Accumulator threshold parameter. Only those lines are returned that get enough\n.   votes ( \\f$>\\texttt{threshold}\\f$ ).\n.   @param srn For the multi-scale Hough transform, it is a divisor for the distance resolution rho .\n.   The coarse accumulator distance resolution is rho and the accurate accumulator resolution is\n.   rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these\n.   parameters should be positive.\n.   @param stn For the multi-scale Hough transform, it is a divisor for the distance resolution theta.\n.   @param min_theta For standard and multi-scale Hough transform, minimum angle to check for lines.\n.   Must fall between 0 and max_theta.\n.   @param max_theta For standard and multi-scale Hough transform, maximum angle to check for lines.\n.   Must fall between min_theta and CV_PI.'
    """
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, 100,50)
    
    #cv2.HoughLinesP() 
    if lines is None or len(lines) == 0:
        return np.array([])
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + SIZE_IMAGE*(-b))
        y1 = int(y0 + SIZE_IMAGE*(a))
        x2 = int(x0 - SIZE_IMAGE*(-b))
        y2 = int(y0 - SIZE_IMAGE*(a))
        m = (y2 - y1) / ((x2 - x1)+0.0001)
        c = y1 - m*x1
        x00 = -c/m
        x01 = (WINDOW_HEIGHT-c)/m
        if np.abs(y1-y2) > 1:
            cv2.line(image, (x1,y1), (x2,y2), (0, 0, 255), 1, cv2.LINE_AA)
            detections.append((rho, theta, a, b, x00, x01, ))

    return np.array(detections)



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
    patos_mask = cv2.inRange(converted, np.array([10,50,150]) , np.array([25,255,255]))
    segment_image = cv2.bitwise_and(converted,converted, mask=patos_mask)
    patos = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)

    cuadros_patos = detector_objetos(patos_mask, patos, 100)
    #cv2.imshow("patos", patos)
    cv2.waitKey(1)
    
    #########################################################
    # Detectando la blanca
    blanca_mask = cv2.inRange(converted, np.array([0,0,150]) , np.array([180,25,255]))
    blanca_segment_image = cv2.bitwise_and(converted,converted, mask= blanca_mask)
    blanca = cv2.cvtColor(blanca_segment_image, cv2.COLOR_HSV2BGR)
    lineas_blancas = detector_lineas(blanca_mask, blanca, 120)
    cv2.imshow("blancas", blanca)
    cv2.waitKey(1)
    #########################################################
    # Detectando la amarillas
    amarilla_mask = cv2.inRange(converted, np.array([25,80,10]) , np.array([50,255,255]))
    amarilla_segment_image = cv2.bitwise_and(converted,converted, mask= amarilla_mask)
    amarilla = cv2.cvtColor(amarilla_segment_image, cv2.COLOR_HSV2BGR)
    lineas_amarillas = detector_lineas(amarilla_mask, amarilla, 80)
    #cv2.imshow("amarillas", amarilla)
    cv2.waitKey(1)
    #########################################################

    if mode == 0:
        action = manual_control()
    
    if mode == 1:
        action = seguidor_linea(lineas_blancas, lineas_amarillas, cuadros_patos)

    if mode == 2:
        action = seguidor_linea(lineas_blancas, lineas_amarillas, cuadros_patos)
        action = manual_control()
    #########################################################


    # print("Comparación localizador:")
    # print(f"Posición real {env.unwrapped.cur_pos} - Posición estimada {pos}")



    if done:
        print('done!')
        env.reset()


    env.render(mode="top_down")
    #env.render()
    # env.render()
    time.sleep(1.0 / env.unwrapped.frame_rate)



env.close()
