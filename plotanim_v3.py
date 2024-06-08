import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import numpy as np
from matplotlib import collections  as mc
import random
import matplotlib.patches as patches
import matplotlib.animation as animation
import pandas as pd
import random

import tensorflow as tf
import keras as keras
from keras.layers import PReLU
from keras import layers, losses, models
from keras.models import Model

class Obstacle:
  pos = ''
  width = ''
  height = ''

  def __init__(self, p, w, h):
    self.pos = p
    self.width = w
    self.height = h

class vertex:
  x = ''
  y = ''
  activate = True

  def __init__(self, xpos, ypos):
    self.x = xpos
    self.y = ypos

class edge:
    u = ''
    v = ''

    def __init__(self, u_vtx, v_vtx):
        self.u = u_vtx
        self.v = v_vtx

rootVertex = vertex(1.0, 1.0) #nodo inicial
connect_radius = 0.15 #radio de conexion

#----PARAMETROS DEL ROBOT----#
robot_r = 0.1
v_r = 1 #velocidad del robot
d_t = 0.25 #delta_t (incremento en tiempo)

#-----PARAMETROS DEL OBSTACULO DINAMICO-----#
obstacle_r = 0.1
v_obst = 1 #velocidad del obstaculo dinamico
obst_reach = v_obst*d_t + obstacle_r #radio de alcance
obstaculo_speed = v_obst*d_t*0.5

liminf_X = 0
limsup_X = 8
liminf_Y = 0
limsup_Y = 8


obstacles = []
wt = 0.25/2

#Paredes del laberinto
obstacles.append(Obstacle((2-wt,0), 0.25, 2 + wt))
obstacles.append(Obstacle((2+wt, 2-wt), 2, 0.25))

obstacles.append(Obstacle((0, 4-wt), 4-wt, 0.25))
obstacles.append(Obstacle((4-wt, 4-wt), 0.25, 2 + 2*wt))
obstacles.append(Obstacle((2-wt, 6-wt), 2, 0.25))

obstacles.append(Obstacle((6-wt, 6-wt), 0.25, 2 + 2*wt))

obstacles.append(Obstacle((6-wt, 2-wt), 0.25, 2 + 2*wt))
obstacles.append(Obstacle((6+wt, 4-wt), 2-wt, 0.25))


model_path = 'C:/Users/ramon/Documents/Universidad/Proyecto 2024/Modelos/trained_model_mlp_v2_25_obst_1000_epochs.h5'
mlp3 = keras.models.load_model(model_path)

inicio = [1, 1]
goal = [3, 5]
obstaculo_x = 1.5
obstaculo_y = 6

#############################--INICIALIZACION DEL MAPA--############################################################
fig, ax = plt.subplots()
ax.set_xlim(liminf_X, limsup_X)
ax.set_ylim(liminf_Y, limsup_Y)
ax.set_aspect('equal')

for i in range(len(obstacles)):
  ax.add_patch(Rectangle(obstacles[i].pos, obstacles[i].width, obstacles[i].height,
             facecolor = 'red',
             fill=True))

ax.add_patch(Circle((goal[0], goal[1]), 0.75, edgecolor='green', ls = '--', fill = False))
###################################################################################################################

# Variable global para la posición actual
qt = [[inicio[0], inicio[1], obstaculo_x, obstaculo_y, goal[0], goal[1]]]

# Parche para el obstáculo
obstacle_patch = Circle((obstaculo_x, obstaculo_y), obstacle_r, facecolor='red')
obstacle_artist = ax.add_patch(obstacle_patch)

# Parche para la región de alcanzabilidad del obstaculo
ob_reach_patch = Circle((obstaculo_x, obstaculo_y), obst_reach, edgecolor='red', ls = '--', fill = False)
ob_reach_artist = ax.add_patch(ob_reach_patch)

# Parche para el robot
robot_patch = Circle((inicio[0], inicio[1]), robot_r, facecolor='blue', alpha=0.75)
robot_artist = ax.add_patch(robot_patch)

# Función de inicialización
def init():
    return [robot_artist, obstacle_artist, ob_reach_artist]

# Crear función de actualización para la animación
def update(frame):
    global qt, obstaculo_y, obstaculo_speed
    
    # Actualizar la posición vertical del obstáculo
    obstaculo_y += obstaculo_speed
    if obstaculo_y >= 7 or obstaculo_y <= 5 + 0.01:
        obstaculo_speed *= -1  # Cambiar de dirección si se alcanzan los límites

    # Actualizar la posición del parche del obstáculo
    obstacle_patch.center = (obstaculo_x, obstaculo_y)
    # Actualizar la posición de la region de alcanzabilidad
    ob_reach_patch.center = (obstaculo_x, obstaculo_y)

    # Si el robot no ha llegado al objetivo
    if np.linalg.norm(np.array([qt[0][0], qt[0][1]]) - np.array(goal)) >= 0.75:
        qnext = mlp3.predict(qt, verbose=0)
        qt = np.array([[qnext[0][0], qnext[0][1], obstaculo_x, obstaculo_y, goal[0], goal[1]]])
        robot_patch.center = (qnext[0][0], qnext[0][1])

    else:
        # Detener la animación cuando el robot llega al objetivo
        if frame <= 500:
           ax.add_patch(Circle((goal[0], goal[1]), 0.5, facecolor='green', alpha=0.9))
        ani.event_source.stop()

    return [robot_artist, obstacle_artist, ob_reach_artist]


# Crear animación
ani = animation.FuncAnimation(fig, update, frames=1000, repeat=False, blit=False)

fig.subplots_adjust(left=0.055, right=0.4, top=1, bottom=0.332)
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# Mostrar animación
plt.grid()
plt.show()