import numpy as np
import random

'''---CLASES DE UTILIDAD---'''
####################################################################
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

class celda:
  infX = ''
  infY = ''
  supX = ''
  supY = ''

  def __init__(self, infx, supx, infy, supy):
    self.infX = infx
    self.infY = infy
    self.supX = supx
    self.supY = supy
####################################################################

'''----INICIALIZACIONES----'''
####################################################################
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
#####################################################################

#####################################################################
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
######################################################################
#Declaramos las 16 celdas del laberinto
mazeCells = []

#(De abajo hacia arriba)#
#Primera fila
mazeCells.append(celda(0, 2, 0, 2))
mazeCells.append(celda(2, 4, 0, 2))
mazeCells.append(celda(4, 6, 0, 2))
mazeCells.append(celda(6, 8, 0, 2))

#Segunda fila
mazeCells.append(celda(0, 2, 2, 4))
mazeCells.append(celda(2, 4, 2, 4))
mazeCells.append(celda(4, 6, 2, 4))
mazeCells.append(celda(6, 8, 2, 4))

#Tercera fila
mazeCells.append(celda(0, 2, 4, 6))
mazeCells.append(celda(2, 4, 4, 6))
mazeCells.append(celda(4, 6, 4, 6))
mazeCells.append(celda(6, 8, 4, 6))

#Cuarta fila
mazeCells.append(celda(0, 2, 6, 8))
mazeCells.append(celda(2, 4, 6, 8))
mazeCells.append(celda(4, 6, 6, 8))
mazeCells.append(celda(6, 8, 6, 8))
######################################################################


'''---FUNCIONES DE UTILIDAD---'''
####################################################################
# CIRCLE/RECTANGLE
def circleRect(cx,cy,radius, rx, ry, rw, rh):

  # temporary variables to set edges for testing
  testX = cx;
  testY = cy;

  # which edge is closest?
  if (cx < rx):
    testX = rx      # test left edge
  elif (cx > rx+rw):
    testX = rx+rw   # right edge
  if (cy < ry):
    testY = ry      # top edge
  elif (cy > ry+rh):
    testY = ry+rh   # bottom edge

  # get distance from closest edges
  distX = cx-testX
  distY = cy-testY
  distance = np.sqrt((distX*distX) + (distY*distY))

  # if the distance is less than the radius, collision!
  if (distance <= radius):
    return True

  return False

def sampleFree(x,y,r):

  if x+r > limsup_X or x-r < liminf_X:
    return False

  if y+r > limsup_Y or y-r < liminf_Y:
    return False

  for rec in obstacles:
    rx = rec.pos[0]
    ry = rec.pos[1]
    rw = rec.width
    rh = rec.height

    if circleRect(x,y,r, rx, ry, rw, rh) == True:
      return False

  return True


def collisionFree(x1, y1, x2, y2, r):
  numSamples = 10

  for i in range(numSamples+1):
    xm = (x2 - x1)*(i/numSamples) + x1
    ym = (y2 - y1)*(i/numSamples) + y1

    #Si un circulo de la muestra no es libre de colision,
    #regresamos que la trayectoria no es posible
    if sampleFree(xm,ym,r) == False:
      return False

  return True

# Función para verificar la distancia mínima entre puntos
def checkMinDist(punto, puntos, min_dist):
    for p in puntos:
        if np.linalg.norm(np.array(p) - np.array(punto)) < min_dist:
            return False
    return True

def Near(V, x, y, r):
  B = []

  for vtx in V:
    v = np.array([vtx.x, vtx.y])
    u = np.array([x, y])
    if np.linalg.norm(u - v) < r:
      B.append(vtx)
  return B

def Nearest(V,x,y):
  vmin = np.array([np.inf,np.inf])

  for vtx in V:
    v = np.array([vtx.x, vtx.y])
    u = np.array([x, y])
    if np.linalg.norm(u - v) < np.linalg.norm(vmin - u):
      vmin = v

  return vmin
####################################################################

####################################################################
def PRM(numSamples, xinit, velocity, delta_T, pointsInTxt):
    if pointsInTxt == False:
        file = open("PRMpoints.txt", "w")
    else:
        file = open("PRMpoints.txt", "r")
        print("Leyendo archivo de texto existente...")
        lines = file.readlines()

    V = []
    E = []

    V.append(xinit)

    i = 0
    while i < numSamples:

        if pointsInTxt == False:
            while True:
                #Generamos punto aleatorio xrand
                xrand_x = np.random.uniform(liminf_X, limsup_X)
                xrand_y = np.random.uniform(liminf_Y, limsup_Y)

                #Verificamos que el punto aleatorio este libre de colisiones
                if sampleFree(xrand_x, xrand_y, connect_radius):
                    break

            file.write(str(xrand_x))
            file.write(" ")
            file.write(str(xrand_y))
            file.write("\n")
            V.append(vertex(xrand_x, xrand_y))

        else:

            line = lines[i]
            points = line.split()

            # Dividir la línea en dos números
            xrand_x = float(points[0])
            xrand_y = float(points[1])
            V.append(vertex(xrand_x, xrand_y))

        i += 1


    neighbors = [[] for it in range(len(V))] #En este arreglo pondremos los vecinos de cada vertice

    for v_id in range(len(V)):
        v = V[v_id]

        U = Near(V, v.x, v.y, velocity*delta_T)
        U.remove(v)

        for u in U:
            if collisionFree(v.x, v.y, u.x, u.y, connect_radius):
                E.append(((v.x, v.y), (u.x, u.y)))

                #Agregamos a los vecinos
                u_id = np.where(np.array(V) == u)[0][0]
                neighbors[v_id].append(u_id)

        if v_id % 500 == 0:
            print(v_id)


    V = np.array(V)
    E = np.array(E)
    file.close()

    return V, E, neighbors
####################################################################

####################################################################
def goalReached(v, goal, radius):
    if np.linalg.norm(np.array([v.x, v.y]) -  np.array([goal[0], goal[1]])) <= radius:
        return True
    else:
        return False

def distance(u, v):
    return np.linalg.norm(np.array([u.x, u.y]) - np.array([v.x, v.y]))

def Dijkstra(V, Neighbors, init, goal, goalradius):
    #Arreglo de distancias
    D = []
    #Inicializamos en infinito
    for v in V:
        D.append(np.inf)

    for i in range(len(V)):
        if V[i].x == init.x and V[i].y == init.y:
            init_id = i

    D[init_id] = 0

    #Arreglo para registrar los nodos visitados
    Visited = []
    for v in V:
        Visited.append(False)

    #Queue
    Q = set()
    for i in range(len(V)):
        Q.add(i)

    #Parents array
    Parents = []
    for i in range(len(V)):
        Parents.append(None)

    #Mientras la queue sea no vacia
    while(len(Q)):

        while(True):
            current_id = random.choice(tuple(Q))
            if V[current_id].activate == True:
                break
            else:
                Q.remove(current_id)

        #Seleccionamos el nodo con menor distancia
        for q in Q:
            if D[q] < D[current_id]:
                current_id = q

        #Marcamos el nodo como visitado
        Visited[current_id] = True

        if D[current_id] == np.infty:
            break

        if goalReached(V[current_id], goal, goalradius):
            break

        Q.remove(current_id)

        for n in Neighbors[current_id]:
            if V[n].activate == True:
                dist = D[current_id] + distance(V[current_id], V[n])
                if dist < D[n]:
                    D[n] = dist
                    Parents[n] = current_id

    if Parents[current_id] == None:
        return None, np.inf

    PathV = []
    PathE = []
    current_node = V[current_id]
    PathV.append(current_node)

    while current_node.x != init.x or current_node.y != init.y:
        parent_node = V[Parents[current_id]]
        parent_id = Parents[current_id]

        PathE.append(((parent_node.x, parent_node.y), (current_node.x, current_node.y)))

        current_node = parent_node
        current_id = parent_id
        PathV.append(current_node)

    PathV.reverse()

    return PathV, PathE
#################################################################################


#################################################################################
def getPath(V, Neighbors, init, goal, goalradius):
    initialPos = Nearest(V, init[0], init[1])
    initialVtx = vertex(initialPos[0], initialPos[1])

    optimalPathV, optimalPathE = Dijkstra(V, Neighbors, initialVtx, goal, goalradius)

    if optimalPathV is None:
      print("No hay camino desde el punto de inicio al punto final.")

    return optimalPathV, optimalPathE

def getPathObst(V, Neighbors, init, obst, obstradius, goal, goalradius):
    initialPos = Nearest(V, init[0], init[1])
    initialVtx = vertex(initialPos[0], initialPos[1])

    nearObst = Near(V, obst[0], obst[1], 2*obstradius)

    for v in V:
       if v in nearObst:
          v.activate = False

    optimalPathV, optimalPathE = Dijkstra(V, Neighbors, initialVtx, goal, goalradius)

    #if optimalPathV is None:
      #print("No hay camino desde el punto de inicio al punto final.")

    for v in V:
       v.activate = True

    return optimalPathV, optimalPathE





