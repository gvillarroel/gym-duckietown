#!/usr/bin/env python3
from gym_duckietown import utils
import argparse
import yaml
import numpy
import heapq
import time

parser = argparse.ArgumentParser()
parser.add_argument("--init-point", default="2,1")
args = parser.parse_args()


def str_matrix(matrix):
  ret = "[\n"
  for row in matrix:
    ret = ret+str(row) +"\n"
  ret = ret + "]"
  return ret

def equals_tile(tile1, tile2):
  return tile1[0] == tile2[0] and tile1[1] == tile2[1]

map_name = "udem1"
map_file_path = utils.get_file_path("../gym_duckietown/maps", 
                                    map_name, 
                                    "yaml")
map_data = []
with open(map_file_path, 'r') as f:
  map_data = yaml.load(f, Loader=yaml.Loader)

tiles = map_data['tiles']

print('tiles')
print(tiles)

height = len(tiles)
width = len(tiles[0])
tiles_weight = numpy.zeros((height, width))
max_weight = 1000
start_tile = args.init_point.split(',')
start_tile = (int(start_tile[0]),int(start_tile[1]))

print('start_tile')
print(start_tile)

finish_tile = (1,5)

tiles_direction = [[" " for x in range(width)] for y in range(height)] 

print('tiles_direction')
print(str_matrix(tiles_direction))

for j, row in enumerate(tiles):
  for i, tile in enumerate(row):
    drivable = False
    if '/' in tile:
      drivable = True
    elif '4' in tile:
      drivable = True
    if drivable:
      tiles_weight[j][i] = max_weight


# Djikstra
tiles_weight[start_tile[1]][start_tile[0]] = 1
print('tiles_weight')
print(tiles_weight)

# Este es el heap
marked_nodes = []

# Poner en el heap el primer elemento.
heapq.heappush(marked_nodes, (1, start_tile))
arrived = False

cost = 0
# Mientras hayan valores en el heap.
while marked_nodes and not arrived:
  current_node = heapq.heappop(marked_nodes)
  # Revisamos el nodo de mayor prioridad.
  checking_node = current_node[1]

  # Chequeamos su peso.
  checking_weight = tiles_weight[checking_node[1]][checking_node[0]]

  # Vamos al tile superior, restamos 1 a la coordenada y del tile.
  up_tile = (checking_node[0], checking_node[1] -1)

  # Si es el buscado, lo encontramos.
  if equals_tile(up_tile, finish_tile):
    arrived = True

  # Vamos al tile inferior, anniadimos 1 a la coordenada y del tile.
  down_tile = (checking_node[0], checking_node[1] + 1)
  if equals_tile(down_tile, finish_tile):
    arrived = True

  # Asi para los 4 lados.
  left_tile = (checking_node[0]-1, checking_node[1])
  if equals_tile(left_tile, finish_tile):
    arrived = True
  right_tile = (checking_node[0]+1, checking_node[1])
  if equals_tile(right_tile, finish_tile):
    arrived = True
  visiting_tiles = [up_tile, down_tile, left_tile, right_tile]
  directions = ["D", "U", "R", "L"]

  # Reviso que hay en cada tile.
  for tile, direction in zip(visiting_tiles, directions):

    # Si lo encontre y no es el que reviso ahora, continuo al sgte.
    if arrived and not equals_tile(tile, finish_tile):
      continue

    # tomo el peso del tile.
    tile_weight = tiles_weight[tile[1]][tile[0]]

    # Si llege a un tile sin camino, continuo al sgte.
    if arrived and not tile_weight == 0:
      continue
 
    # Sumo el peso de la heuristica con el actual.
    next_cost = checking_weight + 1
    
    # Si es mayor lo salto.
    if arrived and not next_cost >= tile_weight:
      continue

    # No es mayor, ese valor queda como el peso.  
    tiles_weight[tile[1]][tile[0]] = next_cost

    # Se añade el nodo al heap con el nuevo paso.
    heapq.heappush(marked_nodes, (next_cost, tile))

    # Se guarda la direccion del tile.
    tiles_direction[tile[1]][tile[0]] = direction 
  
  print('temp tiles weight')
  print(tiles_weight)
  
  # Si llegamos al nodo buscado, salimos.
  if arrived:
    break 

print("-- weight matrix --")
print(tiles_weight)

print("-- reverse directions --")
print(str_matrix(tiles_direction))

current_tile = finish_tile
current_direction = tiles_direction[finish_tile[1]][finish_tile[0]]
tiles_direction[finish_tile[1]][finish_tile[0]] = "F"

while current_direction != " ":
  if current_direction == "R":
    current_tile = (current_tile[0]+1, current_tile[1])
    current_direction = tiles_direction[current_tile[1]][current_tile[0]]
    tiles_direction[current_tile[1]][current_tile[0]] = "L"

  if current_direction == "L":
    current_tile = (current_tile[0]-1, current_tile[1])
    current_direction = tiles_direction[current_tile[1]][current_tile[0]]
    tiles_direction[current_tile[1]][current_tile[0]] = "R"
  
  if current_direction == "U":
    current_tile = (current_tile[0], current_tile[1]-1)
    current_direction = tiles_direction[current_tile[1]][current_tile[0]]
    tiles_direction[current_tile[1]][current_tile[0]] = "D"
  if current_direction == "D":
    current_tile = (current_tile[0], current_tile[1]+1)
    current_direction = tiles_direction[current_tile[1]][current_tile[0]]
    tiles_direction[current_tile[1]][current_tile[0]] = "U"


print("-- directions --")
print(str_matrix(tiles_direction))

print("-- final tiles weight --")
print(tiles_weight)