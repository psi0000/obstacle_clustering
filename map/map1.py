import matplotlib.pyplot as plt
import sys
sys.path.append('/home/psi/Desktop/research/mission_planning/clustering')
import map.obstacle_map1 as obstacle_map1
def generate_obstacle_map(obstacle_size=1, resolution=2,obstacles=None):
    obstacle_map = []
    if obstacles is None:
        obstacles = []
    for ox, oy in obstacles:
        obstacle = {
            'x': ox * resolution,
            'y': oy * resolution,
            'width': obstacle_size * resolution,
            'height': obstacle_size * resolution
        }
        obstacle_map.append(obstacle)
    
    return obstacle_map

