import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/home/psi/Desktop/research/mission_planning/clustering')
import map.create_task ,map.map1 , map.obstacle_map1
import argparse

#################### init maps ###################
parser = argparse.ArgumentParser(description='Choose a clustering method')
parser.add_argument('--m', type=str, choices=['k_means_plus_manhattan','k_means_plus_euclidean', 'dbscan'],
                    required=True, help='The clustering method to use')
args = parser.parse_args()

obs=map.obstacle_map1.obstacles()
obstacle_map=map.map1.generate_obstacle_map(1, 2,obs)
width, height = (25,25)
tasks=map.create_task.create_tasks()
tasks=map.create_task.recreate_task(tasks,obstacle_map)



######################### method select ###############
methods=args.m

if methods=='k_means_plus_euclidean':
    import method.k_means_plus_euclidean as k__
        
    class Point:
            def __init__(self, x=0, y=0, group=0):
                self.x = x
                self.y = y
                self.group = group

    points = points = [Point(task["x"], task["y"]) for task in tasks]
    k = 6
    _, clusterCenterTrace = k__.kMeans(points, k)
    k__.showClusterAnalysisResults(points, clusterCenterTrace)
elif methods=='k_means_plus_manhattan':
    import method.k_means_plus_manhattan as k__
        
    class Point:
            def __init__(self, x=0, y=0, group=0):
                self.x = x
                self.y = y
                self.group = group
    points = [Point(task["x"], task["y"]) for task in tasks]
    k = 6
    clusterCenterTrace = k__.k_means(points, k,width*2, height*2, obstacle_map)
    k__.show_cluster_analysis_results(points, clusterCenterTrace, width, height, obstacle_map)

    # k__.showClusterAnalysisResults(points, clusterCenterTrace)

elif methods=='dbscan':
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=5, min_samples=3)
    x=np.array([[task["x"], task["y"]] for task in tasks])
    labels = dbscan.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis')
    plt.title('DBSCAN Clustering')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()  

else:
    pass



######################visualize####################3

plt.figure(figsize=(10, 10))
plt.xlim(0, 50)
plt.ylim(0, 50)

# Plot each obstacle as a square
for obstacle in obstacle_map:
    rect = plt.Rectangle((obstacle['x'], obstacle['y']), obstacle['width'], obstacle['height'],
                            color='red', alpha=0.7)
    plt.gca().add_patch(rect)

# Plot each task as a point
x_coords = [task['x'] for task in tasks]
y_coords = [task['y'] for task in tasks]
plt.scatter(x_coords, y_coords, c='blue', s=50, label='Tasks')

# Annotate each point with its ID
for task in tasks:
    plt.annotate(str(task['id']), (task['x'], task['y']), textcoords="offset points", xytext=(0, 5), ha='center')

plt.title('Obstacle Map')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.show()

