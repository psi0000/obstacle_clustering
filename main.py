import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/home/psi/Desktop/research/mission_planning/clustering')
import map.create_task ,map.map1 , map.obstacle_map1
import argparse

#################### init maps ###################
parser = argparse.ArgumentParser(description='Choose a clustering method')
parser.add_argument('--m', type=str, choices=['k_means_plus_manhattan','k_means_plus_euclidean', 'dbscan','autoclust_p','ascdt_p'],
                    required=True, help='The clustering method to use')
args = parser.parse_args()

obs=map.obstacle_map1.obstacles()
obstacle_map=map.map1.generate_obstacle_map(1, 2,obs)
width, height = (25,25)
tasks=map.create_task.create_tasks()
tasks=map.create_task.recreate_task(tasks,obstacle_map)

facilitators = map.obstacle_map1.facilitators()


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
    import method.dbscan as dbscan
    x=np.array([[task["x"], task["y"]] for task in tasks])
    dbscan.dbscan_with_obstacles(tasks, obstacle_map)

elif methods=='autoclust_p':
    import method.autoclust_p as autoclust
    autoclust.main(tasks, obstacle_map)

elif methods=='ascdt_p':
    import method.ascdt_plus as ascdt
    taskss = np.array([[task['x'], task['y']] for task in tasks])
    ascdt_p = ascdt.ASCDT_PLUS(taskss, obstacle_map, facilitators)
    ascdt_p.run_phase1_to_phase4()
else:
    pass



###################### visualize ####################3

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

