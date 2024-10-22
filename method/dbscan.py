import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def is_obstacle(x, y, obstacle_map):
    """장애물이 있는지 확인하는 함수"""
    for obstacle in obstacle_map:
        if obstacle['x'] <= x < obstacle['x'] + obstacle['width'] and \
           obstacle['y'] <= y < obstacle['y'] + obstacle['height']:
            return True
    return False

def filter_points_with_obstacles(tasks, obstacle_map):
    """장애물이 있는 영역에 있는 포인트를 필터링"""
    filtered_tasks = []
    for task in tasks:
        if not is_obstacle(task['x'], task['y'], obstacle_map):
            filtered_tasks.append(task)
    return filtered_tasks

def dbscan_with_obstacles(tasks, obstacle_map, eps=5, min_samples=3):
    # 장애물에 있는 포인트를 제외하고 DBSCAN 실행
    filtered_tasks = filter_points_with_obstacles(tasks, obstacle_map)
    x = np.array([[task["x"], task["y"]] for task in filtered_tasks])

    # DBSCAN 알고리즘 적용
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(x)

    # 시각화
    plt.figure(figsize=(9, 9))
    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', label='Clusters')
    
    # 장애물 시각화
    for obstacle in obstacle_map:
        rect = plt.Rectangle((obstacle['x'], obstacle['y']), obstacle['width'], obstacle['height'],
                             color='black', alpha=0.5, label='Obstacle')
        plt.gca().add_patch(rect)

    plt.title('DBSCAN Clustering with Obstacles')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()