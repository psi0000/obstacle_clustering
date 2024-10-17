import math
import random
import heapq
import matplotlib.pyplot as plt

FLOAT_MAX = 1e100
random.seed(42)

class Point:
    def __init__(self, x=0, y=0, group=0):
        self.x, self.y, self.group = x, y, group

def is_within_bounds(x, y, width, height):
    return 0 <= x < width and 0 <= y < height

def is_obstacle(x, y, obstacle_map):
    for obstacle in obstacle_map:
        if obstacle['x'] <= x < obstacle['x'] + obstacle['width'] and \
           obstacle['y'] <= y < obstacle['y'] + obstacle['height']:
            return True
    return False

def a_star(start, goal, width, height, obstacle_map):
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            break
        
        for direction in directions:
            next_node = (current[0] + direction[0], current[1] + direction[1])

            if is_within_bounds(next_node[0], next_node[1], width, height) and not is_obstacle(next_node[0], next_node[1], obstacle_map):
                new_cost = cost_so_far[current] + 1
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + abs(goal[0] - next_node[0]) + abs(goal[1] - next_node[1])
                    heapq.heappush(open_list, (priority, next_node))
                    came_from[next_node] = current

    return cost_so_far.get(goal, FLOAT_MAX)

def solve_distance_between_points(pointA, pointB, width, height, obstacle_map):
    start = (int(pointA.x), int(pointA.y))
    goal = (int(pointB.x), int(pointB.y))
    distance = a_star(start, goal, width, height, obstacle_map)
    return distance

def get_nearest_center(point, centers, width, height, obstacle_map):
    min_index = -1
    min_distance = FLOAT_MAX
    for i, center in enumerate(centers):
        distance = solve_distance_between_points(point, center, width, height, obstacle_map)
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return min_index

def k_means(points, k, width, height, obstacle_map, max_iterations=100):
    centers = random.sample(points, k)
    
    for iteration in range(max_iterations):
        # 각 포인트에 대해 가장 가까운 클러스터 중심을 찾음
        for point in points:
            point.group = get_nearest_center(point, centers, width, height, obstacle_map)

        new_centers = []
        for i in range(k):
            cluster_points = [p for p in points if p.group == i]
            if cluster_points:
                mean_x = sum(p.x for p in cluster_points) / len(cluster_points)
                mean_y = sum(p.y for p in cluster_points) / len(cluster_points)
                new_centers.append(Point(mean_x, mean_y, i))
            else:
                # 가장 많은 포인트를 가진 클러스터에서 하나의 포인트를 가져옴
                largest_cluster = max(range(k), key=lambda x: len([p for p in points if p.group == x]))
                chosen_point = random.choice([p for p in points if p.group == largest_cluster])
                new_centers.append(Point(chosen_point.x, chosen_point.y, i))

        if all(new_centers[i].x == centers[i].x and new_centers[i].y == centers[i].y for i in range(k)):
            print("Converged!")
            break
        
        centers = new_centers

    for i in range(k):
        cluster_size = sum(1 for p in points if p.group == i)
        print(f"Cluster {i}: {cluster_size} tasks")

    return centers

def show_cluster_analysis_results(points, centers, width, height, obstacle_map):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    plt.figure(figsize=(9, 9))
    
    # 장애물 시각화
    for obstacle in obstacle_map:
        rect = plt.Rectangle((obstacle['x'], obstacle['y']), obstacle['width'], obstacle['height'],
                             color='gray', alpha=0.7)
        plt.gca().add_patch(rect)
    
    # 각 포인트를 시각화
    for idx, point in enumerate(points):
        plt.scatter(point.x, point.y, color=colors[point.group % len(colors)], s=50)

    # 클러스터 중심 시각화 및 라벨링
    for idx, center in enumerate(centers):
        plt.scatter(center.x, center.y, color='black', marker='x', s=200)
        cluster_size = sum(1 for p in points if p.group == idx)
        plt.text(center.x, center.y, f'Cluster {idx}\n({cluster_size} tasks)', fontsize=10, color='black', fontweight='bold')

    plt.title('K-Means Clustering with Obstacles')
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.grid(True)
    plt.show()