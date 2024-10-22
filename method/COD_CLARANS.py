import numpy as np
import networkx as nx
from scipy.spatial import distance
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import sys

# 장애물 생성 관련 모듈 경로 추가
sys.path.append('/home/psi/Desktop/research/mission_planning/clustering')
import map.create_task ,map.map1 , map.obstacle_map1

#################### init maps ###################

# 장애물 및 태스크 생성
obs = map.obstacle_map1.obstacles()
obstacle_map = map.map1.generate_obstacle_map(1, 2, obs)
width, height = (25, 25)
tasks = map.create_task.create_tasks()
tasks = map.create_task.recreate_task(tasks, obstacle_map)

# 장애물 데이터를 cod_clarans에 맞게 변환 (다각형으로 처리)
def convert_obstacles_to_polygons(obstacle_map):
    polygons = []
    for obstacle in obstacle_map:
        x, y = obstacle['x'], obstacle['y']
        width, height = obstacle['width'], obstacle['height']
        
        # 사각형의 네 꼭짓점 계산
        polygon = [
            (x, y),  # bottom-left
            (x + width, y),  # bottom-right
            (x + width, y + height),  # top-right
            (x, y + height)  # top-left
        ]
        polygons.append(polygon)
    return polygons

# 태스크 데이터를 cod_clarans에서 사용할 수 있도록 변환
def convert_tasks_to_points(tasks):
    return [(task['x'], task['y']) for task in tasks]

# 장애물과 태스크 데이터를 변환
obstacles = convert_obstacles_to_polygons(obstacle_map)
points = convert_tasks_to_points(tasks)

#################### cod_clarans 알고리즘 ###################

# 가시성 그래프를 생성하는 함수 (Visibility Graph)
def create_visibility_graph(points, obstacles):
    G = nx.Graph()

    # 각 포인트를 노드로 추가
    for i in range(len(points)):
        G.add_node(i)

    # 포인트들 간의 가시성 여부를 BSP-tree로 계산하여 연결
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if is_visible(points[i], points[j], obstacles):
                dist = distance.euclidean(points[i], points[j])
                G.add_edge(i, j, weight=dist)  # 가시성이 있으면 유클리디안 거리로 간선 추가
    return G

# 두 포인트 간 가시성을 계산하는 함수 (BSP-tree 적용)
def is_visible(p1, p2, obstacles):
    line = LineString([p1, p2])  # 두 점을 잇는 선분
    for obstacle in obstacles:
        # 선분이 장애물과 교차하는지 확인
        if intersects_obstacle(line, obstacle):
            return False  # 장애물과 교차하면 가시성 없음
    return True  # 교차하지 않으면 가시성 있음

# 장애물과의 교차 여부를 확인하는 함수
def intersects_obstacle(line, obstacle):
    poly = Polygon(obstacle)
    return line.intersects(poly)

# 장애물 거리를 계산하는 함수 (장애물을 피해가는 경로를 계산)
def compute_obstructed_distances(points, centers, obstacles):
    G = create_visibility_graph(points, obstacles)  # VG 생성
    dist_matrix = np.zeros((len(points), len(centers)))
    
    for i, point in enumerate(points):
        for j, center in enumerate(centers):
            try:
                # 가시성이 있으면 VG에서 최단 경로 계산
                dist_matrix[i, j] = nx.shortest_path_length(G, source=i, target=center, weight='weight')
            except nx.NetworkXNoPath:
                dist_matrix[i, j] = float('inf')  # 경로가 없으면 무한대로 설정
    
    return dist_matrix

# CLARANS 알고리즘
def cod_clarans(points, k, obstacles, max_iter=100):
    centers = np.random.choice(len(points), k, replace=False)
    for _ in range(max_iter):
        found_new = False
        for i, center in enumerate(centers):
            remain = centers[np.arange(len(centers)) != i]  # 다른 클러스터 중심들
            dist_matrix = compute_obstructed_distances(points, remain, obstacles)
            new_center = np.random.choice(len(points))
            if not should_prune(new_center, dist_matrix):
                centers[i] = new_center
                found_new = True
        if not found_new:
            break
    return centers

# 가지치기 함수
def should_prune(new_center, dist_matrix):
    current_error = np.sum(np.min(dist_matrix, axis=1) ** 2)
    new_error = np.sum(np.min(dist_matrix + np.random.randn(*dist_matrix.shape), axis=1) ** 2)
    return new_error >= current_error

# 클러스터 할당 (각 포인트가 속한 클러스터를 계산)
def assign_clusters(points, centers):
    clusters = np.zeros(len(points))
    for i, point in enumerate(points):
        # point와 points[center]를 numpy 배열로 변환하여 연산 수행
        distances = [np.linalg.norm(np.array(point) - np.array(points[center])) for center in centers]
        clusters[i] = np.argmin(distances)  # 가장 가까운 클러스터 중심에 할당
    return clusters

# 클러스터 경계를 그리기 위한 함수
def plot_convex_hull(points, ax, color):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], color=color)

# 시각화 함수 (간소화된 가시성 그래프 및 클러스터 경계 추가)
def visualize(points, centers, clusters, obstacles, G):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 장애물 그리기
    for obs in obstacles:
        obs_poly = Polygon(obs)
        x, y = obs_poly.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.5, label="Obstacle")

    # 포인트 그리기
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], color='blue', label="Points")

    # 클러스터 중심 그리기
    ax.scatter(points[centers, 0], points[centers, 1], color='red', marker='x', s=100, label="Cluster Centers")

    # 클러스터 영역 (Convex Hull) 그리기
    for i, center in enumerate(centers):
        cluster_points = points[clusters == i]  # 클러스터에 속한 포인트들
        if len(cluster_points) > 2:  # 최소한 3개 이상의 점이 있어야 Convex Hull을 그릴 수 있음
            plot_convex_hull(cluster_points, ax, 'green')

        # 클러스터 중심과 포인트 간의 가시성 그래프만 그리기
        for point in cluster_points:
            ax.plot([points[center, 0], point[0]], [points[center, 1], point[1]], 'g--', alpha=0.5)
    
    ax.set_title("COD-CLARANS Clustering Result with Simplified Visibility Graph and Cluster Boundaries")
    plt.show()

#################### cod_clarans 실행 ###################
# k 값은 클러스터 개수, max_iter는 최대 반복 횟수
k = 6
max_iter = 100

# cod_clarans 알고리즘을 호출하여 클러스터링 실행
centers = cod_clarans(points, k=k, obstacles=obstacles, max_iter=max_iter)

# 클러스터 할당
clusters = assign_clusters(points, centers)

# 가시성 그래프 생성 및 결과 시각화
G = create_visibility_graph(points, obstacles)
visualize(points, centers, clusters, obstacles, G)
