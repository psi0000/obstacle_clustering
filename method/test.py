import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, LineString
import random

# 태스크 좌표를 (x, y) 포맷으로 변환
def convert_tasks_to_points(tasks):
    """태스크 데이터를 (x, y) 좌표로 변환"""
    return [(task['x'], task['y']) for task in tasks]

# 장애물 데이터를 다각형으로 변환
def convert_obstacles_to_polygons(obstacle_map):
    """장애물 데이터를 다각형 리스트로 변환"""
    polygons = []
    for obstacle in obstacle_map:
        x, y = obstacle['x'], obstacle['y']
        width, height = obstacle['width'], obstacle['height']
        # 장애물을 사각형으로 정의
        polygon = [
            (x, y),  # bottom-left
            (x + width, y),  # bottom-right
            (x + width, y + height),  # top-right
            (x, y + height)  # top-left
        ]
        polygons.append(Polygon(polygon))
    return polygons

# 델로네 삼각분할 생성
def build_delaunay(points):
    """일반 델로네 삼각분할 생성"""
    tri = Delaunay(points)
    edges = []
    for simplex in tri.simplices:
        edges.append((simplex[0], simplex[1]))
        edges.append((simplex[1], simplex[2]))
        edges.append((simplex[2], simplex[0]))
    return edges

# 장애물과 교차하는지 확인
def intersects_obstacle(p1, p2, obstacles):
    """두 점 사이에 장애물이 있는지 확인"""
    line = LineString([p1, p2])
    for obstacle in obstacles:
        if line.intersects(obstacle):
            return True
    return False

# Phase 1: 너무 긴 엣지 제거
def phase1_eliminate_long_edges(points, edges, obstacles):
    """너무 긴 엣지 제거 + 장애물 처리"""
    if len(edges) == 0:  # 에러 방지 위해 len으로 확인
        return edges  # 엣지가 없으면 그대로 반환

    local_means = np.zeros(len(points))
    for i, point in enumerate(points):
        distances = [np.linalg.norm(points[edge[0]] - points[edge[1]]) for edge in edges if i in edge]
        if distances:
            local_means[i] = np.mean(distances)
        else:
            local_means[i] = 0

    global_stdev = np.std(local_means)

    valid_edges = []
    for edge in edges:
        dist = np.linalg.norm(points[edge[0]] - points[edge[1]])
        local_mean = np.mean([local_means[edge[0]], local_means[edge[1]]])
        if dist < local_mean + global_stdev:
            p1 = points[edge[0]]
            p2 = points[edge[1]]
            if not intersects_obstacle(p1, p2, obstacles):  # 장애물과 교차하지 않으면 추가
                valid_edges.append(edge)

    return valid_edges

# Phase 2: 고립된 포인트 복구
def phase2_reconnect_isolated_points(points, edges, obstacles):
    """고립된 점 복구"""
    connected_points = set([p for edge in edges for p in edge])
    isolated_points = [i for i in range(len(points)) if i not in connected_points]
    
    for i in isolated_points:
        nearest_neighbor = find_nearest_neighbor(i, points, edges, obstacles)
        if nearest_neighbor is not None:
            if not intersects_obstacle(points[i], points[nearest_neighbor], obstacles):  # 장애물 처리
                edges.append((i, nearest_neighbor))  # 고립된 점과 이웃을 연결
                edges.append((nearest_neighbor, i))  # 양방향 연결 추가

    return edges

# 가장 가까운 이웃 찾기 (장애물 고려)
def find_nearest_neighbor(point_idx, points, edges, obstacles):
    """가장 가까운 이웃 찾기"""
    min_dist = float('inf')
    nearest_neighbor = None
    for j in range(len(points)):
        if j != point_idx and not any((point_idx, j) == edge or (j, point_idx) == edge for edge in edges):
            dist = np.linalg.norm(points[point_idx] - points[j])
            if dist < min_dist and not intersects_obstacle(points[point_idx], points[j], obstacles):
                min_dist = dist
                nearest_neighbor = j
    return nearest_neighbor

# Phase 3: 이웃 확장 및 긴 엣지 제거
def phase3_extend_neighbors_and_remove_long_edges(points, edges, threshold, obstacles):
    """더 넓은 이웃 탐색 및 긴 엣지 제거"""
    extended_edges = []

    # 기존 엣지에서 이웃을 확장
    for edge in edges:
        p1, p2 = edge
        for i in range(len(points)):
            if i != p1 and i != p2:
                dist_p1 = np.linalg.norm(points[p1] - points[i])
                dist_p2 = np.linalg.norm(points[p2] - points[i])

                # threshold 기준으로 두 번 이동 내에서 이웃을 확장
                if dist_p1 < threshold and dist_p2 < threshold:
                    if not intersects_obstacle(points[p1], points[i], obstacles) and not intersects_obstacle(points[p2], points[i], obstacles):
                        extended_edges.append((p1, i))
                        extended_edges.append((p2, i))

    # 확장된 이웃들 중 긴 엣지 제거
    valid_extended_edges = phase1_eliminate_long_edges(points, extended_edges, obstacles)
    return valid_extended_edges

# 시각화 함수
def visualize_edges(points, edges, obstacles, title="엣지 시각화"):
    """포인트와 엣지를 시각화"""
    fig, ax = plt.subplots()
    
    # 장애물 그리기
    for obs in obstacles:
        x, y = obs.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.5, label="Obstacle")

    # 포인트 그리기
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], color='red', label="Points")
    
    # 엣지 그리기
    for edge in edges:
        p1 = points[edge[0]]
        p2 = points[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.6)

    ax.set_title(title)
    plt.axis('equal')
    plt.show()

# 연결된 엣지들을 기반으로 포인트 그룹 생성 (DFS를 사용하여 그룹화)
def find_connected_groups(edges, num_points):
    """연결된 엣지들을 기반으로 포인트들을 그룹화"""
    visited = [False] * num_points
    groups = []

    def dfs(point, group):
        visited[point] = True
        group.append(point)
        for edge in edges:
            if edge[0] == point and not visited[edge[1]]:
                dfs(edge[1], group)
            elif edge[1] == point and not visited[edge[0]]:
                dfs(edge[0], group)

    for i in range(num_points):
        if not visited[i]:
            group = []
            dfs(i, group)
            if group:  # 비어 있지 않은 그룹만 추가
                groups.append(group)

    return groups

# 시각화 함수 (그룹화된 포인트들을 동일한 색상으로 표시)
def visualize_grouped_points(points, edges, obstacles, title="Grouped Points"):
    """연결된 포인트를 동일한 색상으로 시각화"""
    fig, ax = plt.subplots()

    # 장애물 그리기
    for obs in obstacles:
        x, y = obs.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.5, label="Obstacle")

    # 연결된 포인트 그룹 찾기
    groups = find_connected_groups(edges, len(points))

    print("group's ", len(groups))
    # 각 그룹에 대해 다른 색상 적용
    for group in groups:
        group_color = (random.random(), random.random(), random.random())  # 랜덤 색상 생성
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                if (group[i], group[j]) in edges or (group[j], group[i]) in edges:
                    p1 = points[group[i]]
                    p2 = points[group[j]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=group_color)
            ax.scatter(points[group[i]][0], points[group[i]][1], color=group_color, s=20)

    ax.set_title(title)
    plt.axis('equal')
    plt.show()
def find_connected_groups(edges, num_points):
    """연결된 엣지들을 기반으로 포인트들을 그룹화"""
    visited = [False] * num_points
    groups = []

    def dfs(point, group):
        visited[point] = True
        group.append(point)
        for edge in edges:
            if edge[0] == point and not visited[edge[1]]:
                dfs(edge[1], group)
            elif edge[1] == point and not visited[edge[0]]:
                dfs(edge[0], group)

    for i in range(num_points):
        if not visited[i]:
            group = []
            dfs(i, group)
            if group:  # 비어 있지 않은 그룹만 추가
                groups.append(group)

    return groups

# 거리 기반의 하위 클러스터 형성 함수
def distance_based_subclusters(points, groups, num_clusters):
    """각 그룹을 거리 기반으로 하위 클러스터로 나누기"""
    subclusters = []
    from scipy.spatial import distance_matrix
    for group in groups:
        # 그룹에 속한 포인트 좌표 추출
        group_points = points[group]
        # 거리 행렬 계산
        dist_matrix = distance_matrix(group_points, group_points)
        
        # 초기 중심점 무작위 선택 (K-means 초기화 방식)
        centers = group_points[np.random.choice(len(group_points), num_clusters, replace=False)]
        clusters = [[] for _ in range(num_clusters)]
        
        for _ in range(10):  # K-means와 비슷하게 반복
            clusters = [[] for _ in range(num_clusters)]
            
            # 각 점을 가장 가까운 중심에 배정
            for i, point in enumerate(group_points):
                distances = [np.linalg.norm(point - center) for center in centers]
                closest_center = np.argmin(distances)
                clusters[closest_center].append(group[i])
            
            # 새로운 중심 계산
            for j in range(num_clusters):
                if clusters[j]:  # 클러스터가 비어있지 않은 경우에만 중심 업데이트
                    centers[j] = np.mean(points[clusters[j]], axis=0)
        
        # 결과 subclusters에 추가
        subclusters.extend(clusters)

    return subclusters

# 시각화 함수
def visualize_subclusters(points, subclusters, obstacles, title="Subclusters by Distance"):
    """하위 클러스터를 서로 다른 색상으로 시각화"""
    fig, ax = plt.subplots()

    # 장애물 그리기
    for obs in obstacles:
        x, y = obs.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.5, label="Obstacle")

    # 각 하위 클러스터에 색상 적용
    for subcluster in subclusters:
        cluster_color = (random.random(), random.random(), random.random())
        for point_index in subcluster:
            ax.scatter(points[point_index][0], points[point_index][1], color=cluster_color, s=20)

    ax.set_title(title)
    plt.axis('equal')
    plt.show()
def main(tasks,obstacle_map):
    # 태스크 데이터를 (x, y) 좌표로 변환
    points = convert_tasks_to_points(tasks)
    points = np.array(points)

    # 장애물 데이터를 폴리곤으로 변환
    obstacles = convert_obstacles_to_polygons(obstacle_map)

    # 델로네 삼각분할을 사용하여 엣지 생성
    delaunay_edges = build_delaunay(points)

    # Phase 1: 너무 긴 엣지 제거 + 장애물 처리
    valid_edges_phase1 = phase1_eliminate_long_edges(points, delaunay_edges, obstacles)

    # Phase 2: 고립된 점 복구 + 장애물 처리
    valid_edges_phase2 = phase2_reconnect_isolated_points(points, valid_edges_phase1, obstacles)

    # Phase 3: 이웃 확장 및 긴 엣지 제거 + 장애물 처리
    final_edges = phase3_extend_neighbors_and_remove_long_edges(points, valid_edges_phase2, threshold=10, obstacles=obstacles)


    # 시각화
    # visualize_edges(points, valid_edges_phase1, obstacles, title="Phase 1 후 유효한 엣지")
    # visualize_edges(points, valid_edges_phase2, obstacles, title="Phase 2 후 고립된 점 복구")
    # visualize_edges(points, final_edges, obstacles, title="Phase 3 후 확장된 이웃 및 최종 엣지") 


    # 최종 엣지를 시각화하면서 각 그룹을 다른 색상으로 적용
    visualize_grouped_points(points, final_edges, obstacles, title="AUTOCLUST++")
    # 삼각형 면적 계산
     # 연결된 포인트 그룹 찾기
    groups = find_connected_groups(final_edges, len(points))

    # 거리 기반 하위 클러스터 생성
    subclusters = distance_based_subclusters(points, groups, num_clusters=3)

    # 최종 하위 클러스터 시각화
    visualize_subclusters(points, subclusters, obstacles, title="AUTOCLUST++ with Distance-Based Subclusters")

