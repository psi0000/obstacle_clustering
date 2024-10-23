import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/psi/Desktop/research/mission_planning/clustering')
import map.create_task, map.map1, map.obstacle_map1

import random
import networkx as nx

class ASCDT_PLUS:
    def __init__(self, tasks, obstacle_map, facilitators=None, beta=1.0):
        self.points = tasks  # 입력된 tasks 데이터
        self.obstacle_map = obstacle_map  # 장애물 정보
        self.facilitators = facilitators  # facilitators 정보
        self.tri = Delaunay(self.points)
        self.edges = self.get_edges_from_triangulation(self.tri)
        self.beta = beta
    
    def get_edges_from_triangulation(self, tri):
        """ 델로네 삼각형에서 엣지들을 추출하는 함수 """
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = (min(simplex[i], simplex[(i + 1) % 3]), 
                        max(simplex[i], simplex[(i + 1) % 3]))
                edges.add(edge)
        return np.array(list(edges))

    def calculate_edge_lengths(self):
        """ 각 엣지의 길이를 계산하는 함수 """
        lengths = []
        for edge in self.edges:
            p1, p2 = self.points[edge[0]], self.points[edge[1]]
            length = np.linalg.norm(p1 - p2)
            lengths.append(length)
        return np.array(lengths)

    def global_mean_and_variation(self, lengths):
        """ 엣지 길이들의 글로벌 평균과 변동성을 계산하는 함수 """
        global_mean = np.mean(lengths)
        global_variation = np.std(lengths)
        return global_mean, global_variation

    def calculate_mean1_dt(self):
        """ 각 점 P_i에 연결된 엣지들의 평균 길이를 계산하는 함수 (Mean1_DT) """
        mean1_dt = []
        for i in range(len(self.points)):
            connected_edges = [edge for edge in self.edges if i in edge]
            if connected_edges:
                lengths = [np.linalg.norm(self.points[edge[0]] - self.points[edge[1]]) for edge in connected_edges]
                mean1_dt.append(np.mean(lengths))
            else:
                mean1_dt.append(np.inf)  # 연결된 엣지가 없는 경우 매우 큰 값
        return np.array(mean1_dt)

    def global_cut_value(self, global_mean, global_variation, mean1_dt):
        """ Global_Cut_Value 계산하는 함수 """
        return global_mean + (global_mean / mean1_dt) * global_variation

    def remove_long_edges(self, lengths, global_mean, global_variation, mean1_dt):
        """ Global_Cut_Value를 초과하는 엣지를 제거하는 함수 (Phase 1) """
        cut_values = self.global_cut_value(global_mean, global_variation, mean1_dt)
        # 각 엣지에 연결된 두 점의 Global_Cut_Value 중 큰 값을 사용
        edge_cut_values = np.array([max(cut_values[edge[0]], cut_values[edge[1]]) for edge in self.edges])
        
        # 엣지 길이와 해당 엣지에 적용된 cut_values 비교
        keep_edges = self.edges[lengths <= edge_cut_values]
        return keep_edges

    def calculate_local_cut_value(self, mean2_gi, mean_variation):
        """ Local_Cut_Value 계산 함수 (Phase 2) """
        return mean2_gi + self.beta * mean_variation

    def remove_local_long_edges(self, edges, original_lengths):
        """ 로컬 긴 엣지를 제거하는 함수 (Phase 2) """
        # Phase 1에서 남은 엣지들의 길이를 다시 계산
        filtered_lengths = np.array([original_lengths[np.where((self.edges == edge).all(axis=1))[0][0]] for edge in edges])

        mean2_gi = []
        mean_variation = []
        
        # 각 점에 대해 로컬 평균 길이와 변동성 계산
        for i in range(len(self.points)):
            connected_edges = [edge for edge in edges if i in edge]
            if connected_edges:
                local_lengths = [filtered_lengths[np.where((edges == edge).all(axis=1))[0][0]] for edge in connected_edges]
                mean2_gi.append(np.mean(local_lengths))
                mean_variation.append(np.std(local_lengths))
            else:
                mean2_gi.append(np.inf)
                mean_variation.append(0)  # 연결된 엣지가 없으면 변동성은 0으로 설정
        
        # 로컬 컷 값 계산
        local_cut_values = self.calculate_local_cut_value(np.array(mean2_gi), np.array(mean_variation))
        
        # 각 엣지에 대해 두 점의 Local_Cut_Value 중 큰 값을 선택
        edge_cut_values = np.array([max(local_cut_values[edge[0]], local_cut_values[edge[1]]) for edge in edges])
        
        # 남은 엣지 길이와 Local_Cut_Value 비교하여 긴 엣지 제거
        keep_edges = edges[filtered_lengths <= edge_cut_values]
        
        return keep_edges

    def calculate_local_link_cut_value(self, mean2_gi, mean_variation):
        """ Local Link Cut Value 계산 (Phase 3) """
        return mean2_gi + self.beta * mean_variation

    def remove_local_link_edges(self, edges, original_lengths):
        """ 로컬 링크 엣지를 제거하는 함수 (Phase 3) """
        # Phase 2에서 남은 엣지들의 길이를 다시 계산
        filtered_lengths = np.array([original_lengths[np.where((self.edges == edge).all(axis=1))[0][0]] for edge in edges])

        mean2_gi = []
        mean_variation = []
        
        # 각 점에 대해 로컬 평균 길이와 변동성 계산
        for i in range(len(self.points)):
            connected_edges = [edge for edge in edges if i in edge]
            if connected_edges:
                local_lengths = [filtered_lengths[np.where((edges == edge).all(axis=1))[0][0]] for edge in connected_edges]
                mean2_gi.append(np.mean(local_lengths))
                mean_variation.append(np.std(local_lengths))
            else:
                mean2_gi.append(np.inf)
                mean_variation.append(0)  # 연결된 엣지가 없으면 변동성은 0으로 설정
        
        # 로컬 링크 컷 값 계산
        local_link_cut_values = self.calculate_local_link_cut_value(np.array(mean2_gi), np.array(mean_variation))
        
        # 각 엣지에 대해 두 점의 Local_Link_Cut_Value 중 큰 값을 선택
        edge_cut_values = np.array([max(local_link_cut_values[edge[0]], local_link_cut_values[edge[1]]) for edge in edges])
        
        # 남은 엣지 길이와 Local_Link_Cut_Value 비교하여 긴 엣지 제거
        keep_edges = edges[filtered_lengths <= edge_cut_values]
        
        return keep_edges

    def obstacle_collision(self, edge, obstacle_map):
        """ 엣지가 장애물과 교차하는지 확인하는 함수 (Phase 4) """
        p1, p2 = self.points[edge[0]], self.points[edge[1]]
        
        for obstacle in obstacle_map:
            ox = obstacle['x']
            oy = obstacle['y']
            ow = obstacle['width']
            oh = obstacle['height']
            
            # 엣지와 장애물의 교차 여부를 판단 (단순 AABB 충돌 검사)
            if (min(p1[0], p2[0]) < ox + ow and max(p1[0], p2[0]) > ox and
                min(p1[1], p2[1]) < oy + oh and max(p1[1], p2[1]) > oy):
                return True  # 교차함
        return False  # 교차하지 않음


    def remove_obstacle_crossing_edges(self, edges, obstacle_map):
        """ 장애물과 교차하는 엣지를 제거하는 함수 (Phase 4) """
        keep_edges = []
        for edge in edges:
            if not self.obstacle_collision(edge, obstacle_map):
                keep_edges.append(edge)
        return np.array(keep_edges)
    def merge_clusters_with_facilitators(self, edges):
        """ Facilitators에 의해 클러스터 병합하는 함수 """
        G = nx.Graph()
        
        # 기존 엣지를 그래프에 추가
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        
        # Facilitators가 연결하는 클러스터 병합
        for facilitator in self.facilitators:
            fx, fy, influence_radius = facilitator['x'], facilitator['y'], facilitator['influence']
            
            # Facilitator 근처의 점들을 찾아서 연결
            nearby_points = [i for i, point in enumerate(self.points) if self.is_near_facilitator(point, fx, fy, influence_radius)]
            
            # 근처 점들끼리 클러스터 병합
            for i in range(len(nearby_points) - 1):
                G.add_edge(nearby_points[i], nearby_points[i + 1])
        
        # 병합된 엣지 반환
        merged_edges = list(G.edges)
        return merged_edges

    def is_near_facilitator(self, point, fx, fy, influence_radius):
        """ 특정 포인트가 facilitators의 영향 범위 내에 있는지 확인하는 함수 """
        return np.linalg.norm(point - np.array([fx, fy])) <= influence_radius
    
    def run_phase1_to_phase4(self):
        """ Phase 1~4 실행 """
        lengths = self.calculate_edge_lengths()  # 엣지 길이 계산
        global_mean, global_variation = self.global_mean_and_variation(lengths)  # 글로벌 평균 및 변동성
        mean1_dt = self.calculate_mean1_dt()  # 각 점에 연결된 엣지들의 평균 길이
        
        # Phase 1: 긴 엣지 제거
        filtered_edges_phase1 = self.remove_long_edges(lengths, global_mean, global_variation, mean1_dt)
        print(f"Phase 1에서 남은 엣지 개수: {len(filtered_edges_phase1)}")
        
        # Phase 2: 로컬 긴 엣지 제거
        filtered_edges_phase2 = self.remove_local_long_edges(filtered_edges_phase1, lengths)
        print(f"Phase 2에서 남은 엣지 개수: {len(filtered_edges_phase2)}")
        
        # Phase 3: 로컬 링크 엣지 제거
        filtered_edges_phase3 = self.remove_local_link_edges(filtered_edges_phase2, lengths)
        print(f"Phase 3에서 남은 엣지 개수: {len(filtered_edges_phase3)}")

        # Phase 4: Facilitators에 의한 클러스터 병합 및 장애물과 교차하는 엣지 제거
        filtered_edges_phase3 = self.merge_clusters_with_facilitators(filtered_edges_phase3)
        filtered_edges_phase4 = self.remove_obstacle_crossing_edges(filtered_edges_phase3, self.obstacle_map)
        print(f"Phase 4에서 남은 엣지 개수: {len(filtered_edges_phase4)}")
        
        # 결과 시각화 (장애물 포함)
        self.plot_edges_with_obstacles(filtered_edges_phase4, "Final Edges After Phase 4", self.obstacle_map,self.facilitators)


    ################### visualize ##################
    def plot_edges_with_obstacles(self, edges, title, obstacle_map, facilitators):
        """ 엣지, 장애물, facilitators를 함께 시각화하는 함수 """
        plt.figure()

        # 1. 그래프를 구성하여 연결된 그룹 찾기
        G = nx.Graph()
        for edge in edges:
            G.add_edge(edge[0], edge[1])

        # 2. 연결 요소(connected components)를 찾음
        connected_components = list(nx.connected_components(G))
        
        # 3. 각 연결 그룹에 대해 색상을 지정하고 엣지 및 포인트 시각화
        for i, component in enumerate(connected_components):
            # 각 연결된 컴포넌트를 같은 색으로 표시
            color = [random.random() for _ in range(3)]  # 같은 그룹에 같은 색상 부여
            component_edges = [edge for edge in edges if edge[0] in component and edge[1] in component]
            
            # 엣지 시각화
            for edge in component_edges:
                p1, p2 = self.points[edge[0]], self.points[edge[1]]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.0, alpha=0.7)

            # 포인트 시각화 (각 그룹의 점들)
            for node in component:
                px, py = self.points[node]
                plt.scatter(px, py, color=color, s=50, zorder=5)  # 포인트 크기와 zorder 설정으로 점을 위로 표시

        # 4. 장애물 시각화 (숫자로 변환하여 처리)
        for obstacle in obstacle_map:
            ox = float(obstacle['x'])
            oy = float(obstacle['y'])
            ow = float(obstacle['width'])
            oh = float(obstacle['height'])
            
            rect = plt.Rectangle((ox, oy), ow, oh, color='gray', fill=True)
            plt.gca().add_patch(rect)

        # 5. facilitators 시각화
        for facilitator in facilitators:
            fx = facilitator['x']
            fy = facilitator['y']
            influence_radius = facilitator['influence']
            
            # facilitators를 원(circle)으로 시각화, 영향 반경을 반영
            circle = plt.Circle((fx, fy), influence_radius, color='green', fill=False, linestyle='--', linewidth=2.0)
            plt.gca().add_patch(circle)

            # facilitator 위치 표시
            plt.scatter(fx, fy, color='green', s=100, marker='x')  # facilitator의 중심을 x로 표시

        plt.title(title)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
