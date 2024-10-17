import matplotlib.pyplot as plt
import random
def create_tasks():
    filename = '/home/psi/Desktop/research/mission_planning/clustering/map/kroA100.tsp'

    tasks = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        node_section = False
        
        for line in lines:
            # Start reading the node coordinates when NODE_COORD_SECTION is found
            if line.strip() == 'NODE_COORD_SECTION':
                node_section = True
                continue
            elif line.strip() == 'EOF':
                node_section = False
            
            # If we are in the node section, parse the coordinates
            if node_section:
                parts = line.strip().split()
                node_id = int(parts[0])
                x_coord = float(parts[1])
                y_coord = float(parts[2])
                tasks.append({
                    'id': node_id,
                    'x': x_coord,
                    'y': y_coord
                })
    scaled_tasks = []
    for task in tasks:
        scaled_tasks.append({
            'id': task['id'],
            'x': task['x'] / 80,
            'y': task['y'] / 40
        })
    return scaled_tasks

def recreate_task(tasks,obstacle_map):
    # Check for tasks that intersect with obstacles and regenerate their positions
    for i, task in enumerate(tasks):
        for obstacle in obstacle_map:
            if is_point_in_obstacle(task, obstacle):
                print(f"Task {task['id']} intersects with an obstacle. Regenerating position...")
                tasks[i] = regenerate_task_position(task, obstacle_map)
    return tasks


# Function to check if a point is inside an obstacle
def is_point_in_obstacle(point, obstacle):
    return (obstacle['x'] <= point['x'] <= obstacle['x'] + obstacle['width']) and \
           (obstacle['y'] <= point['y'] <= obstacle['y'] + obstacle['height'])

# Function to regenerate a task position if it intersects with an obstacle
def regenerate_task_position(task, obstacles, seed=42):
    
    random.seed(seed)
    while True:
        # Generate new random coordinates within the map boundary
        new_x = random.uniform(0, 50)
        new_y = random.uniform(0, 50)
        # Check if the new position collides with any obstacle
        collision = False
        for obstacle in obstacles:
            if is_point_in_obstacle({"x": new_x, "y": new_y}, obstacle):
                collision = True
                break
        # If no collision, return the new position
        if not collision:
            return {"x": new_x, "y": new_y, "id": task["id"]}

