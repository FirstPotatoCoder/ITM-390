from p5 import *
import random
from matrix import Matrix

scaler = 4
length = 256

roads = {
    0: [],
    1: [
        [Vector(16, 0), Vector(16, 64)],
        [Vector(48, 0), Vector(48, 64)]
    ],
    2: [
        [Vector(0, 16), Vector(64, 16)],
        [Vector(0, 48), Vector(64, 48)]
    ],
    3: [
        [Vector(16, 64), Vector(16, 16), Vector(16, 16), Vector(64, 16)],
        [Vector(64, 48), Vector(51, 48), Vector(48, 51), Vector(48, 64)]
    ],
    4: [
        [Vector(16, 0), Vector(16, 48), Vector(16, 48), Vector(64, 48)],
        [Vector(48, 0), Vector(48, 13), Vector(51, 16), Vector(64, 16)]
    ],
    5: [
        [Vector(48, 0), Vector(48, 48), Vector(48, 48), Vector(0, 48)],
        [Vector(16, 0), Vector(16, 13), Vector(13, 16), Vector(0, 16)]
    ],
    6: [
        [Vector(48, 64), Vector(48, 16), Vector(48, 16), Vector(0, 16)],
        [Vector(16, 64), Vector(16, 51), Vector(13, 48), Vector(0, 48)]        
    ]
}

def get_label(vehicle, images, row, col):

    snapshot = [Vector(vehicle.position.x, vehicle.position.y), vehicle.angle]
    best_score = 0
    choice = None

    for i in range(3):

        vehicle.position = Vector(snapshot[0].x, snapshot[0].y)
        vehicle.angle = snapshot[1]

        if i == 0:
            vehicle.update()

        elif i == 1:
            vehicle.steer(-0.3)
            vehicle.update()                             
            
        else:
            vehicle.steer(0.3)
            vehicle.update()

        block_x, block_y = vehicle.get_block(length)
        roads = get_road(block_y, block_x, images[block_y][block_x])
        hitbot = vehicle.hitbox()

        vehicle.initialize_receptors(vehicle.eyes)
        array_of_tile_sets = vehicle.get_tiles(row, col)
        vehicle.stimulate_receptors(images, array_of_tile_sets)  # Be careful, sometimes this messes everything up.

        current_score = compute_score(vehicle, vehicle.receptors[0], vehicle.receptors[-1], vehicle.receptors[2])

        if current_score > best_score:
            best_score = current_score
            choice = i

    vehicle.position = Vector(snapshot[0].x, snapshot[0].y)
    vehicle.angle = snapshot[1]
    list_label = [0 for _ in range(3)]
    list_label[choice] = 1
    label = Matrix(3, 1)
    label.data = [[x] for x in list_label]

    return label, choice


def compute_score(vehicle, first_receptor, last_receptor, middle_receptor):
    
    L = round(math.sqrt((last_receptor.x - vehicle.position.x) ** 2 + (last_receptor.y - vehicle.position.y) ** 2)) if not isinstance(last_receptor, list) else 0
    R = round(math.sqrt((first_receptor.x - vehicle.position.x) ** 2 + (first_receptor.y - vehicle.position.y) ** 2)) if not isinstance(first_receptor, list) else 0
    M = round(math.sqrt((middle_receptor.x - vehicle.position.x) ** 2 + (middle_receptor.y - vehicle.position.y) ** 2)) if not isinstance(middle_receptor, list) else 0

    center_score = ((L+R) - abs(L-R)) / (L+R)
    score = (center_score) * sqrt(M)

    return score

# def compute_score(vehicle, L, R, M):
#     '''
#     L: distance from the left side of the road
#     R: distance from the right side of the road
#     M: distance detected by the middle receptor
#     '''

#     center_score = ((L+R) - abs(L-R)) / (L+R)
#     score = (center_score) * sqrt(M)

#     return score


def get_starting_location(map_state):

    # Easier to start on straight roads
    straight_roads = [
        (i, j) for i in range(len(map_state)) for j in range(len(map_state[0])) if map_state[i][j] == 1 or map_state[i][j] == 2
    ]
    
    coordinate = random.choice(straight_roads)
    x, y = coordinate

    # Angle changes depending on which type of road you started on.
    angle = random.choice([0, 3.1415]) if map_state[x][y] == 2 else random.choice([-3.1415/2, 3.1415/2])

    # Offset helps to center the car in the road
    offset = 256/2
    x, y = x * 256 + offset, y * 256 + offset

    # (y,x) because that works, and (x,y) does not.
    return Vector(y, x), angle


def get_tile(x, y):
    return int(x // length), int (y // length)

def get_road(row, col, state):
    global scaler
    x_offset = col * length
    y_offset = row * length

    if state == 0:
        pass

    elif state == 1 or state == 2:
        road = roads[state]
        points = [[0 for _ in range(2)] for _ in range(2)]

        for i, segment in enumerate(road):
            for j, point in enumerate(segment):
                points[i][j] = Vector(point.x * scaler + x_offset, point.y * scaler + y_offset)

        return points

    else:
        road = roads[state]
        points = [[0 for _ in range(4)] for _ in range(2)]
            
        for i, segment in enumerate(road):
            for j, point in enumerate(segment):
                points[i][j] = Vector(point.x * scaler + x_offset, point.y * scaler + y_offset)

        return points
    

def intersect_line(p1, p2, p3, p4):
    tolerance = 5
    a1 = p2.y - p1.y
    b1 = p1.x - p2.x
    c1 = a1 * p1.x + b1 * p1.y

    a2 = p4.y - p3.y
    b2 = p3.x - p4.x
    c2 = a2 * p3.x + b2 * p3.y

    det = a1 * b2 - a2 * b1

    if det == 0:
        # print('F')
        return (False, None)

    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    intersection_point = Vector(x, y)

    if (min(p1.x, p2.x) - tolerance <= x <= max(p1.x, p2.x) + tolerance and
        min(p3.x, p4.x) - tolerance <= x <= max(p3.x, p4.x) + tolerance and
        min(p1.y, p2.y) - tolerance <= y <= max(p1.y, p2.y) + tolerance and
        min(p3.y, p4.y) - tolerance <= y <= max(p3.y, p4.y) + tolerance):

        fill(0, 0, 255)
        # ellipse(intersection_point[0], intersection_point[1], 10, 10)
        return (True, intersection_point)

    return (False, None)

import numpy as np
from scipy.optimize import fsolve

def dist(v1, v2):
    return np.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)

def intersect_curve(p1, p2, P0, P1, P2, P3):
    def equations(vars, p1, p2, P0, P1, P2, P3):
        t, u = vars
        line_x = (1 - t) * p1.x + t * p2.x
        line_y = (1 - t) * p1.y + t * p2.y
        
        bezier_x = (1 - u) ** 3 * P0.x + 3 * (1 - u) ** 2 * u * P1.x + 3 * (1 - u) * u ** 2 * P2.x + u ** 3 * P3.x
        bezier_y = (1 - u) ** 3 * P0.y + 3 * (1 - u) ** 2 * u * P1.y + 3 * (1 - u) * u ** 2 * P2.y + u ** 3 * P3.y
        
        return (line_x - bezier_x, line_y - bezier_y)

    # Create an array of initial guesses using fewer points
    t_vals = np.linspace(0, 1, num=2)
    u_vals = np.linspace(0, 1, num=2)
    initial_guesses = np.array(np.meshgrid(t_vals, u_vals)).T.reshape(-1, 2)

    intersections = []
    threshold = 5  # Distance threshold for validation

    for initial_guess in initial_guesses:
        solution = fsolve(equations, initial_guess, args=(p1, p2, P0, P1, P2, P3))
        t, u = solution
        
        # Check if the solution is within bounds
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = (1 - t) * p1.x + t * p2.x
            intersection_y = (1 - t) * p1.y + t * p2.y
            intersection_point = Vector(intersection_x, intersection_y)
            
            bezier_point = (
                (1 - u) ** 3 * P0.x + 3 * (1 - u) ** 2 * u * P1.x + 
                3 * (1 - u) * u ** 2 * P2.x + u ** 3 * P3.x,
                (1 - u) ** 3 * P0.y + 3 * (1 - u) ** 2 * u * P1.y + 
                3 * (1 - u) * u ** 2 * P2.y + u ** 3 * P3.y
            )

            # Validate with a distance check
            if dist(intersection_point, Vector(*bezier_point)) < threshold:
                intersections.append(intersection_point)
                
                # Early exit if you only need one intersection
                return (True, intersection_point)

    # If there are intersections, return the closest one to p1
    if intersections:
        closest_intersection = min(intersections, key=lambda point: dist(point, p1))
        return (True, closest_intersection)
    
    return (False, None)
