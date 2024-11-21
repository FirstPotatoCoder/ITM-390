from p5 import *
from scipy.optimize import fsolve
from matrix import Matrix
from nn import NeuralNetwork
import json
from other_utils import *

length = 256
scaler = length / 64

class Vehicle:
    def __init__(self):
        self.dead = False
        self.eyes = 5
        self.position = Vector(0, 0)
        self.velocity = Vector(0, 0)
        self.angle = 3.1415/2 #1.3  # Keep track of the heading angle
        self.maxforce = 0.1
        self.maxSpeed = 30
        self.receptors = []
        self.brain = NeuralNetwork(self.eyes, 3)
        self.image = load_image(f"car/turbo.png")

    def save_model(self):
        parameters = {
            'weights_1': self.brain.l1.weights.data,
            'weights_2': self.brain.l2.weights.data,
            'biases_1': self.brain.l1.biases.data,
            'biases_2': self.brain.l2.biases.data
        }

        with open('test_model.json', 'w') as f:
            json.dump(parameters, f)
        print("Model saved successfully.")

    def load_model(self):
        try:
            with open('Models/supervised_model.json', 'r') as f:
                parameters = json.load(f)

            # Assuming weights and biases are numpy arrays or similar
            self.brain.l1.weights.data = parameters['weights_1']
            self.brain.l2.weights.data = parameters['weights_2']
            self.brain.l1.biases.data = parameters['biases_1']
            self.brain.l2.biases.data = parameters['biases_2']

            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Model not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON from the model file.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            

    def initialize_receptors(self, amount):
        self.receptors = []
        push()
        
        # Create a vector from position to goal
        len_vector = Vector(cos(self.angle), sin(self.angle))
        len_vector.normalize()
        len_vector = len_vector * 1500
        
        # angle = (angle/180) * 3.14159 # Convert to Radians
        # alpha = -((amount-1) * angle) / 2


        alpha = -(3.14159/2) if amount > 1 else 0 # Starting angle
        angle = 3.14159/(amount - 1) if amount > 1 else 0 # Shifting angle

        translate(self.position.x, self.position.y)
        
        for i in range(amount):
            x = len_vector.x * cos(alpha) + len_vector.y * sin(alpha)
            y = -len_vector.x * sin(alpha) + len_vector.y * cos(alpha)

            points = []

            # stroke(0, 0, 255)
            # line(0, 0, x, y)
            num_points = 30

            for j in range(num_points):
                # Calculate the points along the line
                px = (x / (num_points - 1)) * j
                py = (y / (num_points - 1)) * j

                points.append(Vector(px + self.position.x, py + self.position.y))
                stroke(0, 255, 0)
                # ellipse(px, py, 5, 5)

            self.receptors.append(points)
            alpha += angle
        
        pop()


    def get_tiles(self, row, col):
        tiles = []
        for receptor in self.receptors:
            segments = []
            current_row = -1
            current_col = -1
            for point in receptor:
                new_col, new_row = get_tile(point.x, point.y)

                if new_col > col-1 or new_row > row-1 or new_col < 0 or new_row < 0:
                    break

                if not (current_row == new_row and current_col == new_col):

                    current_row = new_row
                    current_col = new_col
                    segments.append([current_row, current_col])
                

            # this contains AN ARRAY of tiles sets (segments), which contains many tiles, each containing 2 values.
            # [[[]], [[]], [[]]]
            tiles.append(segments)

        return tiles

    def stimulate_receptors(self, images, tiles):

        # test = []
        
        for i, tiles_sets in enumerate(tiles):

            first_point = self.receptors[i][0]
            second_point = self.receptors[i][-1]

            for j, tile in enumerate(tiles_sets):

                break_free = False

                state = images[tile[0]][tile[1]]

                # Empty Road
                if state == 0:
                    continue

                # Straight Road
                elif state == 1 or state == 2:
                    # Error
                    roads = get_road(tile[0], tile[1], state)

                    for road in roads:
                        # line(road[0], road[1])
                        ip = intersect_line(first_point, second_point, road[0], road[1])
                        if ip[0]:
                            # ellipse(ip[1].x, ip[1].y, 10, 10)
                            self.receptors[i] = Vector(ip[1].x, ip[1].y)
                            # test.append([ip[1].x, ip[1].y])
                            break_free = True
                            break

                    if break_free:
                        break
                # Curved Road
                else: 

                    roads = get_road(tile[0], tile[1], state)

                    for road in roads:
                        # bezier(road[0], road[1], road[2], road[3])
                        ip = intersect_curve(first_point, second_point, road[0], road[1], road[2], road[3])
                        if ip[0]:
                            self.receptors[i] = Vector(ip[1].x, ip[1].y)
                            # test.append([ip[1].x, ip[1].y])
                            break_free = True
                            break

                    if break_free:
                        break


    def get_decision(self):
        inputs = Matrix(len(self.receptors), 1)

        for i, receptor in enumerate(self.receptors):
            
            strokeWeight(2)
            stroke(255, 0, 0)

            if i == 0:
                stroke(0, 0, 255)

            if isinstance(receptor, list):
                line(self.position.x, self.position.y, self.position.x, self.position.y)
                inputs.data[i][0] = 0
            else:
                line(self.position.x, self.position.y, receptor.x, receptor.y)
                distance = math.sqrt((receptor.x - self.position.x) ** 2 + (receptor.y - self.position.y) ** 2)
                inputs.data[i][0] = distance

        output, _ = self.brain.forward(inputs)
        decision = output.data.index(max(output.data))
        return inputs, output, decision
    
    def get_inputs(self):
        inputs = Matrix(len(self.receptors), 1)
        for i, receptor in enumerate(self.receptors):
            if isinstance(receptor, list):
                inputs.data[i][0] = 0
            else:
                distance = math.sqrt((receptor.x - self.position.x) ** 2 + (receptor.y - self.position.y) ** 2)
                inputs.data[i][0] = round(distance)
                
        return inputs

    
    def get_decision_2(self, inputs):
        output, _ = self.brain.forward(inputs)
        decision = output.data.index(max(output.data))
        return decision
    
    def display_receptors(self):
        for i, receptor in enumerate(self.receptors):
            
            strokeWeight(2)
            stroke(255, 0, 0)

            # if i == 0:
            #     stroke(0, 0, 255)

            if i == 1 or i == 3:
                continue

            if isinstance(receptor, list):
                line(self.position.x, self.position.y, self.position.x, self.position.y)
            else:
                line(self.position.x, self.position.y, receptor.x, receptor.y)



    def show(self):
        push_matrix()
        translate(self.position.x, self.position.y)
        rotate(self.angle)  # Rotate around the center of the vehicle
        # Draw the image centered
        # image(self.image, -self.image.width / 2, -self.image.height / 2, 60, 30)
        image(self.image, -30, -15, 60, 30)
        pop_matrix()

    def update(self):
        self.velocity += Vector(cos(self.angle), sin(self.angle)) * self.maxSpeed
        self.velocity.limit(self.maxSpeed)
        self.position += self.velocity
        self.velocity *= 0  # Reset velocity

    def steer(self, angle_change):
        self.angle += angle_change


    def hitbox(self):
        half_width = self.image.width / 2
        half_height = self.image.height / 2

        # Create corners as vectors and rotate them
        corners = [
            Vector(-half_width + 30, -half_height),
            Vector(half_width + 30, -half_height),
            Vector(half_width + 30, half_height),
            Vector(-half_width + 30, half_height),
        ] # Hard-coded to shift the hit box to fit the car better

        # Rotate the corners and translate to the vehicle's position
        rotated_corners = []
        for corner in corners:

            rotated_x = corner.x * cos(self.angle) - corner.y * sin(self.angle)
            rotated_y = corner.x * sin(self.angle) + corner.y * cos(self.angle)
            rotated_corners.append(Vector(rotated_x, rotated_y) + self.position)

        # Draw lines connecting the corners to visualize the hitbox
        stroke(255, 0, 0)  # Set line color to red
        stroke_weight(3)
        for i, corner in enumerate(rotated_corners):
            # ellipse(corner.x, corner.y, 5, 5)
            start = rotated_corners[i]
            end = rotated_corners[(i + 1) % len(rotated_corners)]

            # line(start.x, start.y, end.x, end.y)

        return rotated_corners


    def get_block(self, length):
        return int(self.position.x // length), int (self.position.y // length)
    
    def check_dead(self, state, hitbot, roads):
        if state == 0:
            pass
        elif state == 1 or state == 2:
            for i in range(4):
                t = i + 1
                if t > 3:
                    t = 0
                self.dead = self.dead or intersect_line(hitbot[i], hitbot[t], roads[0][0], roads[0][1])[0]
                self.dead = self.dead or intersect_line(hitbot[i], hitbot[t], roads[1][0], roads[1][1])[0]
        else:
            for i in range(4):
                t = i + 1
                if t > 3:
                    t = 0
                self.dead = self.dead or intersect_curve(hitbot[i], hitbot[t], roads[0][0], roads[0][1], roads[0][2], roads[0][3])[0]
                self.dead = self.dead or intersect_curve(hitbot[i], hitbot[t], roads[1][0], roads[1][1], roads[1][2], roads[1][3])[0]

        return self.dead
    





