import json 
import random
from p5 import *
from vehicle import *
from nn import nll_loss
from other_utils import *
import json

# Load the map data from the json file
with open('maps.json', 'r') as f:
    ummm = json.load(f) 

# Default variables
AI = True
length = 256
frame = 1
population = 1
vehicles = []  # Initialize vehicle variable
can_move = False  # Variable to track if the vehicle can move

data = []

# Initialize images and map dimensions before setup
images = ummm[str(random.randint(0, 20))]  # Select a random map
row = len(images)
col = len(images[0])

def setup():
    global images, row, col, vehicle, road_images, frame
    # Set an initial canvas size
    width = int(length * col)
    height = int(length * row)

    size(width, height)  # Note: size(width, height)

    background(0)

    # Ensure road_images has enough images to match indices in images
    road_images = [load_image(f"roads/{i}.png") for i in range(7)]  # Preload images

def place_image(x, y, img):
    # Display the image at (x, y)
    image(img, x, y, length, length)  # Resize the image to length x length
    noStroke()

total_loss = 0
frame_updated = True
temp = None

# real_data = []

# input_dat = []
# target_dat = []
# xyz = 0

def draw():
    global frame, vehicles, images, total_loss, frame_updated, temp, input_dat, target_dat, xyz

    background(0)

    # if frame % 300 == 0:
    #     xyz+=1
    #     input_dat = input_dat[:-1]
    #     target_dat = target_dat[1:]
    #     paired_data = list(zip(input_dat, target_dat))
    #     dict_to_be_dumped = { xyz : paired_data }
    #     real_data.append(dict_to_be_dumped)
    #     input_dat = []
    #     target_dat = []
    #     change()
    #     print(xyz)

    #     if xyz > 20:
    #         frame -= 1
    #         frame_updated=False
    #         with open('data.json', 'w') as f:
    #             json.dump(real_data, f, indent=4)   

    #         print('FINISHED!')         
        

        # Draw the map grid
    for i in range(row):
        for j in range(col):
            x = j * length
            y = i * length
            place_image(x, y, road_images[images[i][j]])

    if frame_updated:
        frame += 1
        # Update and show the vehicles
        for vehicle in vehicles:

            block_x, block_y = vehicle.get_block(length)
            roads = get_road(block_y, block_x, images[block_y][block_x])
            hitbot = vehicle.hitbox()
            vehicle.update()

            snapshot = [Vector(vehicle.position.x, vehicle.position.y), vehicle.angle]
            vehicle.dead = vehicle.check_dead(images[block_y][block_x], hitbot, roads)

            vehicle.initialize_receptors(vehicle.eyes)
            array_of_tile_sets = vehicle.get_tiles(row, col)
            vehicle.stimulate_receptors(images, array_of_tile_sets)  # Be careful, sometimes this messes everything up.

            if can_move:
                if AI:
                    if frame % 1 == 0: # Maybe limit how many decisions it gets to make.
                        inputs = vehicle.get_inputs()
                        choice = vehicle.get_decision_2(inputs) 
                        #                
                        # label, choice = get_label(vehicle, images, row, col)               

                        # print(f"{inputs.to_array()} : {label.to_array()}") 
                        # print(len(input_dat), len(target_dat))
                        # data.append([inputs.to_array(), label.to_array()])
                        # print(data)
                        
                        if not vehicle.dead:
                            # input_dat.append(inputs.to_array())
                            # target_dat.append(label.to_array()) 

                            if choice == 0:  # type: ignore
                                pass
                            if choice == 1:  # type: ignore
                                vehicle.steer(-0.3)  # Magic numbers: 20-0.153 (0.155)
                            if choice == 2:  # type: ignore
                                vehicle.steer(0.3)

                else:
                    vehicle.maxSpeed = 10
                    if not vehicle.dead:
                        if key_is_pressed:  # type: ignore
                            if key == 'A':  # type: ignore
                                vehicle.steer(-0.125)
                            if key == 'D':  # type: ignore
                                vehicle.steer(0.125)
                            if key == 'W':  # type: ignore
                                vehicle.maxSpeed *= 0.1
                            if key == 'S':  # type: ignore
                                vehicle.maxSpeed *= 0.1
            
                if vehicle.dead:
                    vehicles.remove(vehicle)
        
    # frame_updated = False

    for vehicle in vehicles:
        vehicle.initialize_receptors(vehicle.eyes)
        array_of_tile_sets = vehicle.get_tiles(row, col)
        vehicle.stimulate_receptors(images, array_of_tile_sets) 
        # vehicle.display_receptors()
        vehicle.show()
        

def change():
    global can_move, vehicles, length, images, row, col 
    can_move = True  # Allow the vehicle to move again

    # Generate a new random map and load it into images
    images = ummm[str(random.randint(0, 40))]  # Fetch a new random map
    row = len(images)
    col = len(images[0])

    # Reset vehicles
    vehicles.clear()  # Clear any existing vehicles

    for i in range(population):
        vehicle = Vehicle()
        # vehicle.load_model()
        vehicle.position, vehicle.angle = get_starting_location(images)  # Assuming this function returns a valid location
        vehicles.append(vehicle)

def key_pressed():
    global can_move, vehicles, length, images, row, col, frame_updated, frame

    if AI:
        if key == ' ':  # type: ignore  # If space is pressed 
            # frame = 1
            can_move = True  # Allow the vehicle to move again

            # Generate a new random map and load it into images
            images = ummm[str(random.randint(0,40))]  # Fetch a new random map
            row = len(images)
            col = len(images[0])

            # Reset vehicles
            vehicles.clear()  # Clear any existing vehicles

            for i in range(population):
                vehicle = Vehicle()
                vehicle.load_model()
                vehicle.position, vehicle.angle = get_starting_location(images)  # Assuming this function returns a valid location
                vehicles.append(vehicle)
                
        if key == 'M':  # type: ignore
            vehicles[0].save_model()
            pass

        if key == "1": # type: ignore
            frame_updated = True
              
    else: 
        if key == ' ':  # type: ignore # If space is pressed 
            can_move = True  # Allow the vehicle to move again
            vehicles.clear()  # Clear any existing vehicles

            for i in range(population):
                vehicle = Vehicle()
                vehicle.position, vehicle.angle = get_starting_location(images)
                vehicles.append(vehicle)


if __name__ == '__main__':
    run()



