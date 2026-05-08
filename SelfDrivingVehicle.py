import sys
sys.path.append(r"C:\Users\auric\Downloads\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl")
import carla
import time
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import os
import pandas as pd
import torch.nn as nn

### Data Collection and Model Training ###
df = pd.read_csv('car_control_data.csv')

x = df[["speed", "acceleration", "steer", "throttle", "brake"]].values
y = df["direction"].values



x_tensor = torch.tensor(x, dtype=torch.float32) # Convert input features to float32 for the model
y_tensor = torch.tensor(y, dtype=torch.long) # Convert target labels to long (integer) type for classification

pytorch_model = nn.Sequential(
    nn.Linear(5, 16), #5 inputs (speed, acceleration, steer, throttle, brake)
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 5)

)



loss = nn.CrossEntropyLoss()#uses softmax internally, so no need to apply it to the output layer
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)

for epoch in range(1000): # Train for 1000 epochs
    prediction = pytorch_model(x_tensor)
    l = loss(prediction, y_tensor)#calculate loss between predicted and actual labels
    
    optimizer.zero_grad() 
    l.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {l.item():.4f}")

yolo_model = YOLO('yolo11m.pt')
client = carla.Client('localhost', 2000)#connect to the CARLA simulator running on localhost at port 2000
client.set_timeout(10.0)#wait 10 seconds for a response from the server before timing out

world = client.get_world()#get the current simulation world, which contains all the actors, environment, and settings


print("Connected to:", world.get_map().name)



for actor in world.get_actors().filter('vehicle.*'):#find all actors in the world that are vehicles and destroy them to start with a clean slate
    actor.destroy()


blueprint_library = world.get_blueprint_library()#access the blueprint library, which contains different types of actors (vehicles, sensors, pedestrians, etc.) that can be spawned in the world.
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]#picked the tesla model 3 blueprint from the library to spawn as our self-driving vehicle. You can choose other models if you prefer.

camera_bp = blueprint_library.find('sensor.camera.rgb')#attach and RGB camera sensor to the vehicle to capture images from the front view. 
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')


camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7), carla.Rotation(pitch=-15))#Position of the camera on the car


spawn_points = world.get_map().get_spawn_points()#get a list of predefined spawn points in the world where we can place our vehicle.
vehicle = None


for spawn_point in spawn_points:#try to spawn the vehicle at each spawn point until one works (some spawn points may be occupied or invalid)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)#try to spawn the vehicle at the current spawn point. If it returns None, it means the spawn failed (e.g., due to collision with another actor), and we will try the next spawn point.
    if vehicle is not None:
        print("Vehicle spawned!")
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)#spawn the camera and attach it to the vehicle so it moves with the car.
      
        break

if vehicle is None:
    print("No spawn point worked.")
    exit()


spectator = world.get_spectator()#get the spectator actor, which represents the camera view in the simulator. We will use this to follow our vehicle from a third-person perspective.
transform = vehicle.get_transform()#get the current position and orientation of the vehicle to set the spectator's view accordingly.


### Real-time Control and Visualization ###
idle_prob = 0
left_prob = 0   #Initialize probabilities for each direction to 0. These will be updated in real-time based on the model's predictions and displayed on the camera feed.
right_prob = 0
forward_prob = 0
reverse_prob = 0

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8) #Convert the raw image data from the camera sensor into a NumPy array. 
    frame = array[:, :, :3]#Reshape the array to match the image dimensions (height, width, channels) and keep only the RGB channels.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)#Convert the image from RGB to BGR format, which is the default color format used by OpenCV for displaying images. 

    results = yolo_model(frame)#train yolo model on the current camera frame to detect objects.
    annotated_frame = results[0].plot()

    text_color = (0,0,255)  # Red text
    bg_color = (0, 0, 0)  # Black background

    font = cv2.FONT_HERSHEY_COMPLEX
    thickness = 2

    # Helper function to draw text with background
    def draw_text_with_bg(frame, text, pos, font, font_scale, text_color, bg_color, thickness):
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = pos
        cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), bg_color, -1)
        cv2.putText(frame, text, pos, font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Left side - vehicle controls
    draw_text_with_bg(annotated_frame, f"Speed: {vehicle.get_velocity().length():.2f} m/s", (10,40), font, 0.8, text_color, bg_color, thickness)
    draw_text_with_bg(annotated_frame, f"Acceleration: {vehicle.get_acceleration().length():.2f} m/s^2", (10,100), font, 0.7, text_color, bg_color, thickness)
    draw_text_with_bg(annotated_frame, f"Brake: {vehicle.get_control().brake:.2f}", (10,140), font, 0.7, text_color, bg_color, thickness)
    draw_text_with_bg(annotated_frame, f"Steer: {vehicle.get_control().steer:.2f}", (10,180), font, 0.7, text_color, bg_color, thickness)
    draw_text_with_bg(annotated_frame, f"Throttle: {vehicle.get_control().throttle:.2f}", (10, 220), font, 0.7, text_color, bg_color, thickness)

    # Right side - model predictions
    draw_text_with_bg(annotated_frame, f"Idle: {idle_prob:.2f}%", (550, 40), font, 0.7, text_color, bg_color, thickness)
    draw_text_with_bg(annotated_frame, f"Left: {left_prob:.2f}%", (550, 80), font, 0.7, text_color, bg_color, thickness)
    draw_text_with_bg(annotated_frame, f"Right: {right_prob:.2f}%", (550, 120), font, 0.7, text_color, bg_color, thickness)
    draw_text_with_bg(annotated_frame, f"Forward: {forward_prob:.2f}%", (550, 160), font, 0.7, text_color, bg_color, thickness)
    draw_text_with_bg(annotated_frame, f"Reverse: {reverse_prob:.2f}%", (550, 200), font, 0.7, text_color, bg_color, thickness)

    cv2.imshow("Camera View", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        sys.exit(0)

 

camera.listen(process_image)#Starts listening to the camera sensor and calls the process_image function every time a new image is captured.

start_time = time.time()

car_control = []
file_exists = os.path.exists('car_control_data.csv')

while True:
    elapsed = time.time() - start_time


    if elapsed < 15:#just used this for testing and can be adjusted as needed. Vehicle is stationary for the first 15 seconds.
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))

    else:
        vehicle.set_autopilot(True)#set up autopilot with CARLA.
   
    transform = vehicle.get_transform()

    spectator.set_transform(
        carla.Transform(
            transform.location + carla.Location(x=1.5, y=0.0, z=2.5), #position the spectator slightly behind and above the vehicle for a better view
            transform.rotation
        )
    )


    ### gets the main features of the car ###
    speed = vehicle.get_velocity().length()
    acceleration = vehicle.get_acceleration().length()  
    steer = vehicle.get_control().steer
    throttle = vehicle.get_control().throttle
    brake = vehicle.get_control().brake


    # Prepare the input data for the model and make a prediction about the driving direction based on the current state of the vehicle. 
    input_data = torch.tensor([[speed, acceleration, steer, throttle, brake]], dtype=torch.float32)

    with torch.no_grad():
        output = pytorch_model(input_data)
        predicted_direction = torch.softmax(output, dim=1)#use softmax to convert the model's output into probabilities for each direction.

    idle_prob = predicted_direction[0][0].item() * 100
    left_prob = predicted_direction[0][1].item() * 100
    right_prob = predicted_direction[0][2].item() * 100 # extract the probabilities for each direction and convert them to percentages for display on the camera feed.
    forward_prob = predicted_direction[0][3].item() * 100
    reverse_prob = predicted_direction[0][4].item() * 100
    
    # Lower thresholds to capture more steering events
    if steer < -0.1:
        direction = 1 # Left turn
    elif steer > 0.1:
        direction = 2 # Right turn
    elif throttle > 0.1 and brake < 0.05:
        direction = 3 # Forward
    elif vehicle.get_control().reverse == True:
        direction = 4 # Reverse
    else:
        direction = 0 #Not moving

    row = {
        "timestamp": elapsed,
        "speed": speed,
        "acceleration": acceleration, #record the current speed and acceleration of the vehicle, as well as the steering, throttle, and brake values. 
        "throttle": throttle,
        "brake": brake,
        "direction": direction
    }

    car_control.append(row)#append the current control data to a list for later analysis or training.
    pd.DataFrame([row]).to_csv('car_control_data.csv', mode='a', header=not file_exists, index=False)#save the control data to a CSV file. 
    file_exists = True

    time.sleep(0.05)

