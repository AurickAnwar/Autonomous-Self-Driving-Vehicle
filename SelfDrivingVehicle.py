import sys

sys.path.append(r"C:\Users\auric\Downloads\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl")
import carla
import time
from ultralytics import YOLO
import numpy as np
import cv2


yolo_model = YOLO('yolo11m.pt')
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()


print("Connected to:", world.get_map().name)



for actor in world.get_actors().filter('vehicle.*'):
    actor.destroy()


blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')


camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7), carla.Rotation(pitch=-15))


spawn_points = world.get_map().get_spawn_points()

vehicle = None


for spawn_point in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is not None:
        print("Vehicle spawned!")
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
      
        break

if vehicle is None:
    print("No spawn point worked.")
    exit()


spectator = world.get_spectator()
transform = vehicle.get_transform()


def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    frame = array[:, :, :3]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = yolo_model(frame)
    annotated_frame = results[0].plot()

    color = (0,165,255)

    font = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(annotated_frame, f"Speed: {vehicle.get_velocity().length():.2f} m/s", (10,40), font, 0.8, color, 1, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Acceleration: {vehicle.get_acceleration().length():.2f} m/s^2", (10,120), font, 0.7, color, 1, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Brake: {vehicle.get_control().brake:.2f}", (10,160), font, 0.7, color, 1, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Steer: {vehicle.get_control().steer:.2f}", (10,200), font, 0.7, color, 1, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Throttle:{vehicle.get_control().throttle:.2f}", (10, 240), font, 0.7, color, 1, cv2.LINE_AA)

    cv2.imshow("Camera View", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        sys.exit(0)

 

camera.listen(process_image)

start_time = time.time()

while True:
    elapsed = time.time() - start_time
    
    if elapsed < 60:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
    else:
        
        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))


    transform = vehicle.get_transform()
        
    spectator.set_transform(
        carla.Transform(
            transform.location + carla.Location(x = 1.5 ,y=0.0, z=2.5),
            transform.rotation
        )
    )
    
    time.sleep(0.05)

 







   

time.sleep(20)