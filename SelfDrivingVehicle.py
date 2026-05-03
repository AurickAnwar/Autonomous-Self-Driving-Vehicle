import sys

sys.path.append(r"C:\Users\auric\Downloads\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl")

import carla
import time
#from ultralytics import YOLO


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()

print("Connected to:", world.get_map().name)



for actor in world.get_actors().filter('vehicle.*'):
    actor.destroy()


blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]


spawn_points = world.get_map().get_spawn_points()

vehicle = None


for spawn_point in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is not None:
        print("Vehicle spawned!")
        break

if vehicle is None:
    print("❌ No spawn point worked.")
    exit()


spectator = world.get_spectator()
transform = vehicle.get_transform()


while True:

    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))


    transform = vehicle.get_transform()
    

    spectator.set_transform(
        carla.Transform(
            transform.location + carla.Location(x = 2.2 ,y=-0.35, z=1.2),
            transform.rotation
        )
    )
    time.sleep(0.05)





time.sleep(20)