import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.localization.visualize_map import MapVisualizer
from app.localization.map import OccupancyGrid
from app.stream.zmq_stream import ParticleReceiver

def visualize_map():
    map = OccupancyGrid.from_json("config/map/map0.json", default_value=0)

    map_visualizer = MapVisualizer(map)
    def call_back(p):
        print(p)
    # particle_receiver = ParticleReceiver(callback=call_back)
    particle_receiver = ParticleReceiver(callback=map_visualizer.update_particles)

    try:
        map_visualizer.serve_forever()
    finally:
        particle_receiver.stop()

if __name__ == "__main__":
    visualize_map()