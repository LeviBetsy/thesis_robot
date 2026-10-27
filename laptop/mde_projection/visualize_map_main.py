import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.localization.visualize_map import MapVisualizer
from app.localization.map import OccupancyGrid
from app.stream.zmq_stream import ParticleReceiver
from app.util.config import Config

def visualize_map():
    config = Config.load()
    map = OccupancyGrid.from_json("config/map/real_map0.json", config, default_value=0)

    map_visualizer = MapVisualizer(map)
    def call_back(p):
        print(p)
    # particle_receiver = ParticleReceiver(callback=call_back)
    particle_receiver = ParticleReceiver(host=config.stream.pi_host, port=config.stream.particle_port,
                                         callback=map_visualizer.update_particles)

    try:
        map_visualizer.serve_forever()
    finally:
        particle_receiver.stop()

if __name__ == "__main__":
    visualize_map()