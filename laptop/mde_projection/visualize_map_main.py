import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.localization.visualize_map import MapVisualizer
from app.localization.map import OccupancyGrid
from app.stream.zmq_stream import ParticleReceiver, NavReceiver, GoalStreamer
from app.util.config import Config

def visualize_map():
    config = Config.load()
    map = OccupancyGrid.from_json("config/map/real_map0.json", config, default_value=0)

    goal_streamer = GoalStreamer(port=config.stream.goal_port) #picked cell -> nav.py's goal on the Pi
    map_visualizer = MapVisualizer(map, on_pick=goal_streamer.send_goal)
    def call_back(p, estimate):
        print(p, estimate)
    # particle_receiver = ParticleReceiver(callback=call_back)
    particle_receiver = ParticleReceiver(host=config.stream.pi_host, port=config.stream.particle_port,
                                         callback=map_visualizer.update_particles)
    nav_receiver = NavReceiver(host=config.stream.pi_host, port=config.stream.nav_port,
                               callback=map_visualizer.update_nav)

    try:
        map_visualizer.serve_forever()
    finally:
        particle_receiver.stop()
        nav_receiver.stop()
        goal_streamer.stop()

if __name__ == "__main__":
    visualize_map()