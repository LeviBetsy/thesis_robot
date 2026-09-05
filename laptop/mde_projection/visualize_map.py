import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.localization.visualize_map import MapVisualizer
from app.localization.map import OccupancyGrid

def visualize_map():
    map = OccupancyGrid(1.07, 1.78, 0.025, default_value=0)
    map.load_walls_from_json("config/map/map0.json")

    map_visualizer = MapVisualizer(map)
    map_visualizer.serve_forever()

if __name__ == "__main__":
    visualize_map()