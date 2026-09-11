import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.localization.visualize_map import MapVisualizer
from app.localization.map import OccupancyGrid

def visualize_map():
    map = OccupancyGrid.from_json("config/map/map0.json", default_value=0)

    map_visualizer = MapVisualizer(map)
    map_visualizer.serve_forever()

if __name__ == "__main__":
    visualize_map()