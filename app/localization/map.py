import numpy as np
import math
import json

'''
Map is an occupancy grid with 0 for free, 1 for occupied.
0,0 start at the bottom left corner, the outer perimeter occupy cells the first and last rows and columns
'''
class OccupancyGrid:
    def __init__(self, internal_width, internal_length, cell_size, default_value=0.5):
        #total size of map is internal size plus the walls
        self.width = float(internal_width + cell_size*2) 
        self.length = float(internal_length + cell_size*2)
        self.cell_size = float(cell_size)
        
        # Calculate grid dimensions based on map size and cell resolution
        self.cols = math.ceil(self.width / self.cell_size)
        self.rows = math.ceil(self.length / self.cell_size)
        
        # Initialize the internal grid
        self.data = np.full((self.rows, self.cols), default_value)

        # Fill the outer perimeter (walls) with 1
        self.data[0, :] = 1
        self.data[-1, :] = 1
        self.data[:, 0] = 1
        self.data[:, -1] = 1

    def world_to_grid(self, x, y):
        """Converts physical coordinates (meters) to 2D array indices."""
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        return row, col

    def grid_to_world(self, row, col):
        """Returns the physical coordinates (meters) at the center of a grid cell."""
        x = (col * self.cell_size) + (self.cell_size / 2.0)
        y = (row * self.cell_size) + (self.cell_size / 2.0)
        return x, y

    def add_wall(self, x1, y1, x2, y2):
        """
        Marks a one-cell-thick straight wall segment (in meters) as occupied.
        Segment must be axis-aligned (x1==x2 for a vertical wall, or y1==y2 for a horizontal wall).
        """
        if x1 == x2:
            col = int(x1 // self.cell_size)
            row_start, _ = self.world_to_grid(x1, min(y1, y2))
            row_end, _ = self.world_to_grid(x1, max(y1, y2))
            self.data[row_start:row_end + 1, col] = 1
        elif y1 == y2:
            row = int(y1 // self.cell_size)
            _, col_start = self.world_to_grid(min(x1, x2), y1)
            _, col_end = self.world_to_grid(max(x1, x2), y1)
            self.data[row, col_start:col_end + 1] = 1
        else:
            raise ValueError("Wall segment must be axis-aligned: x1 == x2 or y1 == y2")

    def load_walls_from_json(self, filepath):
        """
        Loads wall segments from a shared JSON file (usable by both laptop and Pi).

        Expected format:
        {
          "walls": [
            {"x1": 0.05, "y1": 0.2, "x2": 0.3, "y2": 0.2},
            {"x1": 0.1, "y1": 0.0, "x2": 0.1, "y2": 0.5}
          ]
        }
        """
        with open(filepath, "r") as f:
            layout = json.load(f)

        for wall in layout.get("walls", []):
            self.add_wall(wall["x1"], wall["y1"], wall["x2"], wall["y2"])
