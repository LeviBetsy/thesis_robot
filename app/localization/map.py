import numpy as np
import math
import json

'''
Map is an occupancy grid with 0 for free, 1 for occupied.
0,0 is the bottom left corner of the grid itself -- there is no implicit perimeter wall,
so any boundary wall has to be a wall segment in the JSON like any other.
'''
class OccupancyGrid:
    def __init__(self, width, length, cell_size, default_value=0.5):
        self.width = float(width)
        self.length = float(length)
        self.cell_size = float(cell_size)

        # Calculate grid dimensions based on map size and cell resolution
        self.cols = math.ceil(self.width / self.cell_size)
        self.rows = math.ceil(self.length / self.cell_size)

        # Initialize the internal grid. dtype is forced to float64 rather than inferred
        # from default_value: np.full infers int64 when default_value is a plain int (the
        # common call pattern is default_value=0), which would silently truncate every
        # fractional value ever written into the grid -- the 0.5 "unknown" fill, add_wall's
        # margin, anything short of exactly 0 or 1.
        self.data = np.full((self.rows, self.cols), default_value, dtype=np.float64)

    def world_to_grid(self, x, y):
        # round before floor: (0.35) / 0.025 is 13.999999999999998 in floats, which would
        # put an exact cell-boundary coordinate one cell short
        col = np.floor(np.round(x / self.cell_size, 9)).astype(int)
        row = np.floor(np.round(y / self.cell_size, 9)).astype(int)
        return row, col


    def grid_to_world(self, row, col):
        """Returns the physical coordinates (meters) at the center of a grid cell."""
        x = (col * self.cell_size) + (self.cell_size / 2.0)
        y = (row * self.cell_size) + (self.cell_size / 2.0)
        return x, y

    # verified, have not tested
    def is_placeable(self, x, y, occ_threshold=0.5):
        """
        Checks whether world points land on a non-occupied cell inside the map.
        Accepts scalars or arrays; arrays must broadcast against each other.

        Args:
            x, y (np.ndarray | float): world coordinates in meters.
            occ_threshold (float): cells strictly above this count as occupied.
        Returns:
            free (np.ndarray): boolean, same shape as the broadcast inputs. False when
                               the point falls outside the grid or on an occupied cell.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        row, col = self.world_to_grid(x, y)

        inside = (col >= 0) & (col < self.cols) & (row >= 0) & (row < self.rows)
        #clip before indexing so out-of-bounds entries stay valid indices, `inside` masks them off after
        occupied = self.data[np.clip(row, 0, self.rows - 1), np.clip(col, 0, self.cols - 1)] > occ_threshold
        return inside & ~occupied

    def add_wall(self, x1, y1, x2, y2, margin=0.7):
        """
        Marks a one-cell-thick straight wall segment (in meters) as occupied, then raises
        every cell directly adjacent to it (up/down/left/right, not diagonally) to at
        least `margin` -- a one-cell safety buffer so a particle can't sit flush against a
        wall. `margin` only ever raises a cell (max(current, margin)), so it never
        overwrites the wall itself or a bigger margin left by another nearby wall.
        Segment must be axis-aligned (x1==x2 for a vertical wall, or y1==y2 for a horizontal wall).
        """
        if x1 == x2:
            row_start, col = self.world_to_grid(x1, min(y1, y2))
            row_end, _ = self.world_to_grid(x1, max(y1, y2))
            row_lo, row_hi, col_lo, col_hi = row_start, row_end, col, col
        elif y1 == y2:
            row, col_start = self.world_to_grid(min(x1, x2), y1)
            _, col_end = self.world_to_grid(max(x1, x2), y1)
            row_lo, row_hi, col_lo, col_hi = row, row, col_start, col_end
        else:
            raise ValueError("Wall segment must be axis-aligned: x1 == x2 or y1 == y2")

        # Clip to the grid before using these as slice bounds -- a negative row/col (a
        # point before the origin) would otherwise wrap around from the end of the array
        # instead of being dropped, silently corrupting cells on the opposite edge.
        row_lo, row_hi = np.clip([row_lo, row_hi], 0, self.rows - 1)
        col_lo, col_hi = np.clip([col_lo, col_hi], 0, self.cols - 1)

        self.data[row_lo:row_hi + 1, col_lo:col_hi + 1] = 1

        offset = [(0,1), (0,-1), (1,0), (-1,0), (-1,-1), (-1,1), (1,-1), (1,1)]

        #4-connected neighbour of every wall cell, one cell out in each direction. A plain
        #loop is fine here, add_wall only runs a handful of times at map construction.
        for row in range(row_lo, row_hi + 1):
            for col in range(col_lo, col_hi + 1):
                for o_r, o_c in offset:
                    nr, nc = row + o_r, col + o_c
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        self.data[nr, nc] = max(self.data[nr, nc], margin)

    @classmethod
    def from_json(cls, filepath, default_value=0.5):
        """
        Builds a fully populated grid from a single JSON map file (shared by both laptop
        and Pi), rather than constructing the grid and adding walls as separate steps.

        Expected format (width/length are the full grid extent -- include boundary walls
        explicitly in "walls" if you want them occupied, there is no implicit perimeter):
        {
          "width": 1.12,
          "length": 1.83,
          "cell_size": 0.025,
          "walls": [
            {"x1": 0.0, "y1": 0.0, "x2": 1.12, "y2": 0.0},
            {"x1": 0.05, "y1": 0.2, "x2": 0.3, "y2": 0.2},
            {"x1": 0.1, "y1": 0.0, "x2": 0.1, "y2": 0.5}
          ]
        }

        Args:
            filepath (str): path to the map JSON file.
            default_value (float): fill value for cells with no wall, forwarded to __init__.
        Returns:
            grid (OccupancyGrid): a new grid with every wall from the file already added.
        """
        with open(filepath, "r") as f:
            layout = json.load(f)

        grid = cls(layout["width"], layout["length"], layout["cell_size"], default_value)
        for wall in layout.get("walls", []):
            grid.add_wall(wall["x1"], wall["y1"], wall["x2"], wall["y2"])
        return grid
