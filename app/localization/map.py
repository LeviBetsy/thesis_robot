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
        """
        Converts physical coordinates (meters) to 2D array indices. Accepts scalars or
        arrays; floor_divide so negative coordinates round toward -infinity rather than
        truncating toward zero.
        """
        #add self.cell_size to account for the perimeter wall
        col = np.floor_divide(x + self.cell_size, self.cell_size).astype(int)
        row = np.floor_divide(y + self.cell_size, self.cell_size).astype(int)
        return row, col

    def grid_to_world(self, row, col):
        """Returns the physical coordinates (meters) at the center of a grid cell."""
        # subtract self.cell_size to account for perimeter thickness being added
        x = (col * self.cell_size) + (self.cell_size / 2.0) - self.cell_size
        y = (row * self.cell_size) + (self.cell_size / 2.0) - self.cell_size
        return x, y

    # verified, have not tested
    def is_free(self, x, y, occ_threshold=0.5):
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

    # verified, have not tested
    def ray_cast_batch(self, origins, headings, max_range, occ_threshold=0.5):
        """
        Marches M rays through the grid and returns the distance to the first occupied cell.

        Fixed-step marching rather than a true DDA: every ray takes the same number of
        steps, so the whole fan is one numpy loop instead of M variable-length traversals.
        The step is cell_size/3 so a one-cell-thick wall cannot be stepped over even when
        crossed diagonally; the cost is that ranges are quantized to that step.

        Args:
            origins (np.ndarray): ray start points in meters, shape (M, 2).
            headings (np.ndarray): world-frame ray directions in radians, CCW from +x, shape (M,).
            max_range (float): distance at which an un-hit ray is reported.
            occ_threshold (float): cells strictly above this block a ray. The default lets
                                   the constructor's 0.5 "unknown" fill stay transparent.
        Returns:
            ranges (np.ndarray): shape (M,), float32, distance in meters, max_range if nothing hit.
        """
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 2)
        headings = np.asarray(headings, dtype=np.float64).reshape(-1)

        step = self.cell_size / 3.0
        n_steps = int(math.ceil(max_range / step))

        ox, oy = origins[:, 0], origins[:, 1] #shape (M,)
        dx, dy = np.cos(headings) * step, np.sin(headings) * step #per-step delta, shape (M,)

        ranges = np.full(headings.shape[0], max_range, dtype=np.float32)
        alive = np.ones(headings.shape[0], dtype=bool) #rays that have not hit anything yet

        for i in range(1, n_steps + 1):
            #sample the point i steps along every ray at once
            px = ox + dx * i
            py = oy + dy * i

            row, col = self.world_to_grid(px, py)

            outside = (col < 0) | (col >= self.cols) | (row < 0) | (row >= self.rows)
            blocked = self.data[np.clip(row, 0, self.rows - 1), np.clip(col, 0, self.cols - 1)] > occ_threshold

            #a ray leaving the grid is treated as a hit, the perimeter ring means this is rare
            hit = alive & (outside | blocked)
            ranges[hit] = min(i * step, max_range)
            alive &= ~hit

            if not alive.any():
                break

        return ranges

    def add_wall(self, x1, y1, x2, y2):
        """
        Marks a one-cell-thick straight wall segment (in meters) as occupied.
        Segment must be axis-aligned (x1==x2 for a vertical wall, or y1==y2 for a horizontal wall).
        """
        if x1 == x2:
            row_start, col = self.world_to_grid(x1, min(y1, y2))
            row_end, _ = self.world_to_grid(x1, max(y1, y2))
            self.data[row_start:row_end + 1, col] = 1
        elif y1 == y2:
            row, col_start = self.world_to_grid(min(x1, x2), y1)
            _, col_end = self.world_to_grid(max(x1, x2), y1)
            self.data[row, col_start:col_end + 1] = 1
        else:
            raise ValueError("Wall segment must be axis-aligned: x1 == x2 or y1 == y2")

    @classmethod
    def from_json(cls, filepath, default_value=0.5):
        """
        Builds a fully populated grid from a single JSON map file (shared by both laptop
        and Pi), rather than constructing the grid and adding walls as separate steps.

        Expected format:
        {
          "internal_width": 1.07,
          "internal_length": 1.78,
          "cell_size": 0.025,
          "walls": [
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

        grid = cls(layout["internal_width"], layout["internal_length"], layout["cell_size"], default_value)
        for wall in layout.get("walls", []):
            grid.add_wall(wall["x1"], wall["y1"], wall["x2"], wall["y2"])
        return grid
