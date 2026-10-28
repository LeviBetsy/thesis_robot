import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import math
import heapq

from app.localization.map import OccupancyGrid

'''
Path planning on the occupancy grid for autonomous nav.

A free cell is one grid.is_placeable accepts: the wall AND its placeable_radius buffer are
blocked, so a path through free cells keeps the robot's centre far enough from every wall
for its body to fit.
'''

# 8-connected neighbours (d_row, d_col, step cost in cells)
SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def octile(a, b):
    """Cost of the cheapest 8-connected path from cell a to cell b on an empty grid:
    go diagonal until lined up with b, then straight the rest of the way."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    diagonals = min(dr, dc)
    straights = max(dr, dc) - diagonals
    return SQRT2 * diagonals + straights


def free_mask(grid: OccupancyGrid) -> np.ndarray:
    """Boolean (rows, cols), True where grid.is_placeable accepts the cell centre."""
    rows, cols = np.mgrid[0:grid.rows, 0:grid.cols]
    x, y = grid.grid_to_world(rows, cols)
    return grid.is_placeable(x, y)


def nearest_free(free: np.ndarray, row, col):
    """The free cell closest to (row, col), so a mean pose that drifted into a wall buffer
    still has a start to plan from. None if the map has no free cell at all."""
    rows, cols = np.nonzero(free)
    if rows.shape[0] == 0:
        return None
    i = np.argmin((rows - row) ** 2 + (cols - col) ** 2)
    return int(rows[i]), int(cols[i])


def astar(grid: OccupancyGrid, start_xy, goal_xy):
    """
    Shortest 8-connected path over placeable cells.

    Args:
        grid (OccupancyGrid): the map.
        start_xy (tuple): (x, y) meters, usually the MCL mean pose. Snapped to the nearest
                          placeable cell if it is not placeable itself.
        goal_xy (tuple): (x, y) meters. Must be placeable.
    Returns:
        path (list): world (x, y) cell centres from start to goal, or None if unreachable.
    """
    free = free_mask(grid)
    in_grid = lambda r, c: 0 <= r < grid.rows and 0 <= c < grid.cols

    start = tuple(int(v) for v in grid.world_to_grid(*start_xy))
    goal = tuple(int(v) for v in grid.world_to_grid(*goal_xy))

    if not (in_grid(*start) and free[start]):
        start = nearest_free(free, *start)
    if start is None or not (in_grid(*goal) and free[goal]):
        return None

    g = {start: 0.0} #cheapest known cost from start to each cell
    parent = {start: None}
    open_heap = [(octile(start, goal), start)] #(g + heuristic, cell)
    closed = set()

    while open_heap:
        _, cell = heapq.heappop(open_heap)
        if cell in closed:
            continue #stale entry, this cell was already expanded through a cheaper route
        if cell == goal:
            path = []
            while cell is not None:
                path.append(grid.grid_to_world(*cell))
                cell = parent[cell]
            return path[::-1]
        closed.add(cell)

        r, c = cell
        for dr, dc, cost in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not in_grid(nr, nc) or not free[nr, nc]:
                continue
            # no diagonal squeeze between two blocked cells that only touch at a corner
            if dr != 0 and dc != 0 and not (free[r + dr, c] and free[r, c + dc]):
                continue
            ng = g[cell] + cost
            if ng < g.get((nr, nc), math.inf):
                g[(nr, nc)] = ng
                parent[(nr, nc)] = cell
                heapq.heappush(open_heap, (ng + octile((nr, nc), goal), (nr, nc)))

    return None


def line_of_sight(grid: OccupancyGrid, a, b) -> bool:
    """
    True if every point on the straight segment a -> b is placeable, i.e. the robot can
    drive it in a straight line without its body touching a wall.

    Marches at cell_size/3, same as RayCaster.cast, so a one-cell-thick blocked cell cannot
    be stepped over even when crossed diagonally.
    """
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    n_steps = max(int(math.ceil(length / (grid.cell_size / 3.0))), 1)
    t = np.linspace(0.0, 1.0, n_steps + 1)
    x = a[0] + t * (b[0] - a[0])
    y = a[1] + t * (b[1] - a[1])
    return bool(np.all(grid.is_placeable(x, y)))


def lookahead_point(grid: OccupancyGrid, path, robot_xy, lookahead):
    """
    The point to aim at: walk the path forward from the robot and keep the last cell that
    is within `lookahead` meters ALONG the path and has a clear straight line from the
    robot. Stop at the first cell that fails either, so the aim never skips past a corner.

    Args:
        path (list): world (x, y) from astar, path[0] is the robot's cell (replanned every step).
        robot_xy (tuple): (x, y) meters, the MCL mean pose.
        lookahead (float): meters along the path.
    Returns:
        (x, y): the aim point. path[1] if nothing further passes, it is a free neighbour of
                the robot's cell so driving to it is always safe. path[0] if the path is one cell.
    """
    if len(path) < 2:
        return path[0]

    aim = path[1]
    along = math.hypot(path[1][0] - path[0][0], path[1][1] - path[0][1])
    for i in range(2, len(path)):
        along += math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if along > lookahead or not line_of_sight(grid, robot_xy, path[i]):
            break
        aim = path[i]
    return aim


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from app.robot_module.robot import Robot
    from app.util.config import Config

    config = Config.load()
    robot = Robot(config)
    grid = OccupancyGrid.from_json("config/map/real_map0.json", config)

    # test scenario: manual_drive's START_POSE to a goal behind the wall at y=1.10
    start_xy = (0.395, 0.555)
    goal_xy = (0.2, 1.4)

    path = astar(grid, start_xy, goal_xy)
    if path is None:
        raise SystemExit("No path found")
    aim = lookahead_point(grid, path, start_xy, config.nav.lookahead)
    print(f"Path: {len(path)} cells, first aim point ({aim[0]:.3f}, {aim[1]:.3f})")

    free = free_mask(grid)
    px, py = zip(*path)
    plt.imshow(free, origin="lower", cmap="gray",
               extent=(0, grid.cols * grid.cell_size, 0, grid.rows * grid.cell_size))
    plt.plot(px, py, "c-", label="path")
    plt.plot(*start_xy, "go", label="start")
    plt.plot(*goal_xy, "r*", label="goal")
    plt.plot(*aim, "mx", label="aim")
    plt.legend()
    plt.show()
