import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.localization.map import OccupancyGrid

'''
Visualizes an OccupancyGrid in a browser.

Occupancy is drawn as grayscale: 1 (occupied) is black, 0 (free) is white,
anything in between (0.5 default/unknown) is a proportional gray.
Row 0 of the grid is y = 0, so the image is flipped vertically to put the
origin at the bottom left like the map coordinate frame.

Served over the standard library HTTP server (no Flask/display needed), so it
works the same on the Pi over SSH and on the laptop:
    vis = MapVisualizer(grid)
    vis.start()             # non blocking, open http://<host>:8080
    ...
    vis.update()            # after the grid changes
'''
class MapVisualizer:
    def __init__(self, grid: OccupancyGrid, host="0.0.0.0", port=8001,
                 max_size=900, grid_lines=True, refresh_ms=250, on_pick=None):
        self.grid = grid
        self.on_pick = on_pick            # called with the picked cell centre (x, y) meters, e.g. to send it as a nav goal
        self.host = host
        self.port = port
        self.max_size = max_size          # longest side of the rendered image in pixels
        self.grid_lines = grid_lines
        self.refresh_ms = refresh_ms      # how often the browser re-fetches the image

        self.mutex_lock = threading.Lock()
        self.data = np.array(grid.data, dtype=np.float32)
        self.particles = np.zeros((0, 3), dtype=np.float32)
        self.estimate = None              # (x, y, theta) pose estimate highlighted over the cloud, None until one is streamed
        self.path = np.zeros((0, 2), dtype=np.float32)
        self.aim = None
        self.picked = None                # (row, col) of the cell picked in the browser, None until one is
        self.server = None
        self.server_thread = None

    # ---------- state ----------

    def pick_cell(self, row, col):
        """
        Marks (row, col) as the picked cell if A* could use it as a goal.
        Returns the cell centre (x, y) in meters, or None when the cell is outside the map,
        a wall, or inside a wall's placeable buffer (the picked cell is left unchanged).
        """
        if not (0 <= row < self.grid.rows and 0 <= col < self.grid.cols):
            return None
        if not self.grid.free_mask()[row, col]:
            return None
        with self.mutex_lock:
            self.picked = (row, col)
        x, y = self.grid.grid_to_world(row, col)
        if self.on_pick is not None:
            self.on_pick(x, y)
        return x, y

    def update(self, data=None):
        """Snapshots the latest occupancy values. Pass an array to override the bound grid."""
        source = self.grid.data if data is None else data
        with self.mutex_lock:
            self.data = np.array(source, dtype=np.float32)

    def update_particles(self, particles, estimate=None):
        """
        Snapshots the latest particle poses (x, y, theta) to draw on the map, plus the
        optional pose estimate (x, y, theta) to highlight, e.g. the mean or best particle.
        """
        with self.mutex_lock:
            self.particles = np.asarray(particles, dtype=np.float32).reshape(-1, 3)
            self.estimate = None if estimate is None else np.asarray(estimate, dtype=np.float32).reshape(3)

    def update_nav(self, path, aim):
        """Snapshots the latest A* path (M, 2) and lookahead aim (x, y) to draw on the map."""
        with self.mutex_lock:
            self.path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
            self.aim = np.asarray(aim, dtype=np.float32).reshape(2)

    # ---------- rendering ----------

    def _world_to_pixel(self, x, y, rows, scale):
        """
        Maps a world-frame (x, y) in meters to a pixel in the rendered (flipped, scaled)
        image, matching the row/col convention world_to_grid/grid_to_world use.
        """
        # grid col c spans pixels [c*scale, (c+1)*scale); the image is flipped, so world y=0
        # is the bottom edge of the image (pixel rows*scale), not the top of the bottom row
        cell_size = self.grid.cell_size
        px = (x / cell_size) * scale
        py = (rows - y / cell_size) * scale
        return px, py

    def render(self):
        """Returns the current occupancy grid as a BGR image (occupied = black, free = white)."""
        with self.mutex_lock:
            data = np.clip(self.data, 0.0, 1.0)
            particles = self.particles.copy()
            estimate = None if self.estimate is None else self.estimate.copy()
            path = self.path.copy()
            aim = None if self.aim is None else self.aim.copy()
            picked = self.picked

        gray = ((1.0 - data) * 255).astype(np.uint8)
        gray = np.flipud(gray)  # row 0 is y = 0, images draw top down

        rows, cols = gray.shape
        scale = max(1, int(self.max_size / max(rows, cols)))
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (cols * scale, rows * scale), interpolation=cv2.INTER_NEAREST)

        if self.grid_lines and scale >= 6:
            line = (200, 200, 200)
            for col in range(cols + 1):
                x = min(col * scale, image.shape[1] - 1)
                cv2.line(image, (x, 0), (x, image.shape[0]), line, 1)
            for row in range(rows + 1):
                y = min(row * scale, image.shape[0] - 1)
                cv2.line(image, (0, y), (image.shape[1], y), line, 1)

        if path.size:
            # path entries are cell centres, so a half-scale box around each fills exactly its cell
            green = (0, 180, 0)
            half = scale // 2
            points = [tuple(int(round(v)) for v in self._world_to_pixel(x, y, rows, scale)) for x, y in path]
            for cx, cy in points:
                cv2.rectangle(image, (cx - half, cy - half), (cx + half, cy + half), (170, 230, 170), -1)
            cv2.polylines(image, [np.array(points, dtype=np.int32)], False, green, 2, lineType=cv2.LINE_AA)

        if aim is not None:
            ax, ay = self._world_to_pixel(aim[0], aim[1], rows, scale)
            cv2.circle(image, (int(round(ax)), int(round(ay))), max(4, scale // 2), (255, 0, 0), -1, lineType=cv2.LINE_AA)

        if picked is not None:
            # row 0 is the bottom of the flipped image, so grid row r spans image rows (rows-1-r)*scale onward
            row, col = picked
            top_left = (col * scale, (rows - 1 - row) * scale)
            bottom_right = ((col + 1) * scale - 1, (rows - row) * scale - 1)
            cv2.rectangle(image, top_left, bottom_right, (0, 140, 255), -1)
            cv2.rectangle(image, top_left, bottom_right, (0, 60, 120), 2)

        if particles.size:
            radius = max(3, scale // 2)
            color = (0, 0, 255)
            for x, y, theta in particles:
                cx, cy = self._world_to_pixel(x, y, rows, scale)
                center = (int(round(cx)), int(round(cy)))
                # screen y grows downward while world y grows upward (image is flipped), so
                # the direction line's dy is negated relative to the world-frame heading.
                tip = (int(round(cx + radius * math.cos(theta))),
                       int(round(cy - radius * math.sin(theta))))
                cv2.circle(image, center, radius, color, 1, lineType=cv2.LINE_AA)
                cv2.line(image, center, tip, color, 1, lineType=cv2.LINE_AA)

        if estimate is not None:
            # drawn last so it sits on top of the cloud: filled disc, heading line twice the particle radius
            ex, ey, etheta = estimate
            radius = max(5, scale // 2 + 2)
            cx, cy = self._world_to_pixel(ex, ey, rows, scale)
            center = (int(round(cx)), int(round(cy)))
            tip = (int(round(cx + 2 * radius * math.cos(etheta))),
                   int(round(cy - 2 * radius * math.sin(etheta))))
            cv2.circle(image, center, radius, (255, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(image, center, radius, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.line(image, center, tip, (255, 0, 255), 2, lineType=cv2.LINE_AA)

        return image

    def encode_png(self):
        """Returns the rendered map as PNG bytes."""
        ok, buffer = cv2.imencode(".png", self.render())
        if not ok:
            raise RuntimeError("Failed to encode map as PNG")
        return buffer.tobytes()

    def save(self, filepath):
        """Writes the rendered map to an image file."""
        cv2.imwrite(filepath, self.render())
        return filepath

    def show(self, window_name="Occupancy Grid", wait=1):
        """Local OpenCV window instead of the browser (needs a display)."""
        cv2.imshow(window_name, self.render())
        return cv2.waitKey(wait) & 0xFF

    # ---------- web server ----------

    def _page(self):
        rows, cols = self.grid.data.shape
        free_json = json.dumps(self.grid.free_mask().astype(int).tolist()) # free[row][col], row 0 is y = 0
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Occupancy Grid</title>
<style>
  body {{ background:#222; color:#ddd; font-family:monospace; margin:0;
          display:flex; flex-direction:column; align-items:center; gap:12px; padding:16px; }}
  img {{ image-rendering:pixelated; border:1px solid #555; max-width:100%; }}
  #hover {{ position:fixed; display:none; pointer-events:none; white-space:pre;
            background:rgba(20,20,20,0.9); border:1px solid #666; border-radius:4px;
            padding:4px 8px; font-size:12px; z-index:10; }}
</style>
</head>
<body>
  <div>{cols} x {rows} cells &middot; {self.grid.cell_size} m/cell &middot;
       {self.grid.width:.2f} x {self.grid.length:.2f} m &middot; origin bottom left</div>
  <img id="map" src="/map.png">
  <div id="hover"></div>
  <div id="picked">picked: none</div>
  <script>
    const img = document.getElementById("map");
    const hover = document.getElementById("hover");
    const picked = document.getElementById("picked");
    const ROWS = {rows}, COLS = {cols}, CELL = {self.grid.cell_size};
    const FREE = {free_json};
    setInterval(() => {{ img.src = "/map.png?t=" + Date.now(); }}, {self.refresh_ms});

    // The image is exactly COLS x ROWS cells, possibly CSS-scaled, so use the fraction across it.
    // Image row 0 is the top of the map (highest y), grid row 0 is y = 0.
    function cellAt(e) {{
      const col = Math.floor(e.offsetX / img.clientWidth * COLS);
      const row = ROWS - 1 - Math.floor(e.offsetY / img.clientHeight * ROWS);
      if (row < 0 || row >= ROWS || col < 0 || col >= COLS) return null;
      return {{ row, col, x: (col + 0.5) * CELL, y: (row + 0.5) * CELL, free: FREE[row][col] === 1 }};
    }}

    img.addEventListener("mousemove", (e) => {{
      const c = cellAt(e);
      if (!c) {{ hover.style.display = "none"; return; }}
      hover.textContent = `row ${{c.row}}, col ${{c.col}}\\nx = ${{c.x.toFixed(3)}} m\\ny = ${{c.y.toFixed(3)}} m` +
                          (c.free ? "" : "\\nwall / buffer, not pickable");
      hover.style.display = "block";
      // sit down-right of the cursor, flip to the other side near the window edge so it stays visible
      const pad = 14;
      const left = e.clientX + pad + hover.offsetWidth > window.innerWidth ? e.clientX - pad - hover.offsetWidth : e.clientX + pad;
      const top = e.clientY + pad + hover.offsetHeight > window.innerHeight ? e.clientY - pad - hover.offsetHeight : e.clientY + pad;
      hover.style.left = left + "px";
      hover.style.top = top + "px";
      img.style.cursor = c.free ? "crosshair" : "not-allowed";
    }});
    img.addEventListener("mouseleave", () => {{ hover.style.display = "none"; }});

    img.addEventListener("click", async (e) => {{
      const c = cellAt(e);
      if (!c || !c.free) return;
      const res = await fetch("/pick", {{ method: "POST", headers: {{ "Content-Type": "application/json" }},
                                          body: JSON.stringify({{ row: c.row, col: c.col }}) }});
      const r = await res.json();
      picked.textContent = r.ok ? `picked: (row ${{c.row}}, col ${{c.col}})  x=${{r.x.toFixed(3)}} y=${{r.y.toFixed(3)}} m`
                                : "picked: rejected (not placeable)";
    }});
  </script>
</body>
</html>"""

    def _make_handler(self):
        visualizer = self

        class MapHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._respond("text/html", visualizer._page().encode("utf-8"))
                elif path == "/map.png":
                    self._respond("image/png", visualizer.encode_png())
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path.split("?")[0] != "/pick":
                    self.send_error(404)
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    row, col = int(body["row"]), int(body["col"])
                except (ValueError, KeyError, TypeError):
                    self.send_error(400)
                    return
                xy = visualizer.pick_cell(row, col)
                reply = {"ok": False} if xy is None else {"ok": True, "x": xy[0], "y": xy[1]}
                self._respond("application/json", json.dumps(reply).encode("utf-8"))

            def _respond(self, content_type, body):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # keep the robot console clean

        return MapHandler

    def start(self):
        """Starts the viewer in a background thread and returns its URL."""
        if self.server is not None:
            return self.url()

        self.server = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"Map visualizer running at {self.url()}")
        return self.url()

    def serve_forever(self):
        """Blocking version of start(), for running the visualizer on its own."""
        self.start()
        try:
            self.server_thread.join()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            self.server_thread = None
        cv2.destroyAllWindows()

    def url(self):
        host = "localhost" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}"


if __name__ == "__main__":
    grid = OccupancyGrid(width=1.1, length=0.9, cell_size=0.05, default_value=0)
    grid.add_wall(0.0, 0.0, 1.1, 0.0)   # bottom perimeter, no longer implicit
    grid.add_wall(0.0, 0.0, 0.0, 0.9)   # left perimeter
    grid.add_wall(1.05, 0.0, 1.05, 0.9) # right perimeter
    grid.add_wall(0.0, 0.85, 1.1, 0.85) # top perimeter
    grid.add_wall(0.3, 0.2, 0.3, 0.6)
    grid.add_wall(0.3, 0.6, 0.7, 0.6)

    visualizer = MapVisualizer(grid)
    visualizer.serve_forever()
