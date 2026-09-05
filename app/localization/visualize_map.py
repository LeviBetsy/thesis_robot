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
    def __init__(self, grid: OccupancyGrid, host="0.0.0.0", port=8080,
                 max_size=900, grid_lines=True, refresh_ms=250):
        self.grid = grid
        self.host = host
        self.port = port
        self.max_size = max_size          # longest side of the rendered image in pixels
        self.grid_lines = grid_lines
        self.refresh_ms = refresh_ms      # how often the browser re-fetches the image

        self.mutex_lock = threading.Lock()
        self.data = np.array(grid.data, dtype=np.float32)
        self.server = None
        self.server_thread = None

    # ---------- state ----------

    def update(self, data=None):
        """Snapshots the latest occupancy values. Pass an array to override the bound grid."""
        source = self.grid.data if data is None else data
        with self.mutex_lock:
            self.data = np.array(source, dtype=np.float32)

    # ---------- rendering ----------

    def render(self):
        """Returns the current occupancy grid as a BGR image (occupied = black, free = white)."""
        with self.mutex_lock:
            data = np.clip(self.data, 0.0, 1.0)

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
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Occupancy Grid</title>
<style>
  body {{ background:#222; color:#ddd; font-family:monospace; margin:0;
          display:flex; flex-direction:column; align-items:center; gap:12px; padding:16px; }}
  img {{ image-rendering:pixelated; border:1px solid #555; max-width:100%; }}
</style>
</head>
<body>
  <div>{cols} x {rows} cells &middot; {self.grid.cell_size} m/cell &middot;
       {self.grid.width:.2f} x {self.grid.length:.2f} m &middot; origin bottom left</div>
  <img id="map" src="/map.png">
  <script>
    const img = document.getElementById("map");
    setInterval(() => {{ img.src = "/map.png?t=" + Date.now(); }}, {self.refresh_ms});
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
    grid = OccupancyGrid(internal_width=1.0, internal_length=0.8, cell_size=0.05, default_value=0)
    grid.add_wall(0.3, 0.2, 0.3, 0.6)
    grid.add_wall(0.3, 0.6, 0.7, 0.6)

    visualizer = MapVisualizer(grid)
    visualizer.serve_forever()
