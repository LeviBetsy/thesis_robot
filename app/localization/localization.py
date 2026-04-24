import numpy as np
import cv2
from flask import Flask, Response
import threading

class Localization:
    def __init__(self, width, length):
        """
        Initializes the localization module.
        width (m): Size of the X-axis in grid cells
        length (n): Size of the Y-axis in grid cells
        """
        self.width = width
        self.length = length
        
        # Robot state variables
        self.robot_angle = 0.0
        self.robot_x = 0
        self.robot_y = 0
        
        # Initialize grid
        self.grid = None
        self.build_occupancy_grid(self.width, self.length)
        self.set_coor_robot()

    def build_occupancy_grid(self, m, n):
        """
        Builds an m x n occupancy grid.
        m = width (columns / X-axis)
        n = length (rows / Y-axis)
        Fills the grid with 255 (Free), borders with 0 (Occupied).
        """
        # NumPy shape is (rows, cols) -> (length, width) -> (n, m)
        self.grid = np.full((n, m), 255, dtype=np.uint8)
        
        # Set borders to 0
        self.grid[0, :] = 0      # Top wall
        self.grid[-1, :] = 0     # Bottom wall
        self.grid[:, 0] = 0      # Left wall
        self.grid[:, -1] = 0     # Right wall

    def set_coor_robot(self, x=None, y=None):
        """
        Sets the robot's current (X, Y) grid coordinate.
        If no arguments are provided, defaults to the exact center of the map.
        """
        if x is None or y is None:
            self.robot_x = self.width // 2
            self.robot_y = self.length // 2
        else:
            x_val = int(x)
            y_val = int(y)
        
            if x_val < 0 or x_val >= self.width or y_val < 0 or y_val >= self.length:
                raise ValueError(f"Coordinate out of bounds: ({x_val}, {y_val})")
            self.robot_x = x_val
            self.robot_y = y_val

    def update_odom_coordinate(self, LCount, RCount):
        """
        Updates the robot's coordinate and angle based on wheel encoder counts.
        """
        pass # Add your odometry math here

    def stream_occupancy_grid(self, host='0.0.0.0', port=5000):
        """
        Starts a blocking Flask web server to stream the live occupancy grid.
        """
        app = Flask(__name__)

        def generate_frames():
            while True:
                # 1. Copy the grid for visualization so we don't modify the actual map data
                vis_grid = self.grid.copy()
                
                # 2. Draw the robot on the visual frame (a dark gray circle)
                # Ensure the robot is within bounds before drawing
                if 0 <= self.robot_x < self.width and 0 <= self.robot_y < self.length:
                    cv2.circle(vis_grid, (self.robot_x, self.robot_y), radius=2, color=50, thickness=-1)
                
                # 3. Scale the grid up for the laptop screen (INTER_NEAREST keeps it sharp)
                vis_image = cv2.resize(vis_grid, (600, 600), interpolation=cv2.INTER_NEAREST)
                
                # 4. Encode and yield the frame
                ret, buffer = cv2.imencode('.jpg', vis_image)
                if not ret:
                    continue
                    
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        @app.route('/')
        def index():
            return '''
            <html>
                <head><title>Localization Grid</title></head>
                <body style="background-color: #222; color: white; text-align: center; font-family: sans-serif;">
                    <h2>Live Robot Localization</h2>
                    <img src="/video_feed" style="border: 2px solid white; max-width: 100%;">
                </body>
            </html>
            '''

        @app.route('/video_feed')
        def video_feed():
            return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        def run_server():
            # use_reloader=False is mandatory when running Flask inside a thread
            app.run(host=host, port=port, threaded=True, use_reloader=False)

        # Start the server
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        print(f"Flask server running in background on port {port}.")

# --- Example Usage ---
if __name__ == '__main__':
    # Initialize the class with your dimensions
    # For a 1067cm x 1778cm room at 5cm resolution:
    # Width = 214 cells, Length = 356 cells
    loc = Localization(width=214, length=356)
    
    # The server is blocking. If you want your odometry loop to run simultaneously,
    # you will need to start 'stream_occupancy_grid' in a separate thread, or run 
    # your odometry updates in a separate thread.
    loc.stream_occupancy_grid()
    while True:
        continue