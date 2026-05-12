import numpy as np
import cv2
from flask import Flask, Response
import threading
import math

from app.uart.uart_exchange import *

class Localization:
    def __init__(self, width, length, cell_size):
        """
        Initializes the localization module.
        width (m): Size of the X-axis in grid cells
        length (n): Size of the Y-axis in grid cells
        """
        self.width = width
        self.length = length
        self.cell_size = cell_size
        
        # Robot state variables
        self.robot_theta = 0 #angle facing in radians
        self.robot_x: float = 0 #robot_x is a float!
        self.robot_y: float = 0 #robot_y is a float!
        self.pose_lock = threading.Lock() #mutex lock so only 1 thread access at once

        #Robot description
        self.n = 360 #number of slots/rotation
        self.d = 70 #70 mm diameter
        self.w = 122 #127mm distance between wheels
        self.c = math.pi*self.d
        
        # Initialize grid
        self.build_occupancy_grid()
        self.set_coor_robot(self.width / 2, self.length / 2, math.pi*3/2) #start the robot in the middle of the grid and ponting south

    def build_occupancy_grid(self):
        """
        Fills the grid with 128 (Uncertain), borders with 0 (Occupied).
        """
        # NumPy shape is (rows, cols) -> (length, width)
        self.grid = np.full((self.length, self.width), 128, dtype=np.uint8) #!!! numpy indexing is [y,x]
        
        # Set borders to 0
        self.grid[0, :] = 0      # Top wall
        self.grid[-1, :] = 0     # Bottom wall
        self.grid[:, 0] = 0      #
        self.grid[:, -1] = 0     #

    
    
    def set_cell(self, x, y, value):
        if value >= 0:
            self.grid[y,x] = value  #!!! numpy indexing is [y,x]

    def set_coor_robot(self, x=None, y=None, theta=None):
        """
        Sets the robot's current (X, Y) grid coordinate.
        If no arguments are provided, defaults to the exact center of the map. Looking down
        """
        with self.pose_lock:
            if x:
                if x < 0 or x >= self.width:
                    raise ValueError(f"Robot\'x out of bounds: x ={x}")
                self.robot_x = x
            if y:
                if y < 0 or y>= self.length:
                    raise ValueError(f"Robot\'y out of bounds: y ={y}")
                self.robot_y = y
            if theta:
                self.robot_theta = theta % (2 * math.pi)
    
    def init_odometry_thread(self, msp432_uart: MSP432Uart): #Start thread to change localization data using UART buffer
        def odom_loop():
            while True:
                # Retrieve the latest counts from the UART interface
                # Note: parentheses added assuming get_data is a method call
                Lcount, Rcount = msp432_uart.get_data()
                
                # Update the internal odometry state
                self.update_odom_coordinate(Lcount, Rcount)
        self.odometry_thread = threading.Thread(target=odom_loop, daemon=True)
        self.odometry_thread.start()


    def update_odom_coordinate(self, LCount, RCount):
        """
        Updates the robot's coordinate and angle based on wheel encoder counts.
        """

        dl = LCount*self.c/self.n
        dr = RCount*self.c/self.n
        d = (dl + dr)/2 #distance traveled by the middle point
        delta_theta = (dr - dl)/self.w
        new_x = self.robot_x + (d*math.cos(self.robot_theta + (delta_theta/2)))/self.cell_size
        new_y = self.robot_y - (d*math.sin(self.robot_theta + (delta_theta/2)))/self.cell_size
        new_theta = self.robot_theta + delta_theta
        self.set_coor_robot(new_x, new_y, new_theta)

    def stream_occupancy_grid(self, host='0.0.0.0', port=5000):
        """
        Starts a blocking Flask web server to stream the live occupancy grid.
        """
        app = Flask(__name__)

        def generate_frames():
            # Define display constants must be a multiple of width and length
            scale_factor = 12
            DISPLAY_WIDTH = self.width  * scale_factor
            DISPLAY_LENGTH = self.length * scale_factor
            
            # Calculate scale factors
            # sx = pixels_per_cell_width, sy = pixels_per_cell_height
            sx = DISPLAY_WIDTH / self.width
            sy = DISPLAY_LENGTH / self.length

            while True:
                # 1. Resize the grid first using NEAREST to keep cell boundaries sharp
                vis_image = cv2.resize(self.grid.copy(), (DISPLAY_WIDTH, DISPLAY_LENGTH), interpolation=cv2.INTER_NEAREST)
                
                # Convert to BGR so we can use colors for the grid/labels
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

                # 2. Draw Grid Lines
                grid_color = (200, 200, 200) # Light gray
                # Vertical lines
                for x in range(self.width + 1):
                    line_x = int(x * sx)
                    cv2.line(vis_image, (line_x, 0), (line_x, DISPLAY_LENGTH), grid_color, 1)
                
                # Horizontal lines
                for y in range(self.length + 1):
                    line_y = int(y * sy)
                    cv2.line(vis_image, (0, line_y), (DISPLAY_WIDTH, line_y), grid_color, 1)

                # Draw Robot on Scaled Image
                if 0 <= self.robot_x < self.width and 0 <= self.robot_y < self.length:
                    # Map robot cell coordinates to pixel centers on the scaled image
                    center_x = int(self.robot_x * sx)
                    center_y = int(self.robot_y * sy)
                    
                    # Robot body
                    cv2.circle(vis_image, (center_x, center_y), radius=int(min(sx, sy) * 0.8), color=(0, 0, 255), thickness=-1)
                    
                    # Heading Arrow
                    arrow_length = 30
                    end_x = int(center_x + arrow_length * math.cos(self.robot_theta))
                    end_y = int(center_y - arrow_length * math.sin(self.robot_theta))
                    
                    cv2.arrowedLine(vis_image, (center_x, center_y), (end_x, end_y), (255, 0, 0), 2, tipLength=0.3)

                # 5. Encode and yield
                ret, buffer = cv2.imencode('.jpg', vis_image)
                if not ret: continue
                
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
    
    def print_grid(self):
        """
        Prints the current grid to the console.
        '#' = Obstacle (value 0)
        '.' = Free space (or unknown/default depending on your init)
        '^', 'v', '<', '>' = Robot position and heading
        """
        with self.pose_lock:
            # Snap robot coordinates to integers for grid display
            rx = int(round(self.robot_x))
            ry = int(round(self.robot_y))

            # Determine robot character based on heading
            # Convert to degrees and normalize to 0-360 for easier bucketing
            theta_deg = math.degrees(self.robot_theta) % 360
            
            # Since your Y grows downward: 
            # 0 = Right, 90 = Down, 180 = Left, 270 = Up
            if 45 <= theta_deg < 135:
                robot_char = '^' 
            elif 135 <= theta_deg < 225:
                robot_char = '<'
            elif 225 <= theta_deg < 315:
                robot_char = 'v'
            else:
                robot_char = '>'

            print("-" * (self.width + 2))
            
            # Print row by row (Y is the vertical axis)
            for y in range(self.length):
                row_str = "|"
                
                # Print column by column (X is the horizontal axis)
                for x in range(self.width):
                    if x == rx and y == ry:
                        row_str += robot_char
                    else:
                        try:
                            val = self.grid[y, x] #!!!!numpy index by [y,x]
                            if val == 0:
                                row_str += "#"
                            else:
                                row_str += "."  # Adjust if you have a specific value for unexplored space
                        except IndexError:
                            # Fallback just in case grid sizes and length/width attributes drift
                            row_str += "?" 
                            
                row_str += "|"
                print(row_str)
                
            print("-" * (self.width + 2))

# --- Example Usage ---
if __name__ == '__main__':
    # Initialize the class with your dimensions
    # For a 106.7cm x 177.8cm room at 5cm resolution:
    # Width = 23, Length = 37
    loc = Localization(width=23, length=37, cell_size=50)
    loc.stream_occupancy_grid()
    while True:
        continue