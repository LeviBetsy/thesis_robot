"""This file is for any functions used to calculate the spatial data of objects in the
environment using vision (and yolo)"""

import math

class ObjectResolver:
    def __init__(self, slope, intercept, robot_radius=75, hfov=95, image_width=640):
        """
        Args:
            slope (float): The 'm' from your inverse regression (Focal Constant).
            intercept (float): The 'b' from your regression (Systematic Offset).
            robot_radius (float): Distance from robot center to camera lens (mm).
            hfov (float): Horizontal Field of View of the camera (degrees).
            image_width (int): Total width of the image in pixels.
        """
        self.m = slope
        self.b = intercept
        self.robot_radius = robot_radius
        self.hfov = hfov
        self.w_img = image_width

    def calculate_distance(self, pixel_width):
        """Calculates distance from robot center to object in mm."""
        if pixel_width <= 0:
            return 0
        
        # Distance from lens (Inverse Model)
        dist_from_lens = (self.m / pixel_width) + self.b
        
        # Add robot radius to get distance from robot center
        return dist_from_lens + self.robot_radius


    #THIS IS GEMINI'S MATH, IF WANT TO CHECK, READ THE BOOK
    def calculate_theta(self, x1, x2):
        """
        Calculates the angle (radians) from the camera's centerline.
        Positive = Right of center, Negative = Left of center.
        """

        obj_center_x = (x1 + x2) / 2
        img_center_x = self.w_img / 2
        
        # How many pixels away from the center is the object?
        pixel_offset = obj_center_x - img_center_x
        
        # Convert pixel offset to degrees
        # (Using standard linear mapping for small angles; 
        # for high accuracy, use: math.degrees(math.atan(pixel_offset / focal_length_px)))
        angle_per_pixel = self.hfov / self.w_img
        theta = math.radians(-pixel_offset * angle_per_pixel)
        return round(theta, 2)
    
    def resolve_coor(self, distance, angle, loc):
        """
        Calculates the global grid coordinate of an object and marks it as an obstacle.
        
        Args:
            distance (float): Distance to the object in mm.
            angle (float): Angle of the object from the centerline in RADIANS (CCW positive).
            loc (app.localization.localization.Localization): Instance of the localization map/grid.
        """
        # Acquire the lock to ensure we read a consistent robot pose 
        # and don't write to the grid while another thread is modifying it
        with loc.pose_lock:
            # 1. Scale distance from mm to grid cells
            distance_grid = distance / loc.cell_size
            
            # 2. Calculate global heading of the object
            global_theta = loc.robot_theta + angle
            
            # 3. Calculate target X and Y
            # We use int(round(...)) to map continuous space to discrete grid integer indices
            target_x = int(round(loc.robot_x + (distance_grid * math.cos(global_theta))))
            target_y = int(round(loc.robot_y - (distance_grid * math.sin(global_theta))))
            
            # 4. Update the map (0 = obstacle)
            loc.set_cell(target_x, target_y, 0)
    