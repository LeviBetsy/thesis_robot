import cv2
import matplotlib.pyplot as plt
from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parents[2] 
test_dir = project_root / "test"

img_path = test_dir / "cube_60cm.jpg"

# Load your image
img = cv2.imread(img_path)
# Convert BGR (OpenCV default) to RGB (Matplotlib default)
if img is None:
  raise FileNotFoundError("cant find image")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(img_rgb)
plt.axis('on')  # Ensure the pixel coordinate rulers are visible
plt.show()