#Using Inverse Regression dist = slope*1/width + intercept

import numpy as np
import matplotlib.pyplot as plt
import re

raw_data = """
entry_1 pixel_width: 288.6353454589844 dist 100mm 
entry_2 pixel_width: 140.10574340820312 dist 200mm
entry_3 pixel_width: 119.2796630859375 dist 250mm
entry_4 pixel_width: 97.54135131835938 dist 300mm
entry_5 pixel_width: 71.61834716796875 dist 400mm
entry_6 pixel_width: 57.2908935546875 dist 500mm
entry_7 pixel_width: 48.633392333984375 dist 600mm
entry_8 pixel_width: 40.22088623046875 dist 700mm
entry_9 pixel_width: 35.160247802734375 dist 800mm
entry_10 pixel_width: 33.3565673828125 dist 900mm

"""

widths = np.array([float(x) for x in re.findall(r"pixel_width: ([\d.]+)", raw_data)])
distances = np.array([float(x) for x in re.findall(r"dist ([\d.]+)mm", raw_data)])

# 1. Standard Linear Regression
m1, b1 = np.polyfit(widths, distances, 1)

# 2. Inverse Linear Regression (Better for Cameras)
# We regress: y = m * (1/x) + b
m2, b2 = np.polyfit(1/widths, distances, 1)

# Generate lines for plotting
x_range = np.linspace(min(widths), max(widths), 100)
y_linear = m1 * x_range + b1
y_inverse = m2 * (1/x_range) + b2

plt.figure(figsize=(10, 6))
plt.scatter(widths, distances, color='red', label='Measured Data', zorder=3)
plt.plot(x_range, y_linear, '--', color='blue', label='Standard Linear Fit')
plt.plot(x_range, y_inverse, color='green', linewidth=2, label='Inverse Fit (1/x)')

plt.title("Linear vs. Inverse Regression for Distance Estimation")
plt.xlabel("Pixel Width (px)")
plt.ylabel("Distance (mm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Standard Equation: Dist = ({m1:.2f} * W) + {b1:.2f}")
print(f"Inverse Equation:  Dist = ({m2:.2f} / W) + {b2:.2f}")