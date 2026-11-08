import cv2
import numpy as np

leg = cv2.imread('report/leg9_l.png')
# Color box column is roughly x in [45, 65]
# Let's crop just the color patch column
patch_col = leg[80:490, 48:62] # (height ~410, width 14)
h, w, _ = patch_col.shape

# Compute row-wise median color
row_colors = np.median(patch_col, axis=1) # (h, 3)

# Find boundaries between patches by color distance
diffs = np.linalg.norm(row_colors[1:] - row_colors[:-1], axis=1)

# Find peaks in diffs to locate transitions
# Let's divide into 24 segments
seg_h = h / 24.0
patches = []
for i in range(24):
    y_start = int(i * seg_h + 3)
    y_end = int((i + 1) * seg_h - 3)
    patch_bgr = np.median(patch_col[y_start:y_end], axis=(0, 1)).astype(int).tolist()
    patches.append((i, patch_bgr))

items = [
  '0.1 - 1 mm', '1.1 - 5 mm', '5.1 - 10 mm', '10.1 - 15 mm', '15.1 - 25 mm',
  '25.1 - 50 mm', '50.1 - 75 mm', '75.1 - 100 mm', '100.1 - 125 mm', '125.1 - 150 mm',
  '150.1 - 200 mm', '200.1 - 250 mm', '250.1 - 275 mm', '275.1 - 300 mm', '300.1 - 350 mm',
  '350.1 - 400 mm', '400.1 - 450 mm', '450.1 - 500 mm', '500.1 - 550 mm', '550.1 - 600 mm',
  '600.1 - 650 mm', '650.1 - 700 mm', '700.1 - 750 mm', '750.1 - 800 mm'
]

# Map BGR to standard color name
def bgr_to_name(bgr):
    b, g, r = bgr
    # standard color naming
    if r > 200 and g > 200 and b > 200:
        return "White"
    if r < 50 and g < 50 and b < 50:
        return "Black"
    if np.std([r, g, b]) < 12:
        return "Grey"
    if r > 180 and g > 180 and b < 120:
        return "Yellow"
    if r > 200 and g > 220 and b > 150:
        return "Light Yellow"
    if r > 200 and g > 120 and b < 80:
        return "Orange"
    if r > 150 and g > 80 and b < 60:
        return "Dark Orange"
    if r > 180 and g < 70 and b < 70:
        return "Red"
    if r > 120 and g < 50 and b < 50:
        return "Dark Red"
    if g > r + 30 and g > b + 30:
        if g > 180 and (r > 120 or b > 120):
            return "Light Green"
        elif g > 140:
            return "Green"
        else:
            return "Dark Green"
    if b > r + 30 and b > g + 30:
        if b > 180 and (r > 120 or g > 120):
            return "Light Blue"
        elif b > 140:
            return "Blue"
        else:
            return "Dark Blue"
    if r > 120 and b > 120 and g < 100:
        return "Purple"
    return "Unknown"

for i, bgr in patches:
    print(f"{i:2d} | BGR={bgr} | {bgr_to_name(bgr):12s} | {items[i]}")

