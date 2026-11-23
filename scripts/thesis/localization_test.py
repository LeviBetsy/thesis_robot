import json
import cv2, numpy as np

MAP_PATH = "./config/map/real_map0.json"
VIDEO_PATH = "./data/localization.mov"
ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE  # use ROTATE_90_CLOCKWISE if upside down
SPACING_S = 1.5  # increase if ghosts overlap
LOG_START_IN_VIDEO_S = -3.0  # video time (s) of first mean_pose.csv row; negative = log started first
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

frames, times, next_t = [], [], 0.0
while True:
    ok, f = cap.read()
    if not ok: break
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # real timestamp, safe for variable fps
    last_f, last_t = f, t
    if t < next_t: continue
    if f.shape[1] > f.shape[0]:  # skip if metadata already made it vertical
        f = cv2.rotate(f, ROTATION)
    frames.append(f); times.append(t)
    next_t += SPACING_S

# Always include the final frame so the rover's stopping point gets a ghost and dot
if last_t > times[-1]:
    if last_f.shape[1] > last_f.shape[0]:
        last_f = cv2.rotate(last_f, ROTATION)
    frames.append(last_f); times.append(last_t)

# Median removes the moving rover, leaving an empty scene
bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
comp = bg.astype(np.float32)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

for f in frames:
    mask = (cv2.absdiff(f, bg).max(axis=2) > 35).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2: continue
    rover = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    a = cv2.GaussianBlur((labels == rover).astype(np.float32), (7, 7), 0)[..., None] * 0.8
    comp = comp * (1 - a) + f.astype(np.float32) * a
comp = comp.astype(np.uint8)

# Board corners in order: bottom-left (map origin), bottom-right, top-right, top-left
with open(MAP_PATH) as map_file:
    board = json.load(map_file)
W, L = board["width"], board["length"]
map_pts = np.float32([[0, 0], [W, 0], [W, L], [0, L]])
img_pts = np.float32([[56, 1050], [660, 1047], [648, 26], [44, 40]])  # pixels in the rotated 680x1080 frame
H, _ = cv2.findHomography(map_pts, img_pts)

log = np.loadtxt("./data/mean_pose.csv", delimiter=",", skiprows=1, usecols=(0, 2, 3, 4))  # t, x, y, theta
pix = cv2.perspectiveTransform(log[:, 1:3].reshape(-1, 1, 2).astype(np.float32), H)
pix = pix.reshape(-1, 2).astype(np.int32)
cv2.polylines(comp, [pix.reshape(-1, 1, 2)], False, (0, 0, 255), 4, cv2.LINE_AA)

# Mark the estimate at each ghost's timestamp
log_t = log[:, 0] - log[0, 0]
for t in times:
    j = min(np.searchsorted(log_t, t - LOG_START_IN_VIDEO_S), len(pix) - 1)
    cv2.circle(comp, tuple(int(v) for v in pix[j]), 9, (0, 255, 255), -1, cv2.LINE_AA)

cv2.imwrite("ghost_composite.png", comp)