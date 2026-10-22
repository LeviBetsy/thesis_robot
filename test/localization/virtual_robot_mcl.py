import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import math
import threading
import time
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from app.localization.map import OccupancyGrid
from app.localization.mcl import MCLLocalization
from app.robot_module.odometry import Odometry
from app.robot_module.robot import Robot
from app.util.config import Config

# Sibling script, importable because running this file puts its own directory on sys.path.
# Not imported as test.localization.*: the stdlib ships a `test` package that shadows it.
from measurement_model_test import ParticleWeightVisualizer

'''
Virtual rover on the known map, driven by the arrow keys in the browser, with an MCL
particle cloud tracking it. Stands in for the real robot so the filter can be exercised
with no Pi, no MSP432 and no camera.

At startup a Gaussian cloud is scattered around START_POSE and scored against a scan cast
from the rover's true pose. Every SPACE press then runs one MCLLocalization.on_scan:
    motion        the rover's motion since the last press, via its Odometry
    measurement   z is ray cast from the rover's TRUE pose at the press, no noise added
    resample      SKIPPED by default (do_resample=False), so the same particles are kept
                  and their weights stay on screen. measurement_update overwrites the
                  weights, so what is drawn is THIS scan's likelihood, not the product
                  over presses. Pass --resample to run the real filter (resample after
                  every scan) instead.

Drawn:
    blue -> red circles      particles, red and larger = higher weight (log-space ramp, see
                             measurement_model_test.color_strength)
    black ring               the highest weight particle
    filled green dot + line  the true pose the last scan was cast from (the last SPACE)
    faint green rays         that scan
    green ring + line        the rover's LIVE true pose and its heading
    small dark dot           where the camera sits on it (robot.cam_t), since that, not the
                             rover's centre, is where the scan is cast from

Everything except the test scenario itself comes from config/robot_config.json via a real
Robot, same as measurement_model_test.py.

Run from anywhere:
    python test/localization/virtual_robot_mcl.py            # serve on http://localhost:8001
    python test/localization/virtual_robot_mcl.py --save out.png   # the initial scored cloud, no driving
    python test/localization/virtual_robot_mcl.py --speed 0.3 --turn-rate 120
    python test/localization/virtual_robot_mcl.py --n-particles 30 --sigma-x 0.22 --sigma-y 0.22 --sigma-theta 45

Driving: hold the ARROW KEYS on the map page. Up/down drive, left/right turn, and holding
up plus a turn arcs. SPACE runs one filter step. The page has to have focus, so click the
image once after it loads. The browser polls the image every config.visualizer.refresh_ms
(250 ms by default), so the motion looks steppy even though it is integrated at TICK_HZ -
pass a smaller --refresh-ms if that bothers you.
'''

CONFIG_PATH = "./config/robot_config.json"
MAP_PATH = os.path.join(os.path.dirname(__file__), "../../config/map/map2.json")

# The test scenario, not tunables: meters / radians CCW from +x. Same starting pose as
# measurement_model_test.py, facing the y=0.73 wall of map_wall. config.mcl.init_pose is not
# used because it is a placeholder at the map origin, which is inside the perimeter wall.
START_POSE = (0.55, 0.7, math.pi / 2)

DRIVE_SPEED = 0.3   # meters/second at full throttle, roughly what the real rover does
TURN_RATE = 90.0     # degrees/second on the spot
TICK_HZ = 30.0       # how often the pose is integrated, independent of the browser refresh

TRUE_COLOR = (0, 160, 0)
CAMERA_COLOR = (40, 40, 40)
ROBOT_RADIUS = 0.035  # meters, drawn size only, the rover is a point as far as the map cares
HEADING_LEN = 0.08    # meters, length of the heading stub


class VirtualRobot:
    """
    Ground-truth pose that moves when you hold an arrow key.

    Integrates (forward, turn) commands with the same midpoint composition WheelOdometry
    uses on real tachometer packets, so the increment it accumulates is the one the filter
    would eventually be handed. Refuses to drive into a wall: a step landing on an occupied
    or off-map cell is dropped, the rover just stops there.
    """

    def __init__(self, robot: Robot, grid: OccupancyGrid, pose, speed, turn_rate):
        self.robot = robot
        self.grid = grid
        self.speed = speed
        self.turn_rate = turn_rate

        self.x, self.y, self.theta = pose

        # theta is wrapped for drawing, so the heading change between two marks is read off
        # this unwrapped total instead: a full spin between presses would otherwise vanish.
        self.turned = 0.0

        # Pose at the last SPACE press, plus the turned total at that moment. mark() hands
        # the motion since then to self.odometry and moves this up to the current pose.
        self.position_0 = (self.x, self.y, self.theta, self.turned)
        self.odometry = Odometry()

        self.forward = 0.0   # -1, 0 or +1, set by the browser
        self.turn = 0.0      # -1 (right), 0 or +1 (left)
        self.blocked = False

        # The HTTP handler thread writes the command while the tick loop reads it and
        # writes the pose, so both live under one lock.
        self.lock = threading.Lock()

    def set_command(self, forward, turn):
        with self.lock:
            self.forward = float(forward)
            self.turn = float(turn)

    def pose(self) -> tuple:
        with self.lock:
            return self.x, self.y, self.theta

    def camera_xy(self) -> tuple:
        """Where the camera sits in world coordinates. cam_t is (right, forward, up) in the
        robot frame, so x is the rover's right and y is along its heading."""
        x, y, theta = self.pose()
        right, forward, _ = self.robot.cam_t
        return (x + forward * math.cos(theta) + right * math.sin(theta),
                y + forward * math.sin(theta) - right * math.cos(theta))

    def step(self, dt):
        """Advances the pose by one tick of whatever command is currently held."""
        with self.lock:
            d = self.forward * self.speed * dt
            delta_theta = self.turn * self.turn_rate * dt
            if d == 0.0 and delta_theta == 0.0:
                self.blocked = False
                return

            # Midpoint rule, as in Odometry.accumulate: over this tick the rover sweeps
            # delta_theta, so the step it takes points half way through that sweep.
            heading = self.theta + delta_theta / 2
            x = self.x + d * math.cos(heading)
            y = self.y + d * math.sin(heading)

            # A rover that drives through a wall is not a useful stand-in for one that
            # cannot. Turning on the spot is always allowed, only the translation is vetoed.
            if bool(self.grid.is_placeable(x, y)):
                self.x, self.y = x, y
                self.blocked = False
            else:
                self.blocked = d != 0.0

            self.theta = (self.theta + delta_theta) % (2 * math.pi)
            self.turned += delta_theta

    def mark(self) -> tuple:
        """
        Feeds the motion from position_0 to the current pose into self.odometry, then makes
        the current pose the new position_0.

        Whatever path was driven between presses is replaced by Thrun's rot1 / trans / rot2
        decomposition: turn to face the end point, drive straight to it, turn to the final
        heading. That lands on exactly the same relative pose, and handing it to
        accumulate() as three pure arcs (a pure turn or a pure straight has no midpoint
        error) keeps the result exact. A single accumulate(d, delta_theta) could not: it
        assumes one constant-curvature arc, which drive-turn-drive is not.

        Returns:
            (trans, rot1, rot2): meters and radians of the motion just accumulated.
        """
        with self.lock:
            x0, y0, theta0, turned0 = self.position_0
            wx, wy = self.x - x0, self.y - y0
            trans = math.hypot(wx, wy)
            delta_theta = self.turned - turned0  # unwrapped, same as Odometry.dtheta

            # Direction of travel in position_0's body frame. Undefined for a turn on the
            # spot, where all the rotation goes into rot2.
            rot1 = _wrap(math.atan2(wy, wx) - theta0) if trans > 1e-9 else 0.0
            # Reversing shows up as rot1 near +-180 deg; drive backwards instead of turning
            # around, turning back and driving forwards. Same end pose, smaller rotations.
            if abs(rot1) > math.pi / 2:
                rot1 = _wrap(rot1 - math.pi)
                trans = -trans
            rot2 = delta_theta - rot1

            self.odometry.accumulate(0.0, rot1)
            self.odometry.accumulate(trans, 0.0)
            self.odometry.accumulate(0.0, rot2)

            self.position_0 = (self.x, self.y, self.theta, self.turned)
            return trans, rot1, rot2


def _wrap(angle):
    """Wraps an angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class VirtualRobotVisualizer(ParticleWeightVisualizer):
    """
    ParticleWeightVisualizer (cloud, weights, last scan) with the live rover drawn on top,
    taking arrow keys from the page to drive it and SPACE to run a filter step.
    """

    def bind(self, robot: VirtualRobot, mcl: MCLLocalization, do_resample=False):
        self.robot = robot
        self.mcl = mcl
        self.n_scans = 0
        # Whether filter_step runs the real resample stage, see filter_step's docstring.
        self.do_resample = do_resample
        # ThreadingHTTPServer gives each request its own thread, so two quick SPACE presses
        # could otherwise interleave their mark() and on_scan().
        self.scan_lock = threading.Lock()

    def filter_step(self):
        """
        One SPACE press: hands the motion since the last press to the rover's Odometry,
        then runs mcl.on_scan on a noise-free scan cast from the rover's true pose.

        do_resample (set via bind(), --resample on the command line) controls the last
        filter stage: False (default) skips it so the same particles are kept and their
        weights stay on screen (measurement_update overwrites the weights, so what is
        drawn is THIS scan's likelihood, not the product over presses); True runs the
        real filter, resampling after every scan.
        """
        with self.scan_lock:
            trans, rot1, rot2 = self.robot.mark()
            dx, dy, dtheta = self.robot.odometry.peek()  # what on_scan is about to consume
            scan_pose = np.array([self.robot.position_0[:3]])

            z = self.mcl.caster.predict(scan_pose)[0].astype(np.float64)
            self.mcl.on_scan(z, do_resample=self.do_resample)

            origins, headings = self.mcl.caster.rays(scan_pose)
            hits = origins[0] + z[:, None] * np.stack([np.cos(headings[0]), np.sin(headings[0])], axis=-1)
            missed = z >= self.mcl.caster.max_range - 1e-6
            poses, weights = self.mcl.particle_set.snapshot()
            self.set_cloud(poses, weights, scan_pose[0], origins[0], hits, missed)

            tx, ty, ttheta = scan_pose[0]
            mx, my, mtheta = self.mcl.estimate_mean_pose()
            bx, by, btheta = self.mcl.estimate_best_particle()
            print(f"\nscan {self.n_scans}: true x={tx:.3f} y={ty:.3f} theta={math.degrees(ttheta):.1f}deg")
            print(f"  motion   rot1={math.degrees(rot1):+.1f}deg trans={trans:+.3f}m "
                  f"rot2={math.degrees(rot2):+.1f}deg | odometry dx={dx:+.3f}m "
                  f"dy={dy:+.3f}m dtheta={math.degrees(dtheta):+.1f}deg")
            print(f"  ESS      {self.mcl.effective_sample_size():.2f} / {weights.shape[0]}")
            print(f"  mean     x={mx:.3f} y={my:.3f} theta={math.degrees(mtheta):.1f}deg "
                  f"err_xy={math.hypot(mx - tx, my - ty):.3f}m")
            print(f"  best     x={bx:.3f} y={by:.3f} theta={math.degrees(btheta):.1f}deg "
                  f"err_xy={math.hypot(bx - tx, by - ty):.3f}m")
            self.n_scans += 1

    def render(self):
        image = super().render()
        if not hasattr(self, "robot"):
            return image

        rows = self.grid.data.shape[0]
        scale = max(1, int(self.max_size / max(self.grid.data.shape)))
        to_px = lambda x, y: tuple(int(round(v)) for v in self._world_to_pixel(x, y, rows, scale))

        x, y, theta = self.robot.pose()
        centre = to_px(x, y)
        radius = max(4, int(round(ROBOT_RADIUS / self.grid.cell_size * scale)))

        cv2.circle(image, centre, radius, TRUE_COLOR, 2, lineType=cv2.LINE_AA)
        cv2.line(image, centre,
                 to_px(x + HEADING_LEN * math.cos(theta), y + HEADING_LEN * math.sin(theta)),
                 TRUE_COLOR, 2, lineType=cv2.LINE_AA)
        cv2.circle(image, to_px(*self.robot.camera_xy()), 3, CAMERA_COLOR, -1, lineType=cv2.LINE_AA)
        return image

    # ---------- driving ----------

    def _make_handler(self):
        """Adds /drive and /mark routes to the parent's handler, leaving / and /map.png alone."""
        base = super()._make_handler()
        visualizer = self

        class DriveHandler(base):
            def do_GET(self):
                if urlparse(self.path).path == "/drive":
                    query = parse_qs(urlparse(self.path).query)
                    forward = float(query.get("f", ["0"])[0])
                    turn = float(query.get("t", ["0"])[0])
                    visualizer.robot.set_command(forward, turn)
                    self._respond("text/plain", b"ok")
                elif urlparse(self.path).path == "/mark":
                    visualizer.filter_step()
                    self._respond("text/plain", b"ok")
                else:
                    super().do_GET()

        return DriveHandler

    def _page(self):
        rows, cols = self.grid.data.shape
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Virtual Rover</title>
<style>
  body {{ background:#222; color:#ddd; font-family:monospace; margin:0;
          display:flex; flex-direction:column; align-items:center; gap:12px; padding:16px; }}
  img {{ image-rendering:pixelated; border:1px solid #555; max-width:100%; }}
  #hint {{ color:#888; }}
</style>
</head>
<body>
  <div>{cols} x {rows} cells &middot; {self.grid.cell_size} m/cell &middot;
       {self.grid.width:.2f} x {self.grid.length:.2f} m &middot; origin bottom left</div>
  <img id="map" src="/map.png">
  <div id="hint">click here, then hold the ARROW KEYS to drive, SPACE to run one filter step</div>
  <script>
    const img = document.getElementById("map");
    setInterval(() => {{ img.src = "/map.png?t=" + Date.now(); }}, {self.refresh_ms});

    // Held keys rather than keypress events, so the rover moves for as long as the key is
    // down instead of stuttering at the OS key-repeat rate. Only state CHANGES are sent.
    const held = {{}};
    let last = "";
    function send() {{
      const f = (held.ArrowUp ? 1 : 0) - (held.ArrowDown ? 1 : 0);
      const t = (held.ArrowLeft ? 1 : 0) - (held.ArrowRight ? 1 : 0);
      const command = f + "," + t;
      if (command === last) return;
      last = command;
      fetch("/drive?f=" + f + "&t=" + t);
    }}
    addEventListener("keydown", e => {{
      // SPACE is a one-shot, so the OS key-repeat must not fire it again while held.
      if (e.key === " ") {{
        e.preventDefault();
        if (!e.repeat) fetch("/mark");
        return;
      }}
      if (!e.key.startsWith("Arrow")) return;
      e.preventDefault(); held[e.key] = true; send();
    }});
    addEventListener("keyup", e => {{
      if (!e.key.startsWith("Arrow")) return;
      e.preventDefault(); held[e.key] = false; send();
    }});
    // Let go of everything if the tab loses focus, or the rover drives off on its own.
    addEventListener("blur", () => {{ for (const k in held) held[k] = false; send(); }});
  </script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="write the initial scored cloud to this PNG instead of serving")
    parser.add_argument("--seed", type=int, default=0, help="rng seed for the cloud and the motion noise")
    parser.add_argument("--speed", type=float, default=DRIVE_SPEED, help="drive speed, m/s")
    parser.add_argument("--turn-rate", type=float, default=TURN_RATE, help="turn rate, deg/s")
    parser.add_argument("--refresh-ms", type=int, help="browser refresh, override config.visualizer.refresh_ms")
    parser.add_argument("--n-particles", type=int, help="override config.mcl.n_particles")
    parser.add_argument("--sigma-x", type=float, help="cloud spread in x, meters, override config.mcl.init_sigma_x")
    parser.add_argument("--sigma-y", type=float, help="cloud spread in y, meters, override config.mcl.init_sigma_y")
    parser.add_argument("--sigma-theta", type=float, help="cloud spread in DEGREES, override config.mcl.init_sigma_theta")
    parser.add_argument("--resample", action="store_true",
                        help="resample after every filter step, like the real filter, instead of "
                             "leaving each scan's weights on screen (default: off)")
    args = parser.parse_args()
    np.random.seed(args.seed)

    config = Config.load(CONFIG_PATH)
    robot = Robot(config)  # carries the real camera calibration and offset

    # Overrides are written back into the config so MCLLocalization still reads every number
    # from one place, same as measurement_model_test.py.
    if args.n_particles is not None:
        config.mcl.n_particles = args.n_particles
    sigma_x = args.sigma_x if args.sigma_x is not None else config.mcl.init_sigma_x
    sigma_y = args.sigma_y if args.sigma_y is not None else config.mcl.init_sigma_y
    sigma_theta = math.radians(args.sigma_theta) if args.sigma_theta is not None else config.mcl.init_sigma_theta

    grid = OccupancyGrid.from_json(MAP_PATH, config, default_value=0)
    rover = VirtualRobot(robot, grid, START_POSE, args.speed, math.radians(args.turn_rate))

    # The rover's own Odometry is the filter's motion source, so on_scan consumes exactly
    # what mark() accumulated, and the body frame resets at every press.
    mcl = MCLLocalization(robot, grid, config, odometry=rover.odometry)
    mcl.init_particles_gaussian(*START_POSE, sigma_x=sigma_x, sigma_y=sigma_y,
                                sigma_theta=sigma_theta,
                                max_attempts=config.mcl.init_max_attempts)

    vconfig = config.visualizer
    vis = VirtualRobotVisualizer(grid, host=vconfig.host, port=vconfig.port,
                                 max_size=vconfig.max_size, grid_lines=vconfig.grid_lines,
                                 refresh_ms=args.refresh_ms or vconfig.refresh_ms)
    vis.bind(rover, mcl, do_resample=args.resample)

    print(f"start pose: x={START_POSE[0]:.3f} y={START_POSE[1]:.3f} theta={math.degrees(START_POSE[2]):.1f}deg")
    print(f"map: {grid.width:.2f} x {grid.length:.2f} m, {grid.cell_size} m/cell")
    print(f"camera offset (right, forward, up): {tuple(robot.cam_t)} m")
    print(f"fov_x: {math.degrees(robot.camera.fov_x):.1f}deg, {config.ray_cast.n_rays} rays, "
          f"max_range {config.ray_cast.max_range} m")
    print(f"{config.mcl.n_particles} particles, sigma_x={sigma_x} m, sigma_y={sigma_y} m, "
          f"sigma_theta={math.degrees(sigma_theta):.0f}deg")
    print(f"drive {args.speed} m/s, turn {args.turn_rate} deg/s, integrated at {TICK_HZ:.0f} Hz")
    print(f"resample after each scan: {args.resample}")

    # Scan 0: nothing has moved, so the motion stage is a no-op and this just weights the
    # fresh cloud against the start pose, as measurement_model_test.py does.
    vis.filter_step()

    if args.save:
        print(f"saved {vis.save(args.save)}")
        sys.exit(0)

    vis.start()
    print("arrow keys on the page drive the rover, SPACE runs a filter step, ctrl-c here to stop")

    dt = 1.0 / TICK_HZ
    try:
        while True:
            rover.step(dt)
            time.sleep(dt)
    except KeyboardInterrupt:
        x, y, theta = rover.pose()
        print(f"\nfinal pose: x={x:.3f} y={y:.3f} theta={math.degrees(theta):.1f}deg")
        vis.stop()
