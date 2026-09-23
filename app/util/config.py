import json
from dataclasses import dataclass
from pathlib import Path

'''
Every tunable in the robot, loaded from a JSON file so the Pi and the laptop can be
pointed at the same numbers without editing code on two machines.

    from app.util.config import Config

    cfg = Config.load()                          # config/robot_config.json
    cfg = Config.load("config/experiment_a.json")
    print(cfg.ray_cast.n_rays, cfg.stream.video_port, cfg.robot.cam_t[1])

Values and nothing else. The JSON must be COMPLETE: there are no defaults here, so a
missing or misspelled key raises TypeError naming the section. Nothing falls back, because
a config missing a tunable would otherwise run with a silent value that appears nowhere in
the file, which is the worst thing to debug.
'''

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "robot_config.json"


@dataclass
class Paths:
    """Data files, all relative to config/ (Camera and FloorScaleCalibration join it on themselves)."""
    camera_calib: str          # npz with camera_matrix/distortion_coefficients/width/height
    z_real: str                # npz with cornersOrg/z_real, the floor scale ground truth


@dataclass
class RobotConfig:
    """Physical robot. Mirrors Robot.__init__ and the wheel constants under it."""
    camera_tilt_deg: float     # camera pitched down from the mounting plate
    cam_t: list                # camera offset in the robot frame, meters (right, forward, up)
    wheel_diameter: float      # meters
    wheel_base: float          # meters between the two wheels
    encoder_slots: int         # tachometer slots per wheel revolution


@dataclass
class PerceptionConfig:
    """Monocular depth estimation and the metric scaling in front of it."""
    mde_encoder: str           # vits / vitb / vitl
    mde_device: str            # "", "cuda", "mps", "cpu" - empty means autodetect
    scale_group_n: int         # FloorScaleCalibration piecewise segments
    ground_z: float            # points below this height are floor and get dropped, meters
    jpeg_quality: int          # VideoStreamer frame compression


@dataclass
class RayCastConfig:
    """
    The ray fan. These MUST be identical on both machines: the laptop bins its point cloud
    with them (MDE_Depth.pcd_to_ray_casting) and the Pi predicts against them (RayCaster),
    so a mismatch biases every weight the filter computes.

    fov_x is not here, it is a camera intrinsic: read it off robot.camera.fov_x.
    """
    n_rays: int
    max_range: float           # meters; empty bins are filled with exactly this
    occ_threshold: float       # grid cells strictly above this stop a ray


@dataclass
class MeasurementModelConfig:
    """BeamSensorModel's mixture. max_range is not here, it comes from ray_cast."""
    sigma_hit: float           # monocular depth error, not lidar error, hence large
    lambda_short: float
    z_hit: float
    z_short: float
    z_max: float               # the no-return spike, high because empty bins read as max_range
    z_rand: float
    alpha: float               # tempering for the 16 correlated beams
    max_eps: float             # a reading this close to max_range counts as a no-return


@dataclass
class MCLConfig:
    """The particle filter itself."""
    n_particles: int
    init_sigma_x: float        # init_particles_gaussian spread, meters
    init_sigma_y: float        # init_particles_gaussian spread, meters
    init_sigma_theta: float    # radians
    init_max_attempts: int     # rejection sampling budget before it gives up
    init_pose: list            # where the robot is placed at startup, (x, y, theta)

    # Thrun's odometry motion model (Probabilistic Robotics, Table 5.6). Each alpha is a
    # STANDARD DEVIATION coefficient, not a variance one: sigma_rot = a1*|d_rot| + a2*d_trans.
    motion_alpha1: float       # rotation noise from rotation, rad per rad
    motion_alpha2: float       # rotation noise from translation, rad per meter
    motion_alpha3: float       # translation noise from translation, meters per meter
    motion_alpha4: float       # translation noise from rotation, meters per rad
    motion_min_trans: float    # below this (meters) the step is a pure rotation and the
                               # travel bearing atan2(dy, dx) is just encoder noise


@dataclass
class StreamConfig:
    """ZMQ wiring. host is what a RECEIVER dials; a streamer always binds to *."""
    laptop_host: str           # laptop, seen from the Pi
    pi_host: str               # Pi, seen from the laptop
    video_port: int
    particle_port: int
    range_port: int
    video_fps: int
    range_fps: int
    particle_fps: int


@dataclass
class UartConfig:
    """Link to the MSP432."""
    port: str
    baudrate: int
    timeout: float


@dataclass
class VisualizerConfig:
    """MapVisualizer's browser view."""
    host: str
    port: int
    max_size: int              # longest side of the rendered image, pixels
    grid_lines: bool
    refresh_ms: int


_SECTIONS = {
    "paths": Paths,
    "robot": RobotConfig,
    "perception": PerceptionConfig,
    "ray_cast": RayCastConfig,
    "measurement_model": MeasurementModelConfig,
    "mcl": MCLConfig,
    "stream": StreamConfig,
    "uart": UartConfig,
    "visualizer": VisualizerConfig,
}


@dataclass
class Config:
    paths: Paths
    robot: RobotConfig
    perception: PerceptionConfig
    ray_cast: RayCastConfig
    measurement_model: MeasurementModelConfig
    mcl: MCLConfig
    stream: StreamConfig
    uart: UartConfig
    visualizer: VisualizerConfig

    @classmethod
    def load(cls, filepath=None):
        """
        Reads a config JSON, config/robot_config.json unless told otherwise.

        Args:
            filepath (str | Path | None): path to the JSON, absolute or relative to the
                                          project root.
        Returns:
            config (Config)
        Raises:
            FileNotFoundError if the file is not there, KeyError if a section is missing,
            TypeError if a section is missing a key or has one that is not recognised.
        """
        path = Path(filepath) if filepath is not None else DEFAULT_CONFIG_FILE
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**{name: section(**data[name]) for name, section in _SECTIONS.items()})


if __name__ == "__main__":
    print(Config.load())
