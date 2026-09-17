import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
import math
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

'''
One place for every tunable in the robot, loaded from a JSON file so the Pi and the laptop
can be pointed at the same numbers without editing code on two machines.

    from app.util.config import Config

    cfg = Config.load()                          # config/robot_config.json
    cfg = Config.load("config/experiment_a.json")

    grid = cfg.load_map()
    mcl  = MCLLocalization(robot, grid, n_particles=cfg.mcl.n_particles,
                           n_rays=cfg.ray_cast.n_rays, max_range=cfg.ray_cast.max_range,
                           sensor_model=cfg.beam_sensor_model())

The JSON is a SPARSE OVERRIDE: anything you leave out keeps the default defined here, so a
file holding only {"mcl": {"n_particles": 30}} is valid. An unknown key is an error rather
than a silent no-op, since a typo'd tunable that quietly does nothing is the worst outcome.

Every default below is the value that was already hardcoded in the module it belongs to, so
loading the shipped config/robot_config.json changes no behaviour.
'''

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "robot_config.json"


@dataclass
class Paths:
    """
    Data files, all relative to config/.

    camera_calib and z_real stay RELATIVE: Camera and FloorScaleCalibration each join the
    name onto config/ themselves, so they want the bare filename. Use the *_path() helpers
    on Config when you need something absolute (OccupancyGrid.from_json does).
    """
    camera_calib: str = "fisheye_calib.npz"          # npz with camera_matrix/distortion_coefficients/width/height
    z_real: str = "z_real_fix_checker_size.npz"      # npz with cornersOrg/z_real, the floor scale ground truth
    map: str = "map/map1.json"                       # occupancy grid layout


@dataclass
class RobotConfig:
    """Physical robot. Mirrors Robot.__init__ and the wheel constants under it."""
    camera_tilt_deg: float = -30.0                   # camera pitched down from the mounting plate
    cam_t: tuple = (0.0, 0.07, 0.11)                 # camera offset in the robot frame, meters (right, forward, up)
    wheel_diameter: float = 0.07                     # meters
    wheel_base: float = 0.122                        # meters between the two wheels
    encoder_slots: int = 360                         # tachometer slots per wheel revolution

    @property
    def cam_forward(self) -> float:
        """cam_t[1], the offset RayCaster needs. Named because the index is easy to misread."""
        return float(self.cam_t[1])

    @property
    def wheel_circumference(self) -> float:
        return math.pi * self.wheel_diameter


@dataclass
class PerceptionConfig:
    """Monocular depth estimation and the metric scaling in front of it."""
    mde_encoder: str = "vits"                        # vits / vitb / vitl
    mde_device: str = ""                             # "", "cuda", "mps", "cpu" - empty means autodetect
    scale_group_n: int = 16                          # FloorScaleCalibration piecewise segments
    ground_z: float = 0.01                           # points below this height are floor and get dropped, meters
    jpeg_quality: int = 90                           # VideoStreamer frame compression


@dataclass
class RayCastConfig:
    """
    The ray fan. These MUST be identical on both machines: the laptop bins its point cloud
    with them (MDE_Depth.pcd_to_ray_casting) and the Pi predicts against them (RayCaster),
    so a mismatch biases every weight the filter computes.
    """
    n_rays: int = 16
    max_range: float = 0.6                           # meters; empty bins are filled with exactly this
    occ_threshold: float = 0.8                       # grid cells strictly above this stop a ray
    fov_x: float = 0.0                               # radians; 0 means "read it off the camera calibration"


@dataclass
class MeasurementModelConfig:
    """BeamSensorModel's mixture. max_range is not here, it comes from ray_cast."""
    sigma_hit: float = 0.08                          # monocular depth error, not lidar error, hence large
    lambda_short: float = 2.0
    z_hit: float = 0.65
    z_short: float = 0.05
    z_max: float = 0.20                              # the no-return spike, high because empty bins read as max_range
    z_rand: float = 0.10
    alpha: float = 0.25                              # tempering for the 16 correlated beams
    max_eps: float = 0.005                           # a reading this close to max_range counts as a no-return


@dataclass
class MCLConfig:
    """The particle filter itself."""
    n_particles: int = 500
    init_sigma_xy: float = 0.05                      # init_particles_gaussian spread, meters
    init_sigma_theta: float = 0.1                    # radians
    init_max_attempts: int = 50                      # rejection sampling budget before it gives up
    init_pose: tuple = (0.0, 0.0, 0.0)               # where the robot is placed at startup, (x, y, theta)
    resample_ess_ratio: float = 0.5                  # resample once ESS drops below this fraction of n_particles


@dataclass
class StreamConfig:
    """ZMQ wiring. host is what a RECEIVER dials; a streamer always binds to *."""
    laptop_host: str = "127.0.0.1"                   # laptop, seen from the Pi
    pi_host: str = "127.0.0.1"                       # Pi, seen from the laptop
    video_port: int = 5003
    particle_port: int = 5004
    range_port: int = 5005
    video_fps: int = 3
    range_fps: int = 3
    particle_fps: int = 5


@dataclass
class UartConfig:
    """Link to the MSP432."""
    port: str = "/dev/ttyAMA0"
    baudrate: int = 115200
    timeout: float = 1.0


@dataclass
class VisualizerConfig:
    """MapVisualizer's browser view."""
    host: str = "0.0.0.0"
    port: int = 8001
    max_size: int = 900                              # longest side of the rendered image, pixels
    grid_lines: bool = True
    refresh_ms: int = 250


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    robot: RobotConfig = field(default_factory=RobotConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    ray_cast: RayCastConfig = field(default_factory=RayCastConfig)
    measurement_model: MeasurementModelConfig = field(default_factory=MeasurementModelConfig)
    mcl: MCLConfig = field(default_factory=MCLConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    uart: UartConfig = field(default_factory=UartConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)

    # ****** LOADING / SAVING ******

    @classmethod
    def load(cls, filepath=None):
        """
        Reads a config JSON. Defaults to config/robot_config.json, and falls back to the
        built-in defaults if that file does not exist, so a fresh checkout still runs.

        Args:
            filepath (str | Path | None): path to the JSON, absolute or relative to the
                                          project root.
        Returns:
            config (Config)
        """
        path = Path(filepath) if filepath is not None else DEFAULT_CONFIG_FILE
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if filepath is None and not path.exists():
            print(f"Config: {path} not found, using built-in defaults")
            return cls()
        return cls.from_json(path)

    @classmethod
    def from_json(cls, filepath):
        """Reads a config JSON. Unlike load(), a missing file is an error."""
        path = Path(filepath)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Builds a Config from a nested dict, section by section. Sections and keys that are
        absent keep their defaults; anything present but unrecognised raises, so a typo
        surfaces at startup instead of silently leaving the default in place.
        """
        # "_comment" is allowed anywhere so the JSON can be annotated.
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - {"_comment"}
        if unknown:
            raise ValueError(f"Config: unknown section(s) {sorted(unknown)}, expected {sorted(known)}")

        sections = {}
        for f in fields(cls):
            section_data = data.get(f.name)
            if section_data is None:
                continue
            if not isinstance(section_data, dict):
                raise ValueError(f"Config: section '{f.name}' must be an object, got {type(section_data).__name__}")
            sections[f.name] = _build_section(_SECTION_TYPES[f.name], f.name, section_data)
        return cls(**sections)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, filepath=None, indent=4) -> str:
        """Writes the current values back out as JSON. Handy for regenerating the default file."""
        path = Path(filepath) if filepath is not None else DEFAULT_CONFIG_FILE
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)
            f.write("\n")
        return str(path)

    # ****** RESOLVED PATHS ******

    def camera_calib_path(self) -> str:
        return str(CONFIG_DIR / self.paths.camera_calib)

    def z_real_path(self) -> str:
        return str(CONFIG_DIR / self.paths.z_real)

    def map_path(self) -> str:
        return str(CONFIG_DIR / self.paths.map)

    def validate(self, check_files=True):
        """
        Sanity checks the values, and optionally that the three data files actually exist.
        Call it once at startup: a missing npz otherwise surfaces deep inside Camera or
        FloorScaleCalibration, and a bad mixture weight not at all.

        Raises:
            ValueError / FileNotFoundError on the first problem found.
        """
        if self.ray_cast.n_rays <= 0:
            raise ValueError("ray_cast.n_rays must be positive")
        if self.ray_cast.max_range <= 0:
            raise ValueError("ray_cast.max_range must be positive")
        if self.mcl.n_particles <= 0:
            raise ValueError("mcl.n_particles must be positive")

        mixture = (self.measurement_model.z_hit + self.measurement_model.z_short
                   + self.measurement_model.z_max + self.measurement_model.z_rand)
        if mixture <= 0:
            raise ValueError("measurement_model z_hit/z_short/z_max/z_rand must sum to more than 0")

        if not 0.0 <= self.mcl.resample_ess_ratio <= 1.0:
            raise ValueError("mcl.resample_ess_ratio must be in [0, 1]")

        if check_files:
            missing = [p for p in (self.camera_calib_path(), self.z_real_path(), self.map_path())
                       if not Path(p).exists()]
            if missing:
                raise FileNotFoundError(f"Config: missing data file(s): {missing}")
        return self

    # ****** FACTORIES ******
    # Imported lazily so that merely importing this module does not drag in numpy, scipy,
    # cv2 or torch - the Pi-side scripts that only want a port number should not pay for it.

    def load_map(self, default_value=0):
        """Builds the OccupancyGrid named by paths.map."""
        from app.localization.map import OccupancyGrid
        return OccupancyGrid.from_json(self.map_path(), default_value=default_value)

    def beam_sensor_model(self):
        """Builds a BeamSensorModel from the measurement_model section, with ray_cast.max_range."""
        from app.localization.measurement_model import BeamSensorModel
        mm = self.measurement_model
        model = BeamSensorModel(max_range=self.ray_cast.max_range,
                                sigma_hit=mm.sigma_hit, lambda_short=mm.lambda_short,
                                z_hit=mm.z_hit, z_short=mm.z_short,
                                z_max=mm.z_max, z_rand=mm.z_rand, alpha=mm.alpha)
        model.max_eps = mm.max_eps
        return model

    def ray_caster(self, grid, fov_x=None):
        """
        Builds a RayCaster against `grid`.

        fov_x is a camera intrinsic, so it is not really config: pass the value from
        robot.camera.fov_x. ray_cast.fov_x is the override for tests that have no camera
        (it is used when non-zero and nothing is passed).
        """
        from app.localization.ray_caster import RayCaster
        if fov_x is None:
            fov_x = self.ray_cast.fov_x
        if not fov_x:
            raise ValueError("ray_caster: pass robot.camera.fov_x, or set ray_cast.fov_x in the config")
        return RayCaster(grid, fov_x, n_rays=self.ray_cast.n_rays,
                         max_range=self.ray_cast.max_range,
                         cam_forward=self.robot.cam_forward,
                         occ_threshold=self.ray_cast.occ_threshold)

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)


_SECTION_TYPES = {
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


def _build_section(section_cls, section_name, data: dict):
    """One section of the JSON into its dataclass, rejecting unknown keys."""
    known = {f.name for f in fields(section_cls)}
    unknown = set(data) - known - {"_comment"}
    if unknown:
        raise ValueError(f"Config: unknown key(s) {sorted(unknown)} in section '{section_name}', "
                         f"expected {sorted(known)}")

    values = {k: v for k, v in data.items() if k != "_comment"}
    # JSON has no tuples, so anything defaulting to one (cam_t, init_pose) arrives as a list.
    for f in fields(section_cls):
        if f.name in values and isinstance(f.default, tuple) and isinstance(values[f.name], list):
            values[f.name] = tuple(values[f.name])
    return section_cls(**values)


if __name__ == "__main__":
    config = Config.load()
    print(config)
    print(f"\ncamera calib: {config.camera_calib_path()}")
    print(f"z_real:       {config.z_real_path()}")
    print(f"map:          {config.map_path()}")

    try:
        config.validate()
        print("\nvalidate: ok")
    except (ValueError, FileNotFoundError) as e:
        print(f"\nvalidate: {e}")
