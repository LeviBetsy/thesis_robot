# Embedded Thesis Project - Levi
This project is developed for my senior thesis. The primary objective is to build an autonomous robot capable of navigating real-world environments by attempting to replace traditional depth sensors—such as LiDAR and stereo cameras—with Monocular Depth Estimation (MDE). The robot utilizes Monte Carlo Localization as its localization method.

The system's architecture is distributed across three main hardware components:

- Laptop: Hosts the Monocular Depth Estimation model, processing visual input to output range-finder data for the system.

- Raspberry Pi: Acts as the central brain, running the high-level decision-making software and executing the Monte Carlo localization based on the range-finder data.

- MSP432 Microcontroller: Handles the bare-metal, low-level control of the TI-RSLK rover platform, written without the use of a Hardware Abstraction Layer (HAL).
## TI-RLSK Rover
[Github for MSP432 Code](https://github.com/LeviBetsy/thesis_robot_msp432)

---

## Repository Layout

| Path | Runs on | Purpose |
| --- | --- | --- |
| `app/perception`, `app/localization` | Pi | Ray casting, measurement model, Monte Carlo Localization |
| `app/robot_module`, `app/control` | Pi | Robot state, config loading, motor/UART control |
| `app/stream` | both | ZMQ transport between Pi and laptop |
| `app/util` | both | Config loading and shared helpers |
| `laptop/mde_projection` | Laptop | Depth model inference, depth → range projection, map visualization |
| `test/` | Pi | Entry points and manual test scripts |
| `config/` | both | `robot_config.json` and maps under `config/map/` |
| `app/yolo` | — | Archived, unused |

Configuration (camera intrinsics, FOV, geometry, map files) lives in `config/` — read it through the config loader rather than hardcoding values in scripts.

---

## Installation

### 1. Python environment (Pi and laptop)

Install [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix), then:

```bash
pyenv local 3.12.13
python -m venv myvenv
source myvenv/bin/activate
```

On the Pi, pip may run out of space in the default temp dir:

```bash
export TMPDIR=$HOME/pip_tmp
mkdir -p $TMPDIR
```

### 2. Dependencies

```bash
# perception / YOLO / general
pip install opencv-python ultralytics pyserial scikit-learn

# test + control scripts
pip install pynput python-dotenv

# streaming and MDE localization
sudo apt install libgirepository-2.0-dev
pip install PyGObject pyzmq

# ONNX inference
pip install onnxruntime

# model training notebooks
pip install ipykernel
```

For training, install the VS Code Jupyter extension and select the `myvenv` kernel.

---

## Depth Models

### Depth Anything V2 (laptop)

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2/metric_depth
pip install -r requirements.txt
```

Download the metric checkpoint:

```bash
mkdir -p app/models/DAV2_checkpoint
cd app/models/DAV2_checkpoint
curl -L -o depth_anything_v2_metric_hypersim_vits.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"
```

> If `open3d` fails to install on the Pi, use `pip install open3d-unofficial-arm`.

### Hailo NPU (Pi)

Update firmware first:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo rpi-eeprom-update -a
sudo reboot
```

Install the driver and runtime:

```bash
sudo apt install dkms
sudo apt install hailo-all
sudo reboot
```

Verify the NPU is detected:

```bash
hailortcli fw-control identify
```

Then clone the Hailo apps repo, run their `install.sh` to fetch models and pipelines, activate their environment with `source setup_env.sh`, and check the per-app requirements for compatibility.

### SCDepth on the NPU

Always `source setup_env.sh` first, then build:

```bash
cd hailo-apps/hailo_apps/cpp/depth_estimation_mono
./build.sh
```

Headless run, streaming to port 5001:

```bash
./build/x86_64/mono_depth_estimation \
  -n ../../../resources/models/hailo8l/scdepthv3.hef \
  -i usb --camera-resolution sd --output-resolution sd --no-display
```

Single-image run (ArUco frame):

```bash
./build/x86_64/mono_depth_estimation \
  -n ../../../resources/models/hailo8l/scdepthv3.hef --no-display \
  -i ../../../../thesis_robot/data/aruco/aruco2.jpg \
  -o ../../../../thesis_robot/data/aruco_mde/ --output-resolution sd
```

---

## Networking: Pi ↔ Laptop

Pipeline: the Pi captures and undistorts a frame → streams it to the laptop → the laptop runs MDE → range data goes back to the Pi.

Forward laptop traffic to the Pi (anything sent to `localhost:8080` on the laptop reaches the Pi's `8080`):

```bash
ssh -L 5000:localhost:5000 -L 8080:localhost:8080 USER@PI_IP_ADDRESS
```

Reverse tunnel for the GStreamer feed from the Hailo NPU MDE:

```bash
ssh -R 5001:localhost:5001 USER@PI_IP_ADDRESS
```

---

## Running

Teleop controller (Pi):

```bash
python3 -m test.twitch_test.controller
```

Measurement model test — particles spread in x/y only, heading fixed:

```bash
python3 test/localization/measurement_model_test.py --n-particles 50 --sigma-xy 0.2 --sigma-theta 0
```

Watch odometry as it is logged:

```bash
tail -f data/odometry_log.txt
```

## Instalalation
- [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix)
## Directory set up
- Raspberry pi, PI OS
- pyenv local 3.12.13
- python -m venv myvenv (then activate environment)
## Running YOLO v26 and other things
export TMPDIR=$HOME/pip_tmp
mkdir -p $TMPDIR
- pip install opencv-python ultralytics pyserial scikit-learn
## Running Test code
- pip install pynput python-dotenv


tail -f data/odometry_log.txt 
ssh -L 5000:localhost:5000 -L 8080:localhost:8080 USER@PI_IP_ADDRESS
This is for opening an SSH tunnel so what gets sent to localhost 8080 on the laptop, gets sent to the Pi's 8080

ssh -R 5001:localhost:5001 USER@PI_IP_ADDRESS 
This is for SSH tunnel for gstreamer of hailo npu mde


python3 -m test.twitch_test.controller

## Training model
- pip install ipykernel
- Install the vs code jupyter notebook extension, choose the kernel using the environment you made and run it on VS code

## RUN onnx
- pip install onnxruntime 

## Depth Anything V2
!git clone https://github.com/DepthAnything/Depth-Anything-V2
%cd Depth-Anything-V2/metric_depth
!pip install -r requirements.txt
pip install open3d-unofficial-arm if run into problem with open3d on pi chip

mkdir app/models/DAV2_checkpoint
cd app/models/DAV2_checkpoint
curl -L -o depth_anything_v2_metric_hypersim_vits.pth \
"https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"

# NPU set up
sudo apt update
sudo apt full-upgrade -y
sudo rpi-eeprom-update -a
sudo reboot

sudo apt install dkms
sudo apt install hailo-all
sudo reboot

## verify NPU connection
hailortcli fw-control identify

## next
clone hailo app and do install.sh for their model and pipeline

- activate their environment:
source setup_env.sh
- look at specific app requirements and stuff to see if we can run it

## SCDepth
don't forget to setup_env.sh

cd hailo-apps/hailo_apps/cpp/depth_estimation_mono
./build.sh
### run with no display and stream to 5001
./build/x86_64/mono_depth_estimation -n ../../../resources/models/hailo8l/scdepthv3.hef -i usb --camera-resolution sd --output-resolution sd --no-display
### run on one image aruco
./build/x86_64/mono_depth_estimation -n ../../../resources/models/hailo8l/scdepthv3.hef --no-display -i ../../../../thesis_robot/data/aruco/aruco2.jpg -o ../../../../thesis_robot/data/aruco_mde/ --output-resolution sd

#### see stream
brew install gstreamer gst-plugins-base gst-plugins-good

ffplay tcp://127.0.0.1:5001

# this for streaming through cpp
sudo apt install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-x libgstreamer-plugins-base1.0-dev -y
brew install ffmpeg --with-sdl2 on laptop


## Capture and undistort image from Pi -> Stream to Laptop -> Laptop run DAV2 -> Return depth map to Pi
gst-launch-1.0 tcpclientsrc host=127.0.0.1 port=5002 ! matroskademux ! h264parse ! decodebin ! videoconvert ! autovideosink sync=false

# MDE Localization test
sudo apt install libgirepository-2.0-dev
pip install PyGObject
pip install pyzmq

# Measurement model test
theta does not change: 
python3 test/localization/measurement_model_test.py --n-particles 50 --sigma-xy 0.2 --sigma-theta 0