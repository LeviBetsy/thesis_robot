# Embedded Thesis Project - Levi
## Instalalation
- [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix)
## Directory set up
- Raspberry pi, PI OS
- pyenv local 3.12.13
- python -m venv myvenv (then activate environment)
## Running YOLO v26
export TMPDIR=$HOME/pip_tmp
mkdir -p $TMPDIR
- pip install opencv-python ultralytics pyserial 
## Running Test code
- pip install pynput python-dotenv


tail -f data/odometry_log.txt 
ssh -L 5000:localhost:5000 -L 8080:localhost:8080 USER@PI_IP_ADDRESS
This is for opening an SSH tunnel so what gets sent to localhost 8080 on the laptop, gets sent to the Pi's 8080


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
./build/x86_64/mono_depth_estimation -n ../../../resources/models/hailo8l/scdepthv3.hef -i usb --camera-resolution sd --output-resolution sd
