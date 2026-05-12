# Embedded Thesis Project - Levi
## Instalalation
- [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix)
## Directory set up
- Raspberry pi, PI OS
- pyenv local 3.12.13
- python -m venv myvenv (then activate environment)
## Running YOLO v26
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
