The architecture of my robot contains a Pi streaming camera frame to the laptop which gets processed with Vision Transformer Monocular Depth Estimation and returns some sort of processed vision data (usually range casting) back to the Pi for localization. 

The Pi runs /tests/ python files and the laptop runs /laptop/ python file

dont bother with the app/yolo folder, it is an archived section that never gets used

if you ever write test case or __main__ functionality, always create a Robot instance and loading the appropriate configs, never hardcode those config numbers, such as the fov_x of the camera