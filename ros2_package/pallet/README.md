# Pallet Detection and Segmentation

This package provides ROS2 nodes for pallet detection and segmentation using YOLO models. It processes images from a recorded ROS bag to perform detection and segmentation of pallets.

## Prerequisites

- ROS2 Humble
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO

## Setup Instructions

1. Initialize ROS2 environment:
```bash
source /opt/ros/humble/setup.bash
```

2. Create and build workspace:
```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone or copy the pallet package
# [Copy pallet package into src folder]

# Build the workspace
cd ~/ros2_ws
colcon build

# Source the workspace
source install/setup.bash
```

## Usage

### Running the ROS2 Bag

Play the ROS bag at 1Hz:
```bash
ros2 bag play <path-to-your-rosbag> --rate 0.067
```
Example:
```bash
ros2 bag play /home/user/bags/internship_assignment_sample_bag --rate 0.067
```
This rate setting ensures 1Hz playback (assuming the bag was recorded at ~15Hz).

### Running Inference

In a new terminal:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run pallet infer
```

## Expected Output

The inference node will display:
- Original image (resized)
- Segmentation results
- Detection results
- Terminal output showing detection counts and confidence scores

## Notes

- Adjust the bag playback rate if you need different visualization speeds:
  - `--rate 0.033` for 0.5Hz (one image every 2 seconds)
  - `--rate 0.017` for 0.25Hz (one image every 4 seconds)

## Troubleshooting

If you encounter the QoS warning:
```
[WARN] []: New publisher discovered on topic '/robot1/zed2i/left/image_rect_color', offering incompatible QoS
```
This is expected and handled by the node's QoS settings.

## Model Paths

Make sure the YOLO model weights are in the correct location:
```
pallet/
├── models/
│   ├── best_segment.pt
│   └── best_detect.pt
```

## License

[Your license information here]