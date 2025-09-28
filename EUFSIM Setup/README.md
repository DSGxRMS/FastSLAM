# EUFSIM on Windows 11 (WSL2) — Ubuntu 20.04 + ROS 2 Galactic

> Stable combo that avoids Humble-era API drift: **Ubuntu 20.04 (Focal) + ROS 2 Galactic + Gazebo Classic 11**.  
> Works with **WSL2 + WSLg** (no dual boot, no VM).

---

## 0) Pre-reqs (Windows)

Check WSL + WSLg in **PowerShell**:

```powershell
wsl --version
wsl --status
```
# 1) Install Ubuntu 20.04 (won't work on 22.04, EUFSIM is not compatible with ROS2 Humble)
## In powershell (admin)
```
# List available distros (optional)
wsl --list --online

# Install Ubuntu 20.04 (or via Microsoft Store)
wsl --install -d Ubuntu-20.04

# Ensure WSL2 & set as default
wsl --set-version Ubuntu-20.04 2
wsl --set-default Ubuntu-20.04
wsl --list --verbose
```

# 2) Enable systemd + base OS setup (inside Ubuntu 20.04)
```
# OS sanity (should show Ubuntu 20.04 / focal)
cat /etc/os-release

# Enable systemd in WSL
sudo mkdir -p /etc
printf "[boot]\nsystemd=true\n" | sudo tee /etc/wsl.conf

# Locale + updates + common tools
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y software-properties-common curl gnupg2 lsb-release locales git
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Restart WSL from Windows side, then relaunch Ubuntu 20.04
# (PowerShell/Admin):  wsl --shutdown
# Back in Ubuntu:
systemctl is-system-running || true   # expect: running (or degraded)
```


# 3) Install ROS 2 Galactic
```
# ROS 2 apt source (keyring method)
sudo mkdir -p /usr/share/keyrings
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
 | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
 | sudo tee /etc/apt/sources.list.d/ros2.list

sudo apt update
sudo apt install -y ros-galactic-desktop python3-colcon-common-extensions python3-vcstool python3-rosdep

# Shell setup
echo "source /opt/ros/galactic/setup.bash" >> ~/.bashrc
source /opt/ros/galactic/setup.bash

# rosdep init/update (first time OK to ignore 'already exists')
sudo rosdep init || true
rosdep update


rviz2     # should open; close it

```



# 4) Install Gazebo Classic 11

```
# OSRF Gazebo repo + key
sudo mkdir -p /usr/share/keyrings
sudo wget https://packages.osrfoundation.org/gazebo.gpg \
  -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
 | sudo tee /etc/apt/sources.list.d/gazebo-stable.list

sudo apt update
sudo apt install -y gazebo11 libgazebo11-dev

# ROS ↔ Gazebo bridge + CMake exports needed by EUFSIM
sudo apt install -y \
  ros-galactic-gazebo-ros-pkgs \
  ros-galactic-gazebo-ros \
  ros-galactic-gazebo-dev


gazebo --version   # expect 11.x
gazebo             # should open; close it

```

# 5) Create workspace and clone EUFSIM repos
```
# Workspace
mkdir -p ~/eufs_ws/src
cd ~/eufs_ws/src

# Clone repos (IMPORTANT: eufs_msgs default branch, NOT `-b ros2`)
git clone https://gitlab.com/eufs/eufs_sim.git
git clone https://gitlab.com/eufs/eufs_msgs.git
git clone https://gitlab.com/eufs/eufs_rviz_plugins.git

ls -la ~/eufs_ws/src
```


# 6) Resolve dependencies

```
cd ~/eufs_ws

# Make sure rosdep is ready (safe to re-run)
sudo apt update
sudo apt install -y python3-rosdep
sudo rosdep init || true
rosdep update

# Install package deps (ignore-src so it only installs system deps)
rosdep install --from-paths src --ignore-src -r -y
```

## Install explicitly (gave me error)

```
sudo apt install -y \
  ros-galactic-ackermann-msgs \
  ros-galactic-joint-state-publisher \
  ros-galactic-joint-state-publisher-gui \
  ros-galactic-xacro
```


# Try Build (should work)

```
cd ~/eufs_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Source overlay now and persist
echo "source ~/eufs_ws/install/setup.bash" >> ~/.bashrc
source ~/eufs_ws/install/setup.bash
```
## If fails try this:
```
cd ~/eufs_ws/src
rm -rf eufs_msgs
git clone https://gitlab.com/eufs/eufs_msgs.git
cd ~/eufs_ws
colcon build --symlink-install --packages-select eufs_msgs
colcon build --symlink-install --packages-select eufs_plugins
```


# 8) Environment for EUFSIM (models/resources + project root)

```
# EUFSIM repo root (for launch files that reference EUFS_MASTER)
echo 'export EUFS_MASTER=~/eufs_ws/src/eufs_sim' >> ~/.bashrc

# Gazebo paths (models + tracks + resources)
echo 'export GAZEBO_MODEL_PATH=~/eufs_ws/src/eufs_sim/eufs_models:~/eufs_ws/src/eufs_sim/eufs_tracks:${GAZEBO_MODEL_PATH}' >> ~/.bashrc
echo 'export GAZEBO_RESOURCE_PATH=~/eufs_ws/src/eufs_sim:${GAZEBO_RESOURCE_PATH}' >> ~/.bashrc

# Load the new env now
source ~/.bashrc
```


# 9) Launch EUFSIM (visual cones in Gazebo + RViz)

```
# Make sure base ROS + workspace are sourced
source /opt/ros/galactic/setup.bash
source ~/eufs_ws/install/setup.bash
```

# Start the simulator (GUI ON, skidpad track, ground-truth publisher)
```
ros2 launch eufs_launcher simulation.launch.py \
  use_sim_time:=true \
  track:=skidpad \
  vehicleModel:=DynamicBicycle \
  commandMode:=acceleration \
  vehicleModelConfig:=configDry.yaml \
  robot_name:=eufs \
  gazebo_gui:=true \
  pub_ground_truth:=true \
  launch_group:=no_perception \
  rviz:=true
```

## Verify in separate terminal
```
source /opt/ros/galactic/setup.bash
source ~/eufs_ws/install/setup.bash

ros2 topic list | grep -E 'clock|odom|imu|scan|cmd' || ros2 topic list
ros2 topic echo /clock --once
```

## Test command for car to move
```
# Check message type first (should be ackermann_msgs/msg/AckermannDriveStamped)
ros2 topic info /cmd -v

# Small forward command (Ctrl+C to stop publishing)
ros2 topic pub /cmd ackermann_msgs/msg/AckermannDriveStamped \
  "{drive: {speed: 3.0, acceleration: 1.0, steering_angle: 0.0}}"
```

# 10) One-liner startup (after first setup)
```
cat > ~/run_eufsim.sh <<'EOF'
#!/usr/bin/env bash
set -e
source /opt/ros/galactic/setup.bash
source ~/eufs_ws/install/setup.bash

export EUFS_MASTER=~/eufs_ws/src/eufs_sim
export GAZEBO_MODEL_PATH=~/eufs_ws/src/eufs_sim/eufs_models:~/eufs_ws/src/eufs_sim/eufs_tracks:${GAZEBO_MODEL_PATH}
export GAZEBO_RESOURCE_PATH=~/eufs_ws/src/eufs_sim:${GAZEBO_RESOURCE_PATH}

ros2 launch eufs_launcher simulation.launch.py \
  use_sim_time:=true track:=skidpad \
  vehicleModel:=DynamicBicycle commandMode:=acceleration vehicleModelConfig:=configDry.yaml \
  robot_name:=eufs gazebo_gui:=true pub_ground_truth:=true launch_group:=no_perception rviz:=true
EOF
chmod +x ~/run_eufsim.sh
```


# Run with this (Will open a skidpad map)
```
~/run_eufsim.sh
```