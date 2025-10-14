# Fast Slam base repository
Built to work on Eufsim simulator.
Contains:
* Individual nodes for prediction step, 4 algorithms for Data association over a known map and JCBB based correction step accessible through:

```ros2 run fslam fslam_pred```

Runs prediction without wheel speed corrections. Fairly accurate but drifts.
Gyro bis correction needs 1 second of recorded data at the time of initialization. (Reduced from 5 post gyro based correction and yaw inflation)

```ros2 run fslam fslam_jcbb```

Runs data association currently over a known skidpad map. To be run after prediction starts

```ros2 run fslam plot``` (optional)

Run to live view the prediction & correction results relative to GT. #RUN ONLY AFTER VEHICLE MOVES (ISSUE WITH PLOTTER)

### Subscribe to the output top of JCBB node for correct pose (use while skidpad)


* Path Planning code for live delaunay plots and midpoint generation
 - Two codes : one has a fallback that predicts cones when one side is not visible
 - Other will be developed based off of mapping
 - Also included a temporary curve fitting code (not optimised)

Access nodes through this:

```ros2 run pathp path_plot```



* Mapping Codes yet to be finished. So far map exists if driven correctly and on track. Unnecessary turns blow the dynamic gating algorithm

WILL UPDATE MAPPING AS IT PROGRESSES.
