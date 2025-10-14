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
