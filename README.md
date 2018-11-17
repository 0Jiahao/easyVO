# easyVO
A simple real-time visual odometry approach for stereo camera.
### Method
This project is mainly built based on [eVO](http://w3.onera.fr/copernic/sites/w3.onera.fr.copernic/files/documents/conference_papers/2013_-_iros_-_evo_a_realtime_embedded_stereo_odometry_for_mav_applications.pdf).
### Data Set
The dataset we are using to verify this approach is the [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).
### Result
<div align=center><img width="640" src="https://github.com/0Jiahao/easyVO/blob/master/result.gif"/></div> 

### Future Work
- Adding Bundle Adjustment or EKF for refinement.
- Adding keyframe alignment (short-term loop closure).
