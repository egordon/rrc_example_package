import numpy as np
from scipy.spatial.transform import Rotation as R


def get_rest_arm(observation):
    current = observation["achieved_goal"]["orientation"]
    axis = [1., 0., 0.]
    manip_axis = R.from_quat(current).apply(axis)

    arm_angle = np.rad2deg(np.arctan2(
            manip_axis[1], manip_axis[0]))
    
    if arm_angle > -90.0 and arm_angle <= 30.0:
        rest_arm = 1
    elif arm_angle > 30.0 and arm_angle <= 150:
        rest_arm = 0
    else:
        rest_arm = 2
    
    manip_axis[2] = 0
    return rest_arm, manip_axis

def get_rest_arm2(observation):
    current = observation["achieved_goal"]["orientation"]

    axis = [0., 1., 0.]
    manip_axis = R.from_quat(current).apply(axis)

    rot = R.from_quat(current)
    yaw = np.arctan2(manip_axis[1], manip_axis[0]) * 180. / np.pi

    if yaw > -90.0 and yaw <= 30.0:
        rest_arm = 1
    elif yaw > 30.0 and yaw <= 150:
        rest_arm = 0
    else:
        rest_arm = 2
    return rest_arm, manip_axis