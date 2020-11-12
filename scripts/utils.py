import numpy as np
from scipy.spatial.transform import Rotation as R


def _quat_mult(q1, q2):
    x0, y0, z0, w0 = q2
    x1, y1, z1, w1 = q1
    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0])


def _quat_conj(q):
    ret = np.copy(q)
    ret[:3] *= -1
    return ret


def _get_angle_axis(current, target):
    # Return:
    # (1) angle err between orientations
    # (2) Unit rotation axis
    rot = R.from_quat(_quat_mult(current, _quat_conj(target)))

    rotvec = rot.as_rotvec()
    norm = np.linalg.norm(rotvec)

    if norm > 1E-8:
        return norm, (rotvec / norm)
    else:
        return 0, np.zeros(len(rotvec))


def pitch_orient(observation):
    manip_angle = 0
    manip_axis = np.zeros(3)
    manip_arm = 0
    current = observation["achieved_goal"]["orientation"]
    target = observation["desired_goal"]["orientation"]

    axes = np.eye(3)
    axes = np.concatenate((axes, -1 * np.eye(3)), axis=0)
    z_axis = [0., 0., 1.]
    min_angle = 100.
    min_axis_id = 0
    for i, axis in enumerate(axes):
        qg_i = R.from_quat(target).apply(axis)
        angle = np.arccos(np.dot(qg_i, z_axis))
        if angle < min_angle:
            min_angle = angle
            min_axis_id = i

    manip_axis = R.from_quat(current).apply(axes[min_axis_id])
    if np.abs(1 - manip_axis[2]) < 0.03:
        manip_angle = 0
    elif np.abs(-1 - manip_axis[2]) < 0.03:
        manip_angle = 180
    else:
        manip_angle = 90

    if manip_angle in [90, 180]:
        arm_angle = np.arctan2(
            manip_axis[1], manip_axis[0])
        
        print ("arm angle: ", arm_angle, " ax: ", manip_axis)
        arm_angle += np.pi/2
        if arm_angle > np.pi:
            arm_angle -= 2*np.pi

        if arm_angle < (np.pi/2 + np.pi/3) and arm_angle > (np.pi/2 - np.pi/3):  # 30 to 150
            manip_arm = 0
        elif arm_angle > (-np.pi/2) and arm_angle < (np.pi/2 - np.pi/3):  # -90 to 30
            manip_arm = 1
        else:
            manip_arm = 2

    print("Manip angle: ", manip_angle, " Manip arm: ", manip_arm)
    return manip_angle, manip_axis, manip_arm


def yaw_orient(current, target):
    rot = R.from_quat(current)
    # target - current
    diff_rot = (R.from_quat(target) * R.from_quat(current).inv())
    yaw_diff = diff_rot.as_euler('xyz')[-1]

    return yaw_diff
