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
    current = observation["achieved_goal"]["orientation"]
    target = observation["desired_goal"]["orientation"]

    # Sets pre-manipulation 90 or 180-degree rotation
    manip_angle = 0
    manip_axis = np.zeros(3)
    manip_arm = 0

    # angle between current and goal orientation
    minAngle, _ = _get_angle_axis(current, target)

    # Check 90-degree rotations
    # Find face having min angle with goal orientation
    for axis in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0]), np.array([0, -1, 0])]:
        new_axis = R.from_quat(current).apply(axis)
        rotation = R.from_rotvec(np.pi / 2 * new_axis)
        new_current = rotation * R.from_quat(current)

        angle, _ = _get_angle_axis(new_current.as_quat(), target)
        target_eul = R.from_quat(target).as_euler('xyz')
        new_current_eul = new_current.as_euler('xyz')
        angle = np.abs(target_eul - new_current_eul)[1]
        if angle < minAngle:
            minAngle = angle
            manip_angle = 90
            manip_axis = new_axis

    # Check 180 degree rotation
    # NO TIME FOR 180
    new_axis = R.from_quat(current).apply(np.array([1, 0, 0]))
    rotation = R.from_rotvec(np.pi * new_axis)
    new_current = rotation * R.from_quat(current)
    angle, _ = _get_angle_axis(new_current.as_quat(), target)
    if angle < minAngle:
        minAngle = angle
        manip_angle = 180
        manip_axis = new_axis

    # Determine rotation arm
    arm_angle = np.arctan2(
        manip_axis[1], manip_axis[0]) + np.pi/2
    if arm_angle > np.pi:
        arm_angle -= 2*np.pi
    print("Arm Angle: " + str(arm_angle))

    # how?
    if arm_angle < (np.pi/2 + np.pi/3) and arm_angle > (np.pi/2 - np.pi/3):  # 30 to 150
        manip_arm = 0
    elif arm_angle > (-np.pi/2) and arm_angle < (np.pi/2 - np.pi/3):  # -90 to 30
        manip_arm = 1
    else:
        manip_arm = 2

    print("Manip Arm: " + str(manip_arm))
    print("Manip Angle: " + str(manip_angle))
    return manip_angle, manip_axis, manip_arm


def yaw_orient(current, target):
    rot = R.from_quat(current)
    # target - current
    diff_rot = (R.from_quat(target) * R.from_quat(current).inv())
    yaw_diff = diff_rot.as_euler('xyz')[-1]

    return yaw_diff
