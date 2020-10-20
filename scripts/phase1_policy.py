#!/usr/bin/env python3
"""Simple example on how to move the robot."""
import json
import sys
import enum

import robot_fingers
import robot_interfaces

import numpy as np
from scipy.spatial.transform import Rotation as R

from rrc_example_package import cube_env
from trifinger_simulation import trifingerpro_limits

import pybullet


class States(enum.Enum):
    """ Different States for StateSpacePolicy """

    #: Align fingers to 3 points above cube
    ALIGN = enum.auto()

    #: Lower coplanar with cube
    LOWER = enum.auto()

    #: Move into cube
    INTO = enum.auto()

    #: Move cube to goal
    GOAL = enum.auto()

    #: Orient correctly
    ORIENT = enum.auto()


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


class StateSpacePolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, env, difficulty, observation):
        self.action_space = env.action_space
        self.state = States.ALIGN
        self.difficulty = difficulty

        self.EPS = 1E-2
        self.DAMP = 1E-6
        self.CUBE_SIZE = 0.0325

        if difficulty == 4:
            # Do Pre-manpulation
            self.do_premanip = True
            self._calculate_premanip(observation)
        else:
            self.do_premanip = False

        self.t = 0
        self.finger = env.simfinger

    def _calculate_premanip(self, observation):
        current = observation["achieved_goal"]["orientation"]
        target = observation["desired_goal"]["orientation"]

        # Sets pre-manipulation 90 or 180-degree rotation
        self.manip_angle = 0
        self.manip_axis = np.zeros(3)
        self.manip_arm = 0

        minAngle, _ = _get_angle_axis(current, target)

        # Check 90-degree rotations
        for axis in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0]), np.array([0, -1, 0])]:
            new_axis = R.from_quat(current).apply(axis)
            rotation = R.from_rotvec(np.pi / 2 * new_axis)
            new_current = rotation * R.from_quat(current)

            angle, _ = _get_angle_axis(new_current.as_quat(), target)
            if angle < minAngle:
                minAngle = angle
                self.manip_angle = 90
                self.manip_axis = new_axis

        # Check 180 degree rotation
        """ NO TIME FOR 180
        new_axis = R.from_quat(current).apply(np.array([1, 0, 0]))
        rotation = R.from_rotvec(np.pi * new_axis)
        new_current = rotation * R.from_quat(current)
        angle, _ = _get_angle_axis(new_current.as_quat(), target)
        if angle < minAngle:
            minAngle = angle
            self.manip_angle = 180
            self.manip_axis = new_axis
        """

        # Determine rotation arm
        arm_angle = np.arctan2(
            self.manip_axis[1], self.manip_axis[0]) + np.pi/2
        if arm_angle > np.pi:
            arm_angle -= 2*np.pi
        print("Arm Angle: " + str(arm_angle))

        if arm_angle < (np.pi/2 + np.pi/3) and arm_angle > (np.pi/2 - np.pi/3):
            self.manip_arm = 0
        elif arm_angle > (-np.pi/2) and arm_angle < (np.pi/2 - np.pi/3):
            self.manip_arm = 1
        else:
            self.manip_arm = 2

        print("Manip Arm: " + str(self.manip_arm))
        print("Manip Angle: " + str(self.manip_angle))

    def _get_gravcomp(self, observation):
        # Returns: 9 torques required for grav comp
        ret = pybullet.calculateInverseDynamics(self.finger.finger_id,
                                                observation["observation"]["position"].tolist(
                                                ),
                                                np.zeros(
                                                    len(observation["observation"]["position"])).tolist(),
                                                np.zeros(len(observation["observation"]["position"])).tolist())
        ret = np.array(ret)
        return ret

    def _get_jacobians(self, observation):
        # Returns: numpy(3*num_fingers X num_joints_per_finger)
        ret = []
        for tip in self.finger.pybullet_tip_link_indices:
            J, _ = pybullet.calculateJacobian(
                self.finger.finger_id,
                tip,
                np.zeros(3).tolist(),
                observation["observation"]["position"].tolist(),
                observation["observation"]["velocity"].tolist(),
                np.zeros(len(observation["observation"]["position"])).tolist()
            )
            ret.append(J)
        ret = np.vstack(ret)
        return ret

    def _get_tip_poses(self, observation):
        return observation["observation"]["tip_positions"].flatten()

    def prealign(self, observation):
        # Return torque for align step
        current = self._get_tip_poses(observation)

        # Determine arm locations
        locs = [np.zeros(3), np.zeros(3), np.zeros(3)]

        for i in range(3):
            index = (self.manip_arm + 1 - i) % 3
            locs[index] = 1.5 * \
                R.from_rotvec(
                    np.pi/2 * i * np.array([0, 0, 1])).apply(self.manip_axis)
            locs[index][2] = 2

        desired = np.tile(observation["achieved_goal"]["position"], 3) + \
            self.CUBE_SIZE * np.hstack(locs)

        err = desired - current
        if np.linalg.norm(err) < 3 * self.EPS:
            print("PRE LOWER")
            self.state = States.LOWER
        return 0.08 * err

    def prelower(self, observation):
        # Return torque for align step
        current = self._get_tip_poses(observation)

        # Determine arm locations
        locs = [np.zeros(3), np.zeros(3), np.zeros(3)]

        for i in range(3):
            index = (self.manip_arm + 1 - i) % 3
            locs[index] = 1.5 * \
                R.from_rotvec(
                    np.pi/2 * i * np.array([0, 0, 1])).apply(self.manip_axis)
            if i == 1:
                locs[index][2] += 0.4

        desired = np.tile(observation["achieved_goal"]["position"], 3) + \
            self.CUBE_SIZE * np.hstack(locs)

        err = desired - current
        if np.linalg.norm(err) < 3*self.EPS:
            self.previous_state = observation["observation"]["position"]
            print("PRE INTO")
            self.state = States.INTO
        return 0.08 * err

    def preinto(self, observation):
        # Return torque for into step
        current = self._get_tip_poses(observation)
        
        desired = np.tile(observation["achieved_goal"]["position"], 3)
        desired[3*self.manip_arm+2] += 0.4*self.CUBE_SIZE

        err = desired - current

        # Lower force of manip arm
        err[3*self.manip_arm:3*self.manip_arm + 3] *= 0.5

        # Read Tip Force
        tip_forces = observation["observation"]["tip_force"]
        switch = True
        for f in tip_forces:
            if f < 0.051:
                switch = False

        # Override with small diff
        diff = observation["observation"]["position"] - self.previous_state
        self.previous_state = observation["observation"]["position"]

        if np.amax(diff) < 5e-5:
            switch = True

        if switch:
            self.pregoal_state = observation["achieved_goal"]["position"]
            print("PRE GOAL")
            self.state = States.GOAL

        return 0.1 * err

    def pregoal(self, observation):
        # Return torque for goal step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)
        # Lower force of manip arm
        into_err[3*self.manip_arm:3*self.manip_arm + 3] *= 0

        goal = self.pregoal_state
        goal[2] = 3 * self.CUBE_SIZE
        goal = np.tile(goal, 3)
        goal_err = goal - desired
        goal_err[3*self.manip_arm:3*self.manip_arm + 3] *= 0

        rot_err = np.zeros(9)
        rot_err[3*self.manip_arm:3*self.manip_arm +
                3] = observation["achieved_goal"]["position"] + np.array([0, 0, 1.5 * self.CUBE_SIZE])
        rot_err[3*self.manip_arm:3*self.manip_arm +
                3] -= current[3*self.manip_arm:3*self.manip_arm + 3]

        # Once manip arm is overhead, drop
        diff = np.linalg.norm(
            current[3*self.manip_arm:3*self.manip_arm+2] - observation["achieved_goal"]["position"][:2])
        #print("Diff: " + str(diff))

        #print("End condition: " + str(diff < 0.75 * self.CUBE_SIZE))
        if diff < 0.5 * self.CUBE_SIZE:
            print("PRE ORIENT")
            self.state = States.ORIENT
        # Once high enough, drop
        # if observation["achieved_goal"]["position"][2] > 2 * self.CUBE_SIZE:
        #    print("PRE ORIENT")
        #    self.state = States.ORIENT

        # Override with no force on manip arm
        #tip_forces = self.finger._get_latest_observation().tip_force
        # for f in tip_forces:
        #    if f < 0.051:
        #        print("PRE ORIENT")
        #        self.state = States.ORIENT

        # Override with small diff
        #diff = observation["observation"]["position"] - self.previous_state
        #self.previous_state = observation["observation"]["position"]

        # if np.amax(diff) < 1e-6:
        #    switch = True

        return 0.05 * into_err + 0.1 * goal_err + 0.25 * rot_err

    def preorient(self, observation):
        # Return torque for into step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)

        err = current - desired

        # Read Tip Force
        tip_forces = observation["observation"]["tip_force"]
        switch = False
        for f in tip_forces:
            if f < 0.1:
                switch = True
        if switch:
            self.manip_angle -= 90
            print("MANIP DONE")
            self.state = States.ALIGN

        return 0.1 * err

    def premanip(self, observation):
        force = np.zeros(9)

        if self.state == States.ALIGN:
            force = self.prealign(observation)

        elif self.state == States.LOWER:
            force = self.prelower(observation)

        elif self.state == States.INTO:
            force = self.preinto(observation)

        elif self.state == States.GOAL:
            force = self.pregoal(observation)

        elif self.state == States.ORIENT:
            force = self.preorient(observation)

        if self.manip_angle == 0:
            self.do_premanip = False
            self.state = States.ALIGN

        return force

    def align(self, observation):
        # Return torque for align step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3) + \
            self.CUBE_SIZE * \
            np.array([0, 1.6, 2, 1.6 * 0.866, 1.6 * (-0.5),
                      2, 1.6 * (-0.866), 1.6 * (-0.5), 2])

        print ("[ALIGN] current_pose: ", current)
        print ("[ALIGN] desired_pose: ", desired)

        err = desired - current
        if np.linalg.norm(err) < 2 * self.EPS:
            self.state = States.LOWER
        return 0.1 * err

    def lower(self, observation):
        # Return torque for lower step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3) + \
            self.CUBE_SIZE * \
            np.array([0, 1.6, 0, 1.6 * 0.866, 1.6 * (-0.5),
                      0, 1.6 * (-0.866), 1.6 * (-0.5), 0])

        err = desired - current
        if np.linalg.norm(err) < 2 * self.EPS:
            self.state = States.INTO
        return 0.1 * err

    def into(self, observation):
        # Return torque for into step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["desired_goal"]["position"], 3)

        err = desired - current

        # Read Tip Force
        tip_forces = observation["observation"]["tip_force"]
        switch = True
        for f in tip_forces:
            if f < 0.0515:
                switch = False
        if switch:
            self.state = States.GOAL

        self.goal_err_sum = np.zeros(9)
        return 0.1 * err

    def goal(self, observation):
        # Return torque for goal step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)

        goal = np.tile(observation["desired_goal"]["position"], 3)
        if self.difficulty == 1:
            goal[2] += 0.001  # Reduces friction with floor
        goal_err = goal - desired
        err_mag = np.linalg.norm(goal_err[:3])

        if err_mag < 0.1:
            self.goal_err_sum += goal_err

        if err_mag < 0.01 and self.difficulty == 4:
            self.state = States.ORIENT
        return 0.04 * into_err + 0.11 * goal_err + 0.0004 * self.goal_err_sum

    def orient(self, observation):
        # Return torque for lower step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)

        goal = np.tile(observation["desired_goal"]["position"], 3)
        goal_err = goal - desired
        err_mag = np.linalg.norm(goal_err[:3])

        if err_mag < 0.1:
            self.goal_err_sum += goal_err

        angle, axis = _get_angle_axis(
            observation["achieved_goal"]["orientation"], observation["desired_goal"]["orientation"])
        ang_err = np.zeros(9)
        ang_err[:3] = -angle * \
            np.cross(into_err[:3] / np.linalg.norm(into_err[:3]), axis)
        ang_err[3:6] = -angle * \
            np.cross(into_err[3:6] / np.linalg.norm(into_err[3:6]), axis)
        ang_err[6:] = -angle * \
            np.cross(into_err[6:] / np.linalg.norm(into_err[6:]), axis)

        return 0.04 * into_err + 0.11 * goal_err + 0.0004 * self.goal_err_sum + 0.006 * ang_err

    def predict(self, observation):
        # Get Jacobians
        J = self._get_jacobians(observation)
        self.t += 1

        force = np.zeros(9)

        if self.do_premanip:
            print ("do premanip")
            force = self.premanip(observation)

        elif self.state == States.ALIGN:
            print ("do align")
            force = self.align(observation)

        elif self.state == States.LOWER:
            print ("do lower")
            force = self.lower(observation)

        elif self.state == States.INTO:
            print ("do into")
            force = self.into(observation)

        elif self.state == States.GOAL:
            print ("do goal")
            force = self.goal(observation)

        elif self.state == States.ORIENT:
            print ("do orient")
            force = self.orient(observation)

        torque = J.T.dot(np.linalg.solve(
            J.dot(J.T) + self.DAMP * np.eye(9), force))

        ret = torque + self._get_gravcomp(observation)
        return ret


# Number of actions in one episode (1000 actions per second for two minutes)
episode_length = 2 * 60 * 1000


def main():
    # the difficulty level and the goal pose (as JSON string) are passed as
    # arguments
    difficulty = int(sys.argv[1])
    goal_pose_json = sys.argv[2]
    goal = json.loads(goal_pose_json)
    print(
        "Goal: %s/%s (difficulty: %d)"
        % (goal["position"], goal["orientation"], difficulty)
    )

    # initialize cube env
    env = cube_env.RealRobotCubeEnv(
        goal, difficulty, cube_env.ActionType.TORQUE, frameskip=1
    )
    observation = env.reset()

    # initialize policy object
    policy = StateSpacePolicy(env, difficulty, observation)
    accumulated_reward = 0.
    is_done = False

    ctr = 0
    position_up = [0.5, 1.2, -2.4] * 3
    action = robot_interfaces.trifinger.Action(position=position_up)
    for _ in range(500):
        t = env.platform.append_desired_action(action)
        env.platform.wait_until_timeindex(t)

        # make sure to not exceed the number of allowed actions
        if t >= episode_length - 1:
            return

    while not is_done:
        # ctr += 1
        # if ctr > 12000:
        #     break
        action = policy.predict(observation)
        # debug
        # print ("Tip forces: ", observation["observation"]["tip_force"])

        observation, reward, is_done, info = env.step(action)
        # print("reward:", reward)
        # is_done = False
        accumulated_reward += reward

    print("------")
    print("Accumulated Reward: {:.3f}".format(accumulated_reward))


if __name__ == "__main__":
    main()