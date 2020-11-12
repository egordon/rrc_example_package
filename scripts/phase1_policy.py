#!/usr/bin/env python3
"""Simple example on how to move the robot."""
import enum
import json
import sys
import time
from collections import deque

import numpy as np
import pybullet
import robot_fingers
import robot_interfaces
from rrc_example_package import cube_env
from scipy.spatial.transform import Rotation as R
from trifinger_simulation import trifingerpro_limits
from trifinger_simulation.tasks import move_cube

from utils import pitch_orient

class States(enum.Enum):
    """ Different States for StateSpacePolicy """

    RESET = enum.auto()

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
        self.state = States.RESET
        self.difficulty = difficulty

        if self.difficulty == 4:
            self.EPS = 5e-3
        else:
            self.EPS = 2e-2

        self.DAMP = 1E-6
        self.CUBE_SIZE = 0.0325

        self.do_premanip = False

        self.t = 0
        self.last_reset_error = 0.
        self.iterm_reset = 0.
        self.finger = env.sim_platform.simfinger
        self.iterm_align = 0.
        self.last_align_error = 0.
        self.k_p = 1.1
        self.ctr = 0
        self.force_offset = None
        self.interval = 100
        self.gain_increase_factor = 1.2
        self.start_time = None
        self.goal_begin_time = None
        self.goal_reached = False
        # to avoid completion because of error in cube position
        self.success_ctr = 0
        self.cube_position = deque(maxlen=100)
        self.cube_orient = deque(maxlen=100)
        self.pregoal_reached = False
        self.pregoal_begin_time = None
        # orient vars
        self.manip_angle = None
        self.manip_arm = None
        self.manip_axis = None

    def _get_gravcomp(self, observation):
        # Returns: 9 torques required for grav comp
        ret2 = pybullet.calculateInverseDynamics(self.finger.finger_id,
                                                 observation["observation"]["position"].tolist(
                                                 ),
                                                 observation["observation"]["velocity"].tolist(
                                                 ),
                                                 np.zeros(
                                                     len(observation["observation"]["position"])).tolist(),
                                                 self.finger._pybullet_client_id)

        # ret = pybullet.calculateInverseDynamics(self.finger.finger_id,
        #                                         observation["observation"]["position"].tolist(
        #                                         ),
        #                                         observation["observation"]["velocity"].tolist(),
        #                                         np.zeros(len(observation["observation"]["position"])).tolist())
        ret = np.array(ret2)
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
                np.zeros(len(observation["observation"]["position"])).tolist(),
                self.finger._pybullet_client_id
            )
            ret.append(J)
        ret = np.vstack(ret)
        return ret

    def _get_tip_poses(self, observation):
        return observation["observation"]["tip_positions"].flatten()

    def prealign(self, observation):
        # get mean cube pose
        self.cube_position.append(observation["achieved_goal"]["position"])
        self.cube_orient.append(observation["achieved_goal"]["orientation"])
        curr_cube_position = np.mean(np.array(self.cube_position), axis=0)
        curr_cube_position[2] = self.CUBE_SIZE
        curr_cube_orient = np.mean(np.array(self.cube_orient), axis=0)

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

        desired = np.tile(curr_cube_position, 3) + \
            (self.CUBE_SIZE + 0.015) * np.hstack(locs)

        # desired[self.manip_arm * 3: (self.manip_arm + 1)*3] = [0.5, 1.2, -2.4]

        err = desired - current
        if np.linalg.norm(err) < 0.01:
            self.state = States.LOWER
            print("[ALIGN]: Switching to PRE LOWER")
            print("[ALIGN]: K_p ", self.k_p)
            print("[ALIGN]: Cube pos ", curr_cube_position)
            self.k_p = 1.2
            self.ctr = 0

        return self.k_p * err

    def prelower(self, observation):
        self.cube_position.append(observation["achieved_goal"]["position"])
        self.cube_orient.append(observation["achieved_goal"]["orientation"])
        curr_cube_position = np.mean(np.array(self.cube_position), axis=0)
        curr_cube_position[2] = self.CUBE_SIZE
        curr_cube_orient = np.mean(np.array(self.cube_orient), axis=0)

        # Return torque for align step
        current = self._get_tip_poses(observation)

        # Determine arm locations
        locs = [np.zeros(3), np.zeros(3), np.zeros(3)]

        for i in range(3):
            index = (self.manip_arm + 1 - i) % 3
            locs[index] = 1.5 * \
                R.from_rotvec(
                    np.pi/2 * i * np.array([0, 0, 1])).apply(self.manip_axis)

        locs[self.manip_arm][2] -= 1.0

        desired = np.tile(curr_cube_position, 3) + \
            self.CUBE_SIZE * np.hstack(locs)

        err = desired - current
        if np.linalg.norm(err) < 0.02:
            self.previous_state = observation["observation"]["position"]
            self.state = States.INTO
            print("[LOWER]: Switching to PRE INTO")
            print("[LOWER]: K_p ", self.k_p)
            print("[LOWER]: Cube pos ", curr_cube_position)
            print("[LOWER]: Current Tip Forces ",
                  observation["observation"]["tip_force"])
            self.k_p = 1.2
            self.ctr = 0
            self.cube_position.clear()
            self.cube_orient.clear()
        return self.k_p * err

    def preinto(self, observation):
        # Return torque for into step
        current = self._get_tip_poses(observation)
        current_x = current[0::3]
        difference = [abs(p1 - p2)
                      for p1 in current_x for p2 in current_x if p1 != p2]
        # print ("TIP diff: ", difference)
        k_p = min(15.0, self.k_p)
        if any(y < 0.0001 for y in difference):
            self.state = States.RESET
            print("[INTO]: Switching to RESET")
            print("[INTO]: K_p ", self.k_p)
            print("[INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.5
            self.ctr = 0

        x, y, z = observation["achieved_goal"]["position"]
        z = self.CUBE_SIZE
        desired = np.tile([x, y, z], 3)
        desired[3*self.manip_arm+2] += 0.4*self.CUBE_SIZE

        err = desired - current

        # Lower force of manip arm
        err[3*self.manip_arm:3*self.manip_arm + 3] *= 0.2

        # Read Tip Force
        tip_forces = observation["observation"]["tip_force"] - \
            self.force_offset
        switch = True
        for f in tip_forces:
            if f < 0.1:
                switch = False

        # Override with small diff
        diff = observation["observation"]["position"] - self.previous_state
        self.previous_state = observation["observation"]["position"]

        if np.amax(diff) < 5e-5:
            switch = True

        if switch:
            self.pregoal_state = observation["achieved_goal"]["position"]
            self.state = States.GOAL
            print("[INTO] Tip Forces ", observation["observation"]["tip_force"])
            print("[INTO]: Switching to PRE GOAL")
            print("[INTO]: K_p ", self.k_p)
            print("[INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.65
            self.ctr = 0
            # self.do_premanip = False
            self.interval = 250

        self.goal_err_sum = np.zeros(9)
        return self.k_p * err

    def pregoal(self, observation):
        if self.pregoal_begin_time is None:
            self.pregoal_begin_time = time.time()

        # Return torque for goal step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)
        k_p = min(5.0, self.k_p)

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)
        # Lower force of manip arm
        into_err[3*self.manip_arm:3*self.manip_arm + 3] *= 0

        goal = np.array(self.pregoal_state)
        goal[2] = 3 * self.CUBE_SIZE
        goal = np.tile(goal, 3)
        goal_err = goal - desired
        goal_err[3*self.manip_arm:3*self.manip_arm + 3] *= 0

        err_mag = np.linalg.norm(goal_err[:3])
        if err_mag < 0.1:
            self.goal_err_sum += goal_err

        rot_err = np.zeros(9)
        rot_err[3*self.manip_arm:3*self.manip_arm +
                3] = observation["achieved_goal"]["position"] + np.array([0, 0, 1.5 * self.CUBE_SIZE])
        rot_err[3*self.manip_arm:3*self.manip_arm +
                3] -= current[3*self.manip_arm:3*self.manip_arm + 3]

        # Once manip arm is overhead, drop
        diff = np.linalg.norm(
            current[3*self.manip_arm:3*self.manip_arm+2] - observation["achieved_goal"]["position"][:2])
        #print("Diff: " + str(diff))

        print("[GOAL] Rot err magnitude ", np.linalg.norm(rot_err), " Diff", diff, " K_p ",
              self.k_p, " time: ", time.time() - self.start_time, " orient: ", observation["achieved_goal"]["orientation"])

        if not self.pregoal_reached and time.time() - self.pregoal_begin_time > 20.0:
            self.state = States.RESET
            time.sleep(3.0)
            print("[GOAL]: Switching to RESET")
            print("[GOAL]: K_p ", self.k_p)
            print("[GOAL]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.5
            self.interval = 100
            self.gain_increase_factor = 1.2
            self.ctr = 0
            self.pregoal_begin_time = None
            self.do_premanip = False

        #print("End condition: " + str(diff < 0.75 * self.CUBE_SIZE))
        # TODO: tweak the factor here
        factor = 0.7  # 0.5 previously
        if diff < factor * self.CUBE_SIZE:
            # print("PRE ORIENT")
            self.state = States.ORIENT
            print("[GOAL]: Switching to PRE ORIENT")
            print("[GOAL]: K_p ", self.k_p)
            print("[GOAL]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.3
            self.interval = 500
            self.ctr = 0

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

        return 0.15 * into_err + k_p * goal_err + 0.25 * rot_err #+  0.0005 * self.goal_err_sum

    def preorient(self, observation):
        # Return torque for into step
        current = self._get_tip_poses(observation)
        k_p = min(self.k_p, 6.0)
        desired = np.tile(observation["achieved_goal"]["position"], 3)

        err = current - desired

        # Read Tip Force
        tip_forces = observation["observation"]["tip_force"] - \
            self.force_offset
        switch = False
        for f in tip_forces:
            if f < 0.1:
                switch = True
        if switch:
            self.manip_angle -= 90
            print("MANIP DONE")
            self.state = States.GOAL
            self.k_p = 1.2
            self.interval = 200
            self.ctr = 0

        return k_p * err

    def premanip(self, observation):
        force = np.zeros(9)

        if self.state == States.ALIGN:
            # print("do prealign")
            force = self.prealign(observation)

        elif self.state == States.LOWER:
            # print("do prelower")
            force = self.prelower(observation)

        elif self.state == States.INTO:
            # print("do preinto")
            force = self.preinto(observation)

        elif self.state == States.GOAL:
            # print("do pregoal")
            force = self.pregoal(observation)

        elif self.state == States.ORIENT:
            # print("do preorient")
            force = self.preorient(observation)

        if self.manip_angle == 0:
            # verify
            print ("Verify premanip")
            self.manip_angle, self.manip_axis, self.manip_arm = pitch_orient(observation)
            if self.manip_angle == 0:
                self.do_premanip = False
                self.state = States.ALIGN
            else:
                self.state = States.RESET

        return force

    def reset(self, observation):
        # print ("[RESET]: ON RESET")
        self.cube_position.clear()
        self.cube_orient.clear()
        current = self._get_tip_poses(observation)
        up_position = np.array([0.5, 1.2, -2.4] * 3)
        desired = np.array(
            self.finger.pinocchio_utils.forward_kinematics(up_position)).flatten()
        err = desired - current
        delta_err = err - self.last_reset_error
        if np.linalg.norm(err) < 0.02:
            if self.difficulty == 4:
                print ("[RESET] Verify premanip")
                self.manip_angle, self.manip_axis, self.manip_arm = pitch_orient(observation)
                if self.manip_angle != 0:
                    self.do_premanip = True
                else:
                    self.do_premanip = False
            else:
                self.do_premanip = False
                # self._calculate_premanip(observation)
            self.state = States.ALIGN
            print("[RESET]: Switching to ALIGN")
            print("[RESET]: K_p ", self.k_p)
            print("[RESET]: Cube pos ", observation['achieved_goal']['position'])
            self.force_offset = observation["observation"]["tip_force"]
            self.k_p = 0.8
            self.ctr = 0
            self.interval = 100

        self.last_reset_error = err
        k_i = 0.1
        self.iterm_reset += delta_err
        return self.k_p * err  # + 0.0001* delta_err + 0.16 * self.iterm_reset

    def align(self, observation):
        # get mean cube pose
        self.cube_position.append(observation["achieved_goal"]["position"])
        self.cube_orient.append(observation["achieved_goal"]["orientation"])
        curr_cube_position = np.mean(np.array(self.cube_position), axis=0)
        curr_cube_orient = np.mean(np.array(self.cube_orient), axis=0)

        # Return torque for align step
        current = self._get_tip_poses(observation)
        x, y = curr_cube_position[:2]
        z = self.CUBE_SIZE
        desired = np.tile(np.array([x, y, z]), 3) + \
            (self.CUBE_SIZE + 0.015) * \
            np.array([0, 1.6, 1.5, 1.6 * 0.866, 1.6 * (-0.5),
                      1.5, 1.6 * (-0.866), 1.6 * (-0.5), 1.5])

        err = desired - current
        # print ("[ALIGN] error: ", err)
        if np.linalg.norm(err) < self.EPS:
            self.state = States.LOWER
            print("[ALIGN]: Switching to LOWER")
            print("[ALIGN]: K_p ", self.k_p)
            print("[ALIGN]: Cube pos ", curr_cube_position,
                  " Orient: ", curr_cube_orient)
            self.k_p = 1.2
            self.ctr = 0

        delta_err = err - self.last_align_error
        self.iterm_align += delta_err
        k_i = 0.1
        self.last_align_error = err
        return self.k_p * err  # + 0.01 * self.iterm_align

    def lower(self, observation):
        self.cube_position.append(observation["achieved_goal"]["position"])
        self.cube_orient.append(observation["achieved_goal"]["orientation"])
        curr_cube_position = np.mean(np.array(self.cube_position), axis=0)
        curr_cube_orient = np.mean(np.array(self.cube_orient), axis=0)

        # Return torque for lower step
        current = self._get_tip_poses(observation)

        x, y = curr_cube_position[:2]
        z = self.CUBE_SIZE
        desired = np.tile(np.array([x, y, z]), 3) + \
            (self.CUBE_SIZE + 0.015) * \
            np.array([0, 1.6, 0.015, 1.6 * 0.866, 1.6 * (-0.5),
                      0.015, 1.6 * (-0.866), 1.6 * (-0.5), 0.015])

        err = desired - current
        if np.linalg.norm(err) < self.EPS:
            self.state = States.INTO
            print("[LOWER]: Switching to INTO")
            print("[LOWER]: K_p ", self.k_p)
            print("[LOWER]: Cube pos ", curr_cube_position)
            print("[LOWER]: Current Tip Forces ",
                  observation["observation"]["tip_force"])
            self.k_p = 1.2
            self.ctr = 0

        return self.k_p * err

    def into(self, observation):
        # Return torque for into step
        current = self._get_tip_poses(observation)
        current_x = current[0::3]
        difference = [abs(p1 - p2)
                      for p1 in current_x for p2 in current_x if p1 != p2]
        # print ("TIP diff: ", difference)
        k_p = min(15.0, self.k_p)
        if any(y < 0.0001 for y in difference):
            self.state = States.RESET
            print("[INTO]: Switching to RESET")
            print("[INTO]: K_p ", self.k_p)
            print("[INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.5
            self.ctr = 0

        # print ("Current tip pose: ", current)
        x, y = observation["achieved_goal"]["position"][:2]
        z = self.CUBE_SIZE
        desired = np.tile(np.array([x, y, z]), 3)

        err = desired - current

        # Read Tip Force
        tip_forces = observation["observation"]["tip_force"] - \
            self.force_offset
        switch = True
        for f in tip_forces:
            if f < 0.1:
                switch = False
        if switch:
            self.state = States.GOAL
            print("[INTO] Tip Forces ", observation["observation"]["tip_force"])
            print("[INTO]: Switching to GOAL")
            print("[INTO]: K_p ", self.k_p)
            print("[INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.65
            self.ctr = 0
            self.gain_increase_factor = 1.08
            self.interval = 2500

        self.goal_err_sum = np.zeros(9)
        return k_p * err

    def goal(self, observation):
        # Return torque for goal step
        if self.goal_begin_time is None:
            self.goal_begin_time = time.time()
        current = self._get_tip_poses(observation)
        current_x = current[0::3]
        difference = [abs(p1 - p2)
                      for p1 in current_x for p2 in current_x if p1 != p2]
        # print ("TIP diff: ", difference)
        # if any(y < 0.02 for y in difference):
        #     self.state = States.ALIGN
        #     return 0.0
        k_p = min(2.5, self.k_p)
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

        if not self.goal_reached and time.time() - self.goal_begin_time > 30.0:
            self.state = States.RESET
            print("[GOAL]: Switching to RESET")
            print("[GOAL]: K_p ", self.k_p)
            print("[GOAL]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.5
            self.interval = 100
            self.gain_increase_factor = 1.2
            self.ctr = 0
            self.goal_begin_time = None

        if not self.goal_reached:
            print("[GOAL] Error magnitude ", err_mag, " K_p ",
                  k_p, " time: ", time.time() - self.start_time)

        if err_mag > 0.015:
            self.goal_reached = False

        # if err_mag < 0.01 and self.difficulty == 4:
        #     self.state = States.ORIENT
        #     print("[GOAL]: Switching to ORIENT")
        #     print("[GOAL]: K_p ", self.k_p)
        #     print("[GOAL]: Cube pos ", observation['achieved_goal']['position'])
        #     self.k_p = 0.5
        #     self.ctr = 0

        if err_mag < 0.01:
            self.success_ctr += 1

        # if err_mag < 0.01 and self.success_ctr > 20 and self.difficulty in [2, 3]:
        #     self.state = States.HOLD
        #     print("[GOAL]: Goal state achieved")
        #     print("[GOAL]: Switching to HOLD")
        #     print("[GOAL]: K_p ", self.k_p)
        #     self.ctr = 0

        if not self.goal_reached and err_mag < 0.01 and self.success_ctr > 20:
            # self.state = States.ORIENT
            print("[GOAL]: Goal state achieved")
            print("[GOAL]: K_p ", self.k_p)
            self.goal_reached = True
            self.ctr = 0
            self.gain_increase_factor = 1.0

        # k_p = 0.65
        return k_p * goal_err + 0.30 * into_err + 0.002 * self.goal_err_sum

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

        if self.state == States.RESET:
            # print ("do reset")
            force = self.reset(observation)
        elif self.do_premanip:
            # print ("do premanip")
            force = self.premanip(observation)
        elif self.state == States.ALIGN:
            # print ("do align")
            force = self.align(observation)

        elif self.state == States.LOWER:
            # print ("do lower")
            force = self.lower(observation)

        elif self.state == States.INTO:
            # print ("do into")
            force = self.into(observation)

        elif self.state == States.GOAL:
            # print ("do goal")
            force = self.goal(observation)

        # elif self.state == States.ORIENT:
        #     # print ("do orient")
        #     force = self.orient(observation)

        # force = np.array([0., 0., 0.5, 0., 0., 0.5, 0., 0., 0.5])
        torque = J.T.dot(np.linalg.solve(
            J.dot(J.T) + self.DAMP * np.eye(9), force))

        ret = np.array(torque + self._get_gravcomp(observation),
                       dtype=np.float64)
        # print ("Torque value: ", ret)
        ret = np.clip(ret, -0.396, 0.396)
        # if self.state == States.ALIGN:
        #     ret = np.zeros((9,), dtype=np.float64)
        return ret


# Number of actions in one episode (1000 actions per second for two minutes)
episode_length = 2 * 60 * 1000


def main():
    # the difficulty level and the goal pose (as JSON string) are passed as
    # arguments

    # TODO: Uncomment before submission
    # difficulty = int(sys.argv[1])
    # goal_pose_json = sys.argv[2]
    # goal = json.loads(goal_pose_json)

    # TODO: Comment before submission
    difficulty = 4
    goal_pose = move_cube.sample_goal(difficulty)
    goal = {'position': goal_pose.position,
            'orientation': goal_pose.orientation}
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
    """
    TODOs
    * one of the hands should have a lower tip force
    """

    zero_torque_action = robot_interfaces.trifinger.Action()
    t = env.platform.append_desired_action(zero_torque_action)
    # env.platform.wait_until_timeindex(t)

    policy.start_time = time.time()
    while not is_done:
        ctr += 1
        if ctr % policy.interval == 0 and policy.ctr < 30:
            policy.ctr += 1
            policy.k_p *= policy.gain_increase_factor
        # if ctr % 50 == 0:
        action = policy.predict(observation)
        # action = np.zeros((9))
        observation, reward, is_done, info = env.step(action)
        # print("reward:", reward)
        accumulated_reward += reward

    print("------")
    print("Accumulated Reward: {:.3f}".format(accumulated_reward))


if __name__ == "__main__":
    main()
