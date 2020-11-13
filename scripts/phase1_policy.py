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

from utils import pitch_orient, _get_angle_axis_top_only, _get_yaw_err

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

class StateSpacePolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, env, difficulty, observation):
        self.action_space = env.action_space
        self.state = States.RESET
        self.difficulty = difficulty

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
        self.k_p = 0.4
        self.ctr = 0
        self.force_offset = None
        self.interval = 100
        self.gain_increase_factor = 1.2
        self.start_time = None
        self.goal_begin_time = None
        self.goal_reached = False
        # to avoid completion because of error in cube position
        self.success_ctr = 0
        self.success_ctr_pitch_orient = 0
        self.cube_position = deque(maxlen=100)
        self.cube_orient = deque(maxlen=100)
        self.pregoal_reached = False
        self.pregoal_begin_time = None
        self.preinto_begin_time = None
        self.into_begin_time = None
        self.align_begin_time = None
        self.lower_begin_time = None
        # orient vars
        self.manip_angle = None
        self.manip_arm = None
        self.manip_axis = None

        # Maximum number of premanip steps
        self.num_premanip = 2

        # Maximum number of yaw steps
        self.num_yaw = 1
        self.do_yaw = False
        self.yawgoal_begin_time = None

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
                    np.pi/2 * (i-1.0) * np.array([0, 0, 1])).apply(self.manip_axis)
            locs[index][2] = 2

        desired = np.tile(curr_cube_position, 3) + \
            (self.CUBE_SIZE + 0.015) * np.hstack(locs)

        err = desired - current
        if np.linalg.norm(err) < 0.01:
            self.state = States.LOWER
            print("[PRE ALIGN]: Switching to PRE LOWER at ", time.time() - self.start_time)
            print("[PRE ALIGN]: K_p ", self.k_p)
            print("[PRE ALIGN]: Cube pos ", curr_cube_position)
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
                    np.pi/2 * (i-1.0) * np.array([0, 0, 1])).apply(self.manip_axis)

        desired = np.tile(curr_cube_position, 3) + \
            self.CUBE_SIZE * np.hstack(locs)
        
        desired[3*self.manip_arm: 3*self.manip_arm + 2] -= 0.4*self.CUBE_SIZE

        err = desired - current
        if np.linalg.norm(err) < 0.02:
            self.previous_state = observation["observation"]["position"]
            self.state = States.INTO
            print("[PRE LOWER]: Switching to PRE INTO at ", time.time() - self.start_time)
            print("[PRE LOWER]: K_p ", self.k_p)
            print("[PRE LOWER]: Cube pos ", curr_cube_position)
            print("[PRE LOWER]: Current Tip Forces ",
                  observation["observation"]["tip_force"])
            self.k_p = 1.0
            self.interval = 400
            self.ctr = 0
            self.cube_position.clear()
            self.cube_orient.clear()
        return self.k_p * err

    def preinto(self, observation):
        if self.preinto_begin_time is None:
            self.preinto_begin_time = time.time()

        # Return torque for into step
        current = self._get_tip_poses(observation)
        current_x = current[0::3]
        difference = [abs(p1 - p2)
                      for p1 in current_x for p2 in current_x if p1 != p2]

        k_p = min(6.0, self.k_p)

        x, y = observation["achieved_goal"]["position"][:2]
        z = self.CUBE_SIZE
        desired = np.tile(np.array([x, y, z]), 3)

        err = desired - current

        # Read Tip Force
        tip_forces = observation["observation"]["tip_force"] - \
            self.force_offset
        switch = True
        for i, f in enumerate(tip_forces):
            if i == self.manip_arm:
                continue
            if f < 0.07:
                switch = False
        if switch:
            self.pregoal_state = observation["achieved_goal"]["position"]
            self.state = States.GOAL
            print("[PRE INTO] Tip Forces ", observation["observation"]["tip_force"])
            print("[PRE INTO]: Switching to PRE GOAL at ", time.time() - self.start_time)
            print("[PRE INTO]: K_p ", self.k_p)
            print("[PRE INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.65
            self.ctr = 0
            self.gain_increase_factor = 1.08
            self.preinto_begin_time = None
            self.interval = 1500
        elif time.time() - self.preinto_begin_time > 15.0:
            self.state = States.RESET
            print("[PRE INTO]: Switching to RESET at ", time.time() - self.start_time)
            print("[PRE INTO]: K_p ", self.k_p)
            print("[PRE INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.5
            self.interval = 100
            self.gain_increase_factor = 1.2
            self.ctr = 0
            self.preinto_begin_time = None
            self.do_premanip = False

        self.goal_err_sum = np.zeros(9)
        return self.k_p * err

    def pregoal(self, observation):
        if self.pregoal_begin_time is None:
            self.pregoal_begin_time = time.time()

        # Return torque for goal step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)
        k_p = min(2.0, self.k_p)

        # Keep arms pushing into cube
        into_err = desired - current
        into_err /= np.linalg.norm(into_err)

        # Move cube up above ground
        goal = np.array(self.pregoal_state)
        goal[2] = 3 * self.CUBE_SIZE
        goal = np.tile(goal, 3)
        goal_err = goal - desired

        # Apply torque for manip arm
        inward = into_err[3*self.manip_arm:3*self.manip_arm + 3] / np.linalg.norm(into_err[3*self.manip_arm:3*self.manip_arm + 3])
        axis = np.cross(inward, np.array([0, 0, 1])) # Normalized axis of rotation
        rot_err = np.zeros(9)

        # Command no rotation if close
        if np.linalg.norm(axis) > 1E-8:
            axis = axis / np.linalg.norm(axis)
            rot_err[3*self.manip_arm:3*self.manip_arm + 3] = np.cross(axis, inward)

        # Once manip arm is overhead, drop
        diff = np.linalg.norm(
            current[3*self.manip_arm:3*self.manip_arm+2] - observation["achieved_goal"]["position"][:2])

        if not self.pregoal_reached and time.time() - self.pregoal_begin_time > 20.0:
            self.state = States.RESET
            print("[PRE GOAL]: Switching to RESET at ", time.time() - self.start_time)
            print("[PRE GOAL]: K_p ", self.k_p)
            print("[PRE GOAL]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.5
            self.interval = 100
            self.gain_increase_factor = 1.2
            self.ctr = 0
            self.pregoal_begin_time = None
            self.do_premanip = False

        print("[PRE GOAL] Goal err Diff", diff, " K_p ",
              self.k_p, " time: ", time.time() - self.start_time, " orient: ", observation["achieved_goal"]["orientation"])

        # TODO: tweak the factor here
        factor = 0.7  # 0.5 previously
        if diff < factor * self.CUBE_SIZE:
            self.success_ctr_pitch_orient += 1
        if diff < factor * self.CUBE_SIZE and self.success_ctr_pitch_orient > 20:
            # print("PRE ORIENT")
            self.state = States.ORIENT
            print("[PRE GOAL]: Switching to PRE ORIENT at ", time.time() - self.start_time)
            print("[PRE GOAL]: K_p ", self.k_p)
            print("[PRE GOAL]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.3
            self.interval = 1000
            self.ctr = 0

        return 0.14 * into_err + k_p * goal_err + 0.25 * rot_err

    def preorient(self, observation):
        # Return torque for into step
        current = self._get_tip_poses(observation)
        k_p = min(self.k_p, 3.0)
        desired = np.tile(observation["achieved_goal"]["position"], 3)

        # Flip, move outwards
        err = current - desired

        # Read Tip Force
        tip_forces = observation["observation"]["tip_force"] - \
            self.force_offset
        switch = False
        for f in tip_forces:
            if f < 0.1:
                switch = True

        if switch:
            print("[PRE ORIENT] PRE MANIP DONE")
            if self.num_premanip > 0:
                self.num_premanip = self.num_premanip - 1
            self.state = States.RESET
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
            if self.difficulty == 4 and self.num_premanip > 0:
                time.sleep(4.0)
                print ("[RESET] Verify premanip")
                self.manip_angle, self.manip_axis, self.manip_arm = pitch_orient(observation)
                if self.manip_angle != 0:
                    self.do_premanip = True
                else:
                    self.do_premanip = False
            else:
                self.do_premanip = False
            if self.difficulty == 4 and not self.do_premanip and self.num_yaw > 0:
                yaw_err = _get_yaw_err(observation["achieved_goal"]["orientation"], observation["desired_goal"]["orientation"])
                if yaw_err > 0.5:
                    self.do_yaw = True
                else:
                    self.do_yaw = False
                    self.num_yaw = 0
            else:
                self.do_yaw = False
            self.state = States.ALIGN
            print("[RESET]: Switching to ALIGN at ", time.time() - self.start_time)
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
        if self.align_begin_time is None:
            self.align_begin_time = time.time()
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
            print("[ALIGN]: Switching to LOWER at ", time.time() - self.start_time)
            print("[ALIGN]: K_p ", self.k_p)
            print("[ALIGN]: Cube pos ", curr_cube_position)
            self.k_p = 0.7
            self.ctr = 0
            self.align_begin_time = None
        elif time.time() - self.align_begin_time > 15.0:
            self.state = States.RESET
            print("[INTO]: Switching to RESET at ", time.time() - self.start_time)
            print("[INTO]: K_p ", self.k_p)
            print("[INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.5
            self.ctr = 0
            self.align_begin_time = None

        delta_err = err - self.last_align_error
        self.iterm_align += delta_err
        k_i = 0.1
        self.last_align_error = err
        return self.k_p * err  # + 0.01 * self.iterm_align

    def lower(self, observation):
        if self.lower_begin_time is None:
            self.lower_begin_time = time.time()

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
            print("[LOWER]: Switching to INTO at ", time.time() - self.start_time)
            print("[LOWER]: K_p ", self.k_p)
            print("[LOWER]: Cube pos ", curr_cube_position)
            print("[LOWER]: Current Tip Forces ",
                  observation["observation"]["tip_force"])
            self.k_p = 0.7
            self.ctr = 0
            self.lower_begin_time = None
        elif time.time() - self.lower_begin_time > 15.0:
            self.state = States.RESET
            print("[INTO]: Switching to RESET at ", time.time() - self.start_time)
            print("[INTO]: K_p ", self.k_p)
            print("[INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.5
            self.ctr = 0
            self.lower_begin_time = None

        return self.k_p * err

    def into(self, observation):
        if self.into_begin_time is None:
            self.into_begin_time = time.time()

        # Return torque for into step
        current = self._get_tip_poses(observation)
        current_x = current[0::3]
        difference = [abs(p1 - p2)
                      for p1 in current_x for p2 in current_x if p1 != p2]
        # print ("TIP diff: ", difference)
        k_p = min(15.0, self.k_p)
        if any(y < 0.0001 for y in difference) or time.time() - self.into_begin_time > 15.0:
            self.state = States.RESET
            print("[INTO]: Switching to RESET at ", time.time() - self.start_time)
            print("[INTO]: K_p ", self.k_p)
            print("[INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.5
            self.ctr = 0
            self.into_begin_time = None

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
            if f < 0.08:
                switch = False
        if switch:
            self.state = States.GOAL
            print("[INTO] Tip Forces ", observation["observation"]["tip_force"])
            print("[INTO]: Switching to GOAL at ", time.time() - self.start_time)
            print("[INTO]: K_p ", self.k_p)
            print("[INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.65
            self.ctr = 0
            self.gain_increase_factor = 1.04
            self.interval = 1800

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

        if self.difficulty == 1:
            k_p = min(0.76, self.k_p)
        else:
            k_p = min(0.79, self.k_p)
        desired = np.tile(observation["achieved_goal"]["position"], 3)

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)

        goal = np.tile(observation["desired_goal"]["position"], 3)
        if self.difficulty == 1:
            goal[2] += 0.002  # Reduces friction with floor
        goal_err = goal - desired
        err_mag = np.linalg.norm(goal_err[:3])

        if err_mag < 0.1:
            self.goal_err_sum += goal_err

        if self.difficulty == 1:
            time_threshold = 20.0
        else:
            time_threshold = 30.0
        if not self.goal_reached and time.time() - self.goal_begin_time > time_threshold:
            self.state = States.RESET
            print("[GOAL]: Switching to RESET at ", time.time() - self.start_time)
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

        if err_mag < 0.01:
            self.success_ctr += 1

        if not self.goal_reached and err_mag < 0.01 and self.success_ctr > 20:
            print("[GOAL]: Goal state achieved")
            print("[GOAL]: K_p ", self.k_p)
            self.goal_reached = True
            self.ctr = 0
            self.gain_increase_factor = 1.0

        #if self.goal_reached and self.difficulty == 4:
        #    self.state = States.ORIENT
        #    print("[GOAL]: Switching to ORIENT at ", time.time() - self.start_time)
        #    self.ctr = 0
        #    self.goal_reached = False
        #    self.goal_begin_time = None

        return k_p * goal_err + 0.25 * into_err + 0.002 * self.goal_err_sum

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

        angle, axis = _get_angle_axis_top_only(
            observation["achieved_goal"]["orientation"], observation["desired_goal"]["orientation"])
        ang_err = np.zeros(9)
        ang_err[:3] = -angle * \
            np.cross(into_err[:3] / np.linalg.norm(into_err[:3]), axis)
        ang_err[3:6] = -angle * \
            np.cross(into_err[3:6] / np.linalg.norm(into_err[3:6]), axis)
        ang_err[6:] = -angle * \
            np.cross(into_err[6:] / np.linalg.norm(into_err[6:]), axis)

        return 0.25 * into_err + self.k_p * goal_err + 0.1 * ang_err

    def yawgoal(self, observation):
        if self.yawgoal_begin_time is None:
            self.yawgoal_begin_time = time.time()
        # Return torque for lower step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)

        angle = _get_yaw_err(
            observation["achieved_goal"]["orientation"], observation["desired_goal"]["orientation"])
        axis = np.array([0, 0, 1])

        ang_err = np.zeros(9)
        ang_err[:3] = -angle * \
            np.cross(into_err[:3] / np.linalg.norm(into_err[:3]), axis)
        ang_err[3:6] = -angle * \
            np.cross(into_err[3:6] / np.linalg.norm(into_err[3:6]), axis)
        ang_err[6:] = -angle * \
            np.cross(into_err[6:] / np.linalg.norm(into_err[6:]), axis)

        if angle < 0.1 or time.time() - self.yawgoal_begin_time > 5.0:
            self.state = States.ORIENT
            print("[YAW GOAL]: Switching to ORIENT at ", time.time() - self.start_time)
            print("[PRE GOAL]: K_p ", self.k_p)
            print("[PRE GOAL]: Cube pos ", observation['achieved_goal']['position'])
            self.k_p = 0.3
            self.interval = 1000
            self.ctr = 0
            self.yawgoal_begin_time = None
            self.num_yaw = self.num_yaw - 1

        return 0.25 * into_err + 0.1 * ang_err

    def yawmanip(self, observation):
        force = np.zeros(9)

        if self.state == States.ALIGN:
            # print("do prealign")
            force = self.align(observation)

        elif self.state == States.LOWER:
            # print("do prelower")
            force = self.lower(observation)

        elif self.state == States.INTO:
            # print("do preinto")
            force = self.into(observation)

        elif self.state == States.GOAL:
            # print("do pregoal")
            force = self.yawgoal(observation)

        elif self.state == States.ORIENT:
            # print("do preorient")
            force = self.preorient(observation)

        return force

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
        elif self.do_yaw:
            # print ("do premanip")
            force = self.yawmanip(observation)
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
        if ctr % policy.interval == 0 and policy.ctr < 20:
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
