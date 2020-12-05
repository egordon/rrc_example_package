#!/usr/bin/env python3
"""State Machine Policy"""
import time
import numpy as np

from statemachine import StateMachine, State
from .utils import get_rest_arm, get_rest_arm2
from scipy.spatial.transform import Rotation as R


_CUBOID_WIDTH = 0.02
_CUBOID_HEIGHT = 0.08


def get_tip_poses(observation):
    return observation["observation"]["tip_positions"].flatten()


class RRCMachine(StateMachine):
    reset = State('RESET', initial=True)
    align = State('ALIGN')
    lower = State('LOWER')
    into = State('INTO')
    goal = State('GOAL')

    start = reset.to(align)
    lowering = align.to(lower)
    grasp = lower.to(into)
    move_to_goal = into.to(goal)

    recover_from_align = align.to(reset)
    recover_from_lower = lower.to(reset)
    recover_from_into = into.to(reset)
    recover_from_goal = goal.to(reset)

    def on_enter_reset(self):
        print('Entering RESET!')

    def on_enter_align(self):
        print('Entering ALIGN!')
    
    def on_enter_lower(self):
        print('Entering LOWER!')
    
    def on_enter_into(self):
        print('Entering INTO!')
    
    def on_enter_goal(self):
        print('Entering GOAL!')


class MachinePolicy:

    def __init__(self, root):
        # Root Policy: Used to reference global state
        self.root = root

        # State Machine
        self.machine = RRCMachine()

        # state begin time variables
        self.align_begin_time = None
        self.lower_begin_time = None
        self.into_begin_time = None
        self.goal_begin_time = None

    def reset(self, observation):
        self.root.cube_position.clear()
        self.root.cube_orient.clear()

        # Get Current Position
        current = observation["observation"]["tip_positions"].flatten()

        # Get Desired Reset Position
        up_position = np.array([0.5, 1.2, -2.4] * 3)
        desired = np.array(
            self.root.finger.pinocchio_utils.forward_kinematics(up_position)).flatten()

        # Calculate Error
        err = desired - current

        # Reached Goal
        if np.linalg.norm(err) < 0.02:  # TODO: Remove Magic Number
            # Prevent Further k_p increases
            if self.root.ctr > 1:
                print("Reached RESET Position")
            self.root.ctr = 0
            self.root.k_p = 0.4
            self.rest_arm, self.manip_axis = get_rest_arm2(observation)
            self.force_offset = observation["observation"]["tip_force"]
            self.machine.start()

        # Simple P-controller
        return self.root.k_p * err

    def align(self, observation):
        if self.align_begin_time is None:
            self.align_begin_time = time.time()

        if time.time() - self.lower_begin_time > 10.0:
            print("[ALIGN]: Switching to RESET at ",
                  time.time() - self.root.start_time)
            print("[ALIGN]: K_p ", self.root.k_p)
            print("[ALIGN]: Cube pos ", observation['achieved_goal']['position'])
            self.root.k_p = 0.5
            self.root.ctr = 0
            self.align_begin_time = None
            self.machine.recover_from_align()

        current = get_tip_poses(observation)
        self.root.cube_position.append(observation["achieved_goal"]["position"])
        self.root.cube_orient.append(observation["achieved_goal"]["orientation"])
        curr_cube_position = np.median(np.array(self.cube_position), axis=0)
        x, y = curr_cube_position[:2]
        current_pos = [x, y, self.root.CUBOID_WIDTH]

        # print ("current pos: ", current_pos)
        # print ("current orient: ", observation["achieved_goal"]["orientation"])

        # Determine arm locations
        locs = [np.zeros(3), np.zeros(3), np.zeros(3)]

        for i in range(3):
            index = (self.rest_arm + 1 - i) % 3
            locs[index] = 1.5 * \
                R.from_rotvec(
                    np.pi/4 * (i-1.0) * np.array([0, 0, 1])).apply(self.manip_axis)
            locs[index][2] = 2

        desired = np.tile(current_pos, 3) + \
            (self.root.CUBOID_WIDTH + 0.015) * np.hstack(locs)

        up_position = np.array([0.5, 1.2, -2.4] * 3)
        upward_desired = np.array(
            self.root.finger.pinocchio_utils.forward_kinematics(up_position)).flatten()

        desired[self.rest_arm * 3: (self.rest_arm + 1) *
                3] = upward_desired[self.rest_arm * 3: (self.rest_arm + 1) * 3]

        err = desired - current
        if np.linalg.norm(err) < 0.01:
            print("Reached ALIGN state")
            print("[ALIGN]: K_p ", self.root.k_p)
            self.root.k_p = 0.4
            self.root.ctr = 0
            self.prev_align = desired
            self.interval = 300
            self.gain_increase_factor = 1.1
            self.machine.lowering()

        return self.root.k_p * err

    def lower(self, observation):
        if self.lower_begin_time is None:
            self.lower_begin_time = time.time()
        current = get_tip_poses(observation)

        if time.time() - self.lower_begin_time > 10.0:
            print("[LOWER]: Switching to RESET at ",
                  time.time() - self.root.start_time)
            print("[LOWER]: K_p ", self.root.k_p)
            print("[LOWER]: Cube pos ", observation['achieved_goal']['position'])
            self.root.k_p = 0.5
            self.root.ctr = 0
            self.lower_begin_time = None
            self.machine.recover_from_lower()

        self.root.cube_position.append(observation["achieved_goal"]["position"])
        self.root.cube_orient.append(observation["achieved_goal"]["orientation"])
        curr_cube_position = np.median(np.array(self.cube_position), axis=0)
        x, y = curr_cube_position[:2]
        current_pos = [x, y, self.root.CUBOID_WIDTH]

        desired = np.tile(current_pos, 3) + \
            (self.root.CUBOID_WIDTH + 0.015) * \
            np.array([0, 1.6, 0.015, 1.6 * 0.866, 1.6 * (-0.5),
                      0.015, 1.6 * (-0.866), 1.6 * (-0.5), 0.015])

        # testing with align xy values
        desired = self.prev_align
        desired[2::3] = [self.root.CUBOID_WIDTH] * 3
        up_position = np.array([0.5, 1.2, -2.4] * 3)
        upward_desired = np.array(
            self.root.finger.pinocchio_utils.forward_kinematics(up_position)).flatten()

        desired[self.rest_arm * 3: (self.rest_arm + 1) *
                3] = upward_desired[self.rest_arm * 3: (self.rest_arm + 1) * 3]

        err = desired - current
        if np.linalg.norm(err) < 0.01:
            print("Reached LOWER state")
            print("[LOWER]: K_p ", self.root.k_p)
            self.root.k_p = 0.6
            self.root.ctr = 0
            self.root.gain_increase_factor = 1.2
            self.root.interval = 150
            self.machine.grasp()

        return self.root.k_p * err
    
    def into(self, observation):
        if self.into_begin_time is None:
            self.into_begin_time = time.time()
        current = get_tip_poses(observation)
        current_x = current[0::3]
        current_y = current[1::3]
        difference_x = [abs(p1 - p2)
                        for p1 in current_x for p2 in current_x if p1 != p2]
        difference_y = [abs(p1 - p2)
                        for p1 in current_y for p2 in current_y if p1 != p2]

        k_p = min(4.0, self.root.k_p)
        if self.root.difficulty == 3:
            time_threshold = 5.0  # based on experimental observation
        else:
            time_threshold = 15.0

        close_x = any(d < 0.0001 for d in difference_x)
        close_y = any(d < 0.0001 for d in difference_y)
        close = close_x and close_y

        x, y = observation["achieved_goal"]["position"][:2]
        z = self.root.CUBOID_WIDTH
        desired = np.tile(np.array([x, y, z]), 3)
        up_position = np.array([0.5, 1.2, -2.4] * 3)
        upward_desired = np.array(
            self.root.finger.pinocchio_utils.forward_kinematics(up_position)).flatten()

        desired[self.rest_arm * 3: (self.rest_arm + 1) *
                3] = upward_desired[self.rest_arm * 3: (self.rest_arm + 1) * 3]

        err = desired - current

        # Read Tip Force
        tip_forces = observation["observation"]["tip_force"] - \
            self.force_offset

        if close or time.time() - self.into_begin_time > time_threshold:
            print("[INTO]: Switching to RESET at ",
                  time.time() - self.root.start_time)
            print("[INTO]: K_p ", self.root.k_p)
            print("[INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.root.k_p = 0.5
            self.root.interval = 200
            self.root.gain_increase_factor = 1.2
            self.root.ctr = 0
            self.into_begin_time = None
            self.machine.recover_from_into()

        switch = True
        for i, f in enumerate(tip_forces):
            if f < 0.015 and i != self.rest_arm:
                switch = False
        if switch:
            print("Reached INTO state")
            print("[INTO] Tip Forces ", observation["observation"]["tip_force"])
            print("[INTO]: Switching to GOAL at ",
                  time.time() - self.root.start_time)
            print("[INTO]: K_p ", self.root.k_p)
            print("[INTO]: Cube pos ", observation['achieved_goal']['position'])
            self.root.k_p = 0.65
            self.root.ctr = 0
            self.root.gain_increase_factor = 1.04
            self.root.interval = 1000
            self.into_begin_time = None
            self.machine.move_to_goal()
            

        self.goal_err_sum = np.zeros(9)
        return self.root.k_p * err

    def goal(self, observation):
        if self.goal_begin_time is None:
            self.goal_begin_time = time.time()

        current = get_tip_poses(observation)
        current_x = current[0::3]
        difference = [abs(p1 - p2)
                      for p1 in current_x for p2 in current_x if p1 != p2]

        k_p = min(0.79, self.root.k_p)
        up_position = np.array([0.5, 1.2, -2.4] * 3)
        upward_desired = np.array(
            self.root.finger.pinocchio_utils.forward_kinematics(up_position)).flatten()

        desired = np.tile(observation["achieved_goal"]["position"], 3)
        desired[self.rest_arm * 3: (self.rest_arm + 1) *
                3] = upward_desired[self.rest_arm * 3: (self.rest_arm + 1) * 3]

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)

        goal = np.tile(observation["desired_goal"]["position"], 3)
        if self.root.difficulty == 1 and not self.root.goal_reached:
            goal[2::3] += 0.004  # Reduces friction with floor

        goal[self.rest_arm * 3: (self.rest_arm + 1) *
                3] = upward_desired[self.rest_arm * 3: (self.rest_arm + 1) * 3]

        goal_err = goal - desired
        if self.rest_arm == 0:
            err_mag = np.linalg.norm(goal_err[3:6])
        else:
            err_mag = np.linalg.norm(goal_err[:3])

        if err_mag < 0.1:
            self.goal_err_sum += goal_err

        if self.root.difficulty == 1:
            time_threshold = 40.0
        else:
            time_threshold = 30.0

        if not self.root.goal_reached and time.time() - self.goal_begin_time > time_threshold:
            print("[GOAL]: Switching to RESET at ",
                  time.time() - self.root.start_time)
            print("[GOAL]: K_p ", self.root.k_p)
            print("[GOAL]: Cube pos ", observation['achieved_goal']['position'])
            self.root.k_p = 0.5
            self.root.interval = 100
            self.root.gain_increase_factor = 1.2
            self.root.ctr = 0
            self.root.success_ctr = 0
            self.goal_begin_time = None
            self.machine.recover_from_goal()

        if not self.root.goal_reached:
            print("[GOAL] Error magnitude ", err_mag, " K_p ",
                  k_p, " time: ", time.time() - self.root.start_time)

        if err_mag > 0.015:
            self.root.goal_reached = False
            self.root.success_ctr = 0

        if err_mag < 0.01:
            self.root.success_ctr += 1

        if not self.root.goal_reached and err_mag < 0.01 and self.root.success_ctr > 50:
            print("[GOAL]: Goal state achieved")
            print("[GOAL]: K_p ", self.root.k_p)
            self.root.goal_reached = True
            # self.root.ctr = 0
            self.root.gain_increase_factor = 1.0
            # self.goal_begin_time = None

        return (k_p * goal_err + 0.35 * into_err + 0.002 * self.goal_err_sum) * 0.2


    def predict(self, observation):
        force = np.zeros(9)

        if self.machine.is_reset:
            force = self.reset(observation)
        if self.machine.is_align:
            force = self.align(observation)
        if self.machine.is_lower:
            force = self.lower(observation)
        if self.machine.is_into:
            force = self.into(observation)
        if self.machine.is_goal:
            force = self.goal(observation)

        return force
