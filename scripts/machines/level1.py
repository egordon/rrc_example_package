#!/usr/bin/env python3
"""State Machine Policy"""
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

    # restart = reset.to(reset)
    start = reset.to(align)
    lowering = align.to(lower)
    recover = lower.to(reset)

    def on_enter_reset(self):
        print('Entering RESET!')

    def on_enter_align(self):
        print('Entering ALIGN!')
    
    def on_enter_lower(self):
        print('Entering LOWER!')


class MachinePolicy:

    def __init__(self, root):
        # Root Policy: Used to reference global state
        self.root = root

        # State Machine
        self.machine = RRCMachine()

    def reset(self, observation):
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
            self.machine.start()

        # Simple P-controller
        return self.root.k_p * err

    def align(self, observation):
        # Get rest arm
        # Align the other two arms around cuboid on opposite directions
        current = get_tip_poses(observation)
        current_pos = observation["achieved_goal"]["position"]
        print ("current pos: ", current_pos)
        print ("current orient: ", observation["achieved_goal"]["orientation"])

        # Determine arm locations
        locs = [np.zeros(3), np.zeros(3), np.zeros(3)]

        for i in range(3):
            index = (self.rest_arm + 1 - i) % 3
            locs[index] = 1.7 * \
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
            self.machine.lowering()

        return self.root.k_p * err

    def lower(self, observation):
        current = get_tip_poses(observation)

        x, y = observation["achieved_goal"]["position"][:2]
        z = self.root.CUBOID_WIDTH

        desired = np.tile(np.array([x, y, z]), 3) + \
            (self.root.CUBOID_WIDTH + 0.015) * \
            np.array([0, 1.6, 0.015, 1.6 * 0.866, 1.6 * (-0.5),
                      0.015, 1.6 * (-0.866), 1.6 * (-0.5), 0.015])
        desired = self.prev_align
        desired[2::3] = [self.root.CUBOID_WIDTH] * 3
        up_position = np.array([0.5, 1.2, -2.4] * 3)
        upward_desired = np.array(
            self.root.finger.pinocchio_utils.forward_kinematics(up_position)).flatten()

        desired[self.rest_arm * 3: (self.rest_arm + 1) *
                3] = upward_desired[self.rest_arm * 3: (self.rest_arm + 1) * 3]

        err = desired - current
        if np.linalg.norm(err) < 0.01:
            print("[LOWER]: K_p ", self.root.k_p)
            self.root.k_p = 0.4
            self.root.ctr = 0
            self.machine.recover()

        return self.root.k_p * err

    def predict(self, observation):
        force = np.zeros(9)

        if self.machine.is_reset:
            force = self.reset(observation)
        if self.machine.is_align:
            force = self.align(observation)
        if self.machine.is_lower:
            force = self.lower(observation)

        return force
