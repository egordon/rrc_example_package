#!/usr/bin/env python3
"""State Machine Policy"""
import numpy as np

from statemachine import StateMachine, State
from .utils import get_rest_arm

_CUBOID_WIDTH = 0.02
_CUBOID_HEIGHT = 0.08

def get_tip_poses(observation):
    return observation["observation"]["tip_positions"].flatten()

class RRCMachine(StateMachine):
    reset = State('RESET', initial=True)
    align = State('ALIGN')

    # restart = reset.to(reset)
    start = reset.to(align)
    aligning = align.to(align)

    def on_enter_reset(self):
        print('Entering RESET!')
    
    def on_enter_align(self):
        print('Entering ALIGN!')

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
        if np.linalg.norm(err) < 0.02: # TODO: Remove Magic Number
            # Prevent Further k_p increases
            if self.root.ctr > 1:
                print("Reached RESET Position")
            self.root.ctr = 0
            self.root.k_p = 0.4
            self.machine.start()

        # Simple P-controller
        return self.root.k_p * err

    def align(self, observation):
        # Get rest arm
        # Align the other two arms around cuboid on opposite directions
        current = get_tip_poses(observation)
        current_pos = observation["achieved_goal"]["position"]
        rest_arm = get_rest_arm(observation)

        # Determine arm locations
        locs = [np.zeros(3), np.zeros(3), np.zeros(3)]

        for i in range(3):
            index = (rest_arm + 1 - i) % 3
            locs[index] = 1.5 * \
                R.from_rotvec(
                    np.pi/2 * (i-1.0) * np.array([0, 0, 1])).apply(rest_arm)
            locs[index][2] = 2

        desired = np.tile(current_pos, 3) + \
            (self.root.CUBOID_HEIGHT + 0.015) * np.hstack(locs)

        up_position = np.array([0.5, 1.2, -2.4] * 3)
        upward_desired = np.array(
            self.finger.pinocchio_utils.forward_kinematics(up_position)).flatten()

        desired[rest_arm * 3: (rest_arm + 1) * 3] = upward_desired[rest_arm * 3: (rest_arm + 1) * 3]

        err = desired - current
        if np.linalg.norm(err) < 0.01:
            print ("Reached ALIGN state")
            print("[ALIGN]: K_p ", self.k_p)
            self.root.k_p = 0.4
            self.ctr = 0

        return self.root.k_p * err

    def predict(self, observation):
        force = np.zeros(9)

        if self.machine.is_reset:
            force = self.reset(observation)
        if self.machine.is_align:
            force = self.align(observation)

        return force