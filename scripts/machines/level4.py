#!/usr/bin/env python3
"""State Machine Policy"""
import numpy as np

from statemachine import StateMachine, State

class RRCMachine(StateMachine):
    reset = State('RESET', initial=True)

    restart = reset.to(reset)

    def on_enter_reset(self):
        print('Entering RESET!')

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
            root.finger.pinocchio_utils.forward_kinematics(up_position)).flatten()

        # Calculate Error
        err = desired - current

        # Reached Goal
        if np.linalg.norm(err) < 0.02: # TODO: Remove Magic Number
            # Prevent Further k_p increases
            if root.ctr > 1:
                print("Reached RESET Position")
            root.ctr = 0

        # Simple P-controller
        return self.k_p * err

    def predict(self, observation):
        force = np.zeros(9)

        if self.machine.is_reset:
            force = self.reset()

        return force