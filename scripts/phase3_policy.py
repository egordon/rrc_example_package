#!/usr/bin/env python3
"""Simple example on how to move the robot."""
import json
import sys
import time

import numpy as np
import pybullet
import robot_interfaces
from rrc_example_package import cube_env
from trifinger_simulation.tasks import move_cube

from machines.level1 import MachinePolicy as Level1Machine
from machines.level3 import MachinePolicy as Level3Machine
from machines.level4 import MachinePolicy as Level4Machine

class StateSpacePolicy:
    """Policy references one of many sub-state machines."""

    def __init__(self, env, difficulty, observation):
        self.machine = None
        self.difficulty = difficulty
        self.finger = env.sim_platform.simfinger
        self.DAMP = 1E-6
        self.gain_increase_factor = 1.2

        # Cuboid dimensions
        self.CUBOID_HEIGHT = 0.04 # full height 0.08
        self.CUBOID_WIDTH = 0.01 # full width 0.02

        # Variables Accessible to Sub-Machines
        self.t = 0
        self.k_p = 0.4
        self.ctr = 0
        self.interval = 100
        self.start_time = None

        # Set Submachine
        if difficulty == 1:
            # Level 1 Difficulty
            self.machine = Level1Machine(self)
        elif difficulty == 2 or difficulty == 3:
            # Level 2/3 Difficulty
            self.machine = Level3Machine(self)
        elif difficulty == 4:
            # Level 4 Difficulty
            self.machine = Level4Machine(self)

    def _get_gravcomp(self, observation):
        # Returns: 9 torques required for grav comp
        ret = pybullet.calculateInverseDynamics(self.finger.finger_id,
                                                 observation["observation"]["position"].tolist(
                                                 ),
                                                 observation["observation"]["velocity"].tolist(
                                                 ),
                                                 np.zeros(
                                                     len(observation["observation"]["position"])).tolist(),
                                                 self.finger._pybullet_client_id)

        return np.array(ret)

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

    def predict(self, observation):
        # Get Jacobians
        J = self._get_jacobians(observation)
        self.t += 1

        # Get force from state machine
        force = np.zeros(9)
        if self.machine is not None:
            force = self.machine.predict(observation)

        # Convert requested force to torque
        torque = J.T.dot(np.linalg.solve(
            J.dot(J.T) + self.DAMP * np.eye(9), force))

        # Add Gravity Compensation
        ret = np.array(torque + self._get_gravcomp(observation),
                       dtype=np.float64)

        # Clip to torque limits
        ret = np.clip(ret, -0.396, 0.396)
        return ret


# Number of actions in one episode (1000 actions per second for two minutes)
episode_length = 2 * 60 * 1000


def main():
    # the difficulty level and the goal pose (as JSON string) are passed as
    # arguments

    # UPDATE: Change goal.json to update difficulty / fixed goal
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

    zero_torque_action = robot_interfaces.trifinger.Action()
    t = env.platform.append_desired_action(zero_torque_action)

    policy.start_time = time.time()
    while not is_done:
        ctr += 1
        if ctr % policy.interval == 0 and policy.ctr < 20:
            policy.ctr += 1
            policy.k_p *= policy.gain_increase_factor
        action = policy.predict(observation)
        try:
            observation, reward, is_done, info = env.step(action)
        except RuntimeError:
            print("WARN: RuntimeError in env.step, skipping step")
            reward = -1
        accumulated_reward += reward

    print("------")
    print("Accumulated Reward: {:.3f}".format(accumulated_reward))


if __name__ == "__main__":
    main()
