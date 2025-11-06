"""
A simple script to teleoperate the robot in the TicTacToe environment
using the keyboard.

Press 'Tab' to cycle through camera views.
"""

import time
import numpy as np
from copy import deepcopy

from tic_tac import TicTacToeEnv
from robosuite.devices import Keyboard
from robosuite.wrappers import VisualizationWrapper

# Environment configuration
env = TicTacToeEnv(
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,          # Not needed for teleop
    control_freq=20,
    renderer="mujoco",             # Use the native mujoco renderer
    render_camera="agentview",     # Start with a wide view
    horizon=3000,
)

# Wrap the environment with a visualization wrapper
env = VisualizationWrapper(env, indicator_configs=None)

# Initialize the keyboard device
device = Keyboard(
    env=env,
    pos_sensitivity=1.0,
    rot_sensitivity=1.0,
)

# Add the keyboard callback to the viewer
env.viewer.add_keypress_callback(device.on_press)

# Main teleoperation loop
while True:
    # Reset the environment
    obs = env.reset()

    # Setup rendering
    env.render()

    # Initialize device control
    device.start_control()

    # Store previous gripper action
    prev_gripper_action = {
        f"{robot_arm}_gripper": np.repeat([0], env.robots[0].gripper[robot_arm].dof)
        for robot_arm in env.robots[0].arms
        if env.robots[0].gripper[robot_arm].dof > 0
    }

    # Loop until we get a reset from the input
    while True:
        # Get the newest action from the keyboard
        input_ac_dict = device.input2action()

        # If action is none, then this is a reset, so we should break
        if input_ac_dict is None:
            break

        # Create a copy to modify
        action_dict = deepcopy(input_ac_dict)

        # Set arm actions from the device's delta values
        for arm in env.robots[0].arms:
            action_dict[arm] = input_ac_dict[f"{arm}_delta"]

        # Create the full action vector, maintaining previous gripper state
        env_action = env.robots[0].create_action_vector(action_dict)
        
        # Update the stored gripper action
        for gripper_ac in prev_gripper_action:
            prev_gripper_action[gripper_ac] = action_dict[gripper_ac]

        # Step the simulation
        env.step(env_action)
        env.render()

env.close()