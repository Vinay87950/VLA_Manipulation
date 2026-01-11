import numpy as np
import robosuite as suite
from robosuite.devices import Keyboard
import imageio
import argparse
import numpy as np
from test_tic_tac import TicTacToeEnv
# from tic_tac import TicTacToeEnv
import robosuite.macros as macros
# from robosuite.utils import CameraModder
import robosuite.utils
import robosuite as suite
from robosuite import load_composite_controller_config
import matplotlib.pyplot as plt

# /Users/killuaa/Desktop/mujoco_sim/self_environment/6_dof.json

# Import the file as a dict
# config = load_composite_controller_config(controller=controller_fpath)

def test_teleop():
    # 1. Setup the Controller (OSC_POSE is best for keyboard control)
    config = load_composite_controller_config(controller="BASIC")

    # 2. Start the Environment
    # Replace 'TicTacToeEnv' with the actual name of your class
    env = TicTacToeEnv(
        robots="Panda",
        gripper_types="PandaGripper",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        controller_configs=config,
        render_camera="frontview",  # 'agentview' or 'frontview'
    )

    env.reset()

    # 3. Initialize the Keyboard Device
    # We create the device and tell it to control the robot
    device = Keyboard(env=env, pos_sensitivity=0.05, rot_sensitivity=1.0)
    device.start_control()

    # 4. Print the "Cheat Sheet" based on the code you found
    print("\n" + "="*40)
    print(" ROBOT TELEOP STARTED")
    print("="*40)
    print(" MOVEMENT (Position):")
    print("   ↑ (Up Arrow)    : Move Backward (-X)")
    print("   ↓ (Down Arrow)  : Move Forward  (+X)")
    print("   ← (Left Arrow)  : Move Left     (-Y)")
    print("   → (Right Arrow) : Move Right    (+Y)")
    print("   . (Period)      : Move Down     (-Z)")
    print("   ; (Semicolon)   : Move Up       (+Z)")
    print("-" * 40)
    print(" ROTATION (Orientation):")
    print("   O / P           : Rotate Wrist (Yaw)")
    print("   E / R           : Rotate Roll")
    print("-" * 40)
    print(" ACTIONS:")
    print("   SPACE           : Open/Close Gripper")
    print("   Q               : Reset Simulation")
    print("="*40 + "\n")

    # 5. The Control Loop
    done = False
    while not done:
        # Get the state from the keyboard device
        state = device.get_controller_state()
        
        # Stop if user pressed 'q'
        if state["reset"]:
            env.reset()
            device._reset_internal_state()
            continue

        # Extract movement (dpos) and rotation (rotation)
        dpos = state["dpos"]
        rotation = state["rotation"]
        
        # Extract Grasp (Spacebar toggles this)
        # Note: The device returns 0 or 1. We map this to Robosuite's -1 (open) to 1 (closed)
        grasp = 1.0 if state["grasp"] else -1.0

        # Create the action vector: [x, y, z, ax, ay, az, gripper]
        # We flatten the rotation matrix to a simplified representation if needed, 
        # but OSC_POSE handles rotation matrices or axis-angle. 
        # The device.rotation returns a matrix. We can usually pass the relative change or absolute.
        
        # Robosuite's OSC_POSE controller usually expects: [x, y, z, ax, ay, az, gripper]
        # But the Keyboard device gives us an absolute rotation matrix.
        # To keep it simple for this test, we will just use the position deltas 
        # and a fixed rotation (pointing down) unless you press rotation keys.
        
        # Construct Action
        # We need to convert the 3x3 rotation matrix from the keyboard to an axis-angle or quaternion 
        # depending on controller config. 
        # HOWEVER, the easiest way for OSC_POSE in a loop is often just sending the delta position.
        
        # Let's clean up the action construction for OSC_POSE:
        action = np.concatenate([dpos, [0, 0, 0], [grasp]])
        
        # If user pressed rotation keys, 'raw_drotation' will have values
        if np.any(state["raw_drotation"]):
             action[3:6] = state["raw_drotation"]

        # Step the environment
        obs, reward, done, info = env.step(action)
        env.render()

if __name__ == "__main__":
    test_teleop()