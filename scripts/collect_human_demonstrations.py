"""
A script to collect a batch of human demonstrations for the TicTacToe environment.

The demonstrations can be played back using the standard robosuite playback scripts.
"""

import argparse
import datetime
import json
import os
import sys
import time
from glob import glob

import h5py
import numpy as np

# Add the parent directory to the path so we can import self_environment
# Assuming this script is in self_environment/scripts/
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(repo_root)
# Also add robosuite repo root to path to ensure we pick up the package if it's local
sys.path.insert(0, os.path.join(repo_root, "robosuite"))

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from self_environment.envs.test_tic_tac import TicTacToeEnv

def collect_human_trajectory(env, device, arm, max_fr):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arm (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    """

    env.reset()
    env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action()

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        
        # Check if "arm" is in partial_controllers - standard robosuite logic might overlap with how env sets up controllers
        # For Panda (single arm), 'right' is usually the key if configured as such, or just 'arm' if not bimanual wrapper?
        # Standard robosuite single-arm envs usually have robot.part_controllers['right'] (or 'left')
        
        # set arm actions
        # For our TicTacToe env, we expect a single robot typically
        
        for arm_name in active_robot.arms:
            if arm_name in action_dict:
                 # Already set?
                 pass
            
            # This logic mimics the original script's handling of specific input types
            # but we need to be careful about what 'arm' arg is doing vs what robot.arms has.
            
            # The original script iterates active_robot.arms.
            
            # Determine controller input type
            # Note: We need to handle CompositeController vs direct controller
            # But TicTacToeEnv uses standard robots which usually have CompositeController in recent robosuite
            
            # controller = active_robot.controller <-- This caused AttributeError
            
            # If composite
            if hasattr(active_robot, "composite_controller") and active_robot.composite_controller is not None:
                 # It's likely composite
                 part_controller = active_robot.part_controllers[arm_name]
                 controller_input_type = part_controller.input_type
            else:
                 # Fallback/Older robosuite or simple robot
                 controller = active_robot.controller 
                 controller_input_type = controller.input_type
                 
            print(f"DEBUG: arm={arm_name}, type={controller_input_type}, keys={list(input_ac_dict.keys())}") 
            # Uncomment the above line if needed, but for now let's print if we find the key
            
            target_key = f"{arm_name}_{controller_input_type}"
            print(f"DEBUG: Looking for {target_key} in input_ac_dict")

            if controller_input_type == "delta":
                key = f"{arm_name}_delta"
                if key in input_ac_dict:
                    action_dict[arm_name] = input_ac_dict[key]
                    # if np.linalg.norm(action_dict[arm_name]) > 0:
                    #    print(f"DEBUG: Applying non-zero action to {arm_name}: {action_dict[arm_name]}")
                else:
                    pass
                    # print(f"DEBUG: Warning {key} not found in input (available: {list(input_ac_dict.keys())})")
            elif controller_input_type == "absolute":
                key = f"{arm_name}_abs"
                if key in input_ac_dict:
                    action_dict[arm_name] = input_ac_dict[key]
            else:
                 # Try typical fallback
                 if f"{arm_name}_delta" in input_ac_dict:
                      action_dict[arm_name] = input_ac_dict[f"{arm_name}_delta"]
                 elif f"{arm_name}_abs" in input_ac_dict:
                      action_dict[arm_name] = input_ac_dict[f"{arm_name}_abs"]

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        
        # Update prev gripper actions
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            if gripper_ac in action_dict:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    """
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return 0

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            print("Demonstration is successful and has been saved")
            # Delete the last state logic from original script
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f_xml:
                xml_str = f_xml.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()
    
    return num_eps


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="collected_data",
        help="Directory to store collected data",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="BASIC",
        help="Choice of controller. Default to BASIC (which uses OSC_POSE for arms).",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mujoco",
        help="Renderer to use: 'mujoco' (default) or 'opencv'.",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of successful demonstrations to collect before stopping",
    )
    
    args = parser.parse_args()

    # Load custom controller config
    # User requested 6_dof.json
    # Try to find it relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # script is in self_environment/scripts/
    # 6_dof.json is in self_environment/DOF/
    custom_config_path = os.path.join(os.path.dirname(current_dir), "DOF", "6_dof.json")
    
    # MODIFIED: Logic to prioritize user argument or default over the hardcoded 6_dof.json for now
    # The user asked to "try the original composite_controller", so we skip the forced 6_dof loading
    # unless a specific flag or condition is met. 
    # For now, we will rely on the default behavior (fallback to args.controller) effectively invalidating this block
    # by setting a flag or just commenting out.
    
    # To enable 6_dof.json again, the user can pass --controller .../6_dof.json or we can uncomment below.
    # But wait, the user passed NO arguments in the failed runs (except defaults).
    # If I want to test "original", I should SKIP this block.
    
    load_custom_6dof = False # Set to True to force 6_dof.json
    
    if load_custom_6dof and os.path.exists(custom_config_path):
        print(f"Loading custom controller config from: {custom_config_path}")
        with open(custom_config_path, "r") as f:
            controller_config = json.load(f)
            
        # Manually flatten 'arms' key in body_parts if present, 
        # mimicking robosuite.controllers.composite_controller_factory.load_composite_controller_config
        if "body_parts" in controller_config:
            body_parts = controller_config.pop("body_parts")
            controller_config["body_parts"] = {}
            for part_name, part_config in body_parts.items():
                if part_name == "arms":
                    for arm_name, arm_config in part_config.items():
                        controller_config["body_parts"][arm_name] = arm_config
                else:
                    controller_config["body_parts"][part_name] = part_config
    else:
        # print(f"Warning: Custom config not found at {custom_config_path}. Falling back to default.")
        # Create controller config
        # We use a helper from robosuite to load default configs
        # Use load_composite_controller_config as per standard robosuite usage
        try:
            controller_config = load_composite_controller_config(
                controller=args.controller,
                robot=args.robots[0] if isinstance(args.robots, list) else args.robots
            )
        except Exception as e:
            # Fallback if load_composite fails (e.g. if we requested a PART controller name like OSC_POSE but needed a composite one)
            # Use a basic default or re-raise
            print(f"Error loading composite controller: {e}")
            raise e

    # Create argument configuration for the env
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
        "env_configuration": args.config,
    }

    # Instantiate TicTacToeEnv
    env = TicTacToeEnv(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        renderer=args.renderer,
        use_camera_obs=False, # Faster collection, no need for images yet
        reward_shaping=True,
        control_freq=20,
        use_object_obs=True,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # Create directory for data
    # Resolve relative to script execution or fixed?
    # Let's make it relative to current working directory or absolute
    if not os.path.isabs(args.directory):
        data_collected_dir = os.path.join(os.getcwd(), args.directory)
    else:
        data_collected_dir = args.directory

    os.makedirs(data_collected_dir, exist_ok=True)
    
    # Create a temporary directory inside data_collected for this collection session
    t1, t2 = str(time.time()).split(".")
    tmp_directory = os.path.join(data_collected_dir, "tmp_{}_{}".format(t1, t2))
    os.makedirs(tmp_directory, exist_ok=True)
    
    print(f"Data will be temporarily collected in: {tmp_directory}")
    
    # wrap the environment with data collection wrapper
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense
        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=False, # Default
        )
    else:
        raise Exception("Invalid device choice: choose 'keyboard', 'spacemouse', or 'dualsense'.")

    # Directory for final HDF5
    new_dir = os.path.join(data_collected_dir, "{}_{}".format(t1, t2))
    os.makedirs(new_dir, exist_ok=True)
    
    print(f"Final HDF5 file will be saved in: {new_dir}")

    # collect demonstrations
    successful_demos = 0
    print(f"\nStarting data collection. Target: {args.num_samples} successful demonstrations")
    print("Press standard keyboard controls (e.g., arrow keys, w/s/a/d, space to grasp) to control the robot.")
    
    while successful_demos < args.num_samples:
        collect_human_trajectory(env, device, args.arm, max_fr=20)
        num_saved = gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
        successful_demos = num_saved
        print(f"\nProgress: {successful_demos}/{args.num_samples} successful demonstrations collected")
        
        if successful_demos >= args.num_samples:
            print(f"\n✓ Target reached! {successful_demos} successful demonstrations collected.")
            print(f"Data saved in: {new_dir}/demo.hdf5")
            break
    
    print("\nData collection completed!")
