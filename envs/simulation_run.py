import imageio
import argparse
import numpy as np
import time  # Import time module for tracking real-world time
# from tic_tac import TicTacToeEnv
from test_tic_tac import TicTacToeEnv
import robosuite.macros as macros
import robosuite.utils
import robosuite as suite
from robosuite import load_composite_controller_config
import matplotlib.pyplot as plt

# Path to config file
controller_fpath = "/Users/killuaa/Desktop/mujoco_sim/self_environment/DOF/6_dof.json"
config = load_composite_controller_config(controller=controller_fpath)

macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="video.mp4")
    parser.add_argument("--timesteps", type=int, default=10000)  # Increased default (we'll break by time)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    args = parser.parse_args()

    # env = suite.make(
    #     "PickPlace",
    #     robots="Panda",
    #     controller_configs=config,
    #     has_renderer=True,
    #     use_camera_obs=True,
    #     render_camera="frontview",   
    #     render_gpu_device_id=0,
    #     camera_heights=args.height,
    #     camera_widths=args.width,
    #     renderer="mujoco",
    # )
    env = TicTacToeEnv(
        robots="Panda",
        controller_configs=config,
        has_renderer=True,
        use_camera_obs=True,
        render_camera="frontview",   
        render_gpu_device_id=0,
        camera_heights=args.height,
        camera_widths=args.width,
        renderer="mujoco",
    )

    obs = env.reset()

    total_runtime = 15.0  # 15 seconds runtime
    start_time = time.time()  # Record start time
    elapsed_time = 0.0

    def get_policy_action(obs):
        # a trained policy could be used here, but we choose a random action
        low, high = env.action_spec
        return np.random.uniform(low, high)
    
    # Main simulation loop with time limit
    for i in range(args.timesteps):
        # Check elapsed time at start of iteration
        elapsed_time = time.time() - start_time
        if elapsed_time >= total_runtime:
            print(f"\nSimulation reached {total_runtime} second time limit. Closing...")
            break
            
        # action = env.action_space.sample()
        # obs, reward, done, info = env.step(action)
        ret = 0.
        action = get_policy_action(obs)         # use observation to decide on an action
        obs, reward, done, _ = env.step(action) # play action
        ret += reward
        env.render()  # Render current frame

        if done:
            env.reset()
    
    
    # Properly close the environment and viewer
    print("Cleaning up resources...")
    env.close()  # Critical for proper shutdown
    print("Simulation closed successfully")