import imageio
import argparse
import numpy as np
from stl_files_tic_tac import TicTacToeEnv
import robosuite.macros as macros
# import robosuite as suite
from robosuite import load_composite_controller_config
import matplotlib.pyplot as plt

'''
To define the json file for DOF the idea is taken from 'https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/controllers/config/robots/default_panda.json'
Define End effector for testing 'https://me336.ancorasir.com/?page_id=584'
'''

# Path to config file
controller_fpath = "/Users/killuaa/Desktop/mujoco_sim/self_environment/dof.json"

# Import the file as a dict
config = load_composite_controller_config(controller=controller_fpath)


macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--environment", type=str, default="TicTacToeEnv")
    # parser.add_argument("--robots", nargs="+", type=str, default="Panda")
    parser.add_argument("--video_path", type=str, default="video.mp4")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    args = parser.parse_args()



# Camera options for switching
    camera_cycle = ["frontview", "agentview", "robot0_eye_in_hand"]
    current_camera_idx = 0

    env = TicTacToeEnv(
        robots="Panda",
        controller_configs=config,
        has_renderer=True,
        use_camera_obs=True,
        camera_names=camera_cycle,         
        render_camera=camera_cycle[current_camera_idx],
        render_gpu_device_id=0,
        camera_heights=args.height,
        camera_widths=args.width,
    )

    env.reset()

    ori_pos = []

    def PD_Controller(target, eef_pos, last_eff_pos):
        p = 8
        d = 7
        action = np.zeros(4)
        target = ori_pos + target
        action[0:3] = (target - eef_pos) * p - abs(eef_pos - last_eff_pos) * d
        return action


    action = np.zeros(4)
    eef_pos = np.zeros(3)
    last_eff_pos = np.zeros(3)

    obs, reward, done, info = env.step(action)
    eef_pos = obs['robot0_eef_pos'] # position of end effector
    ori_pos = eef_pos # original position of end effector
    last_eff_pos = eef_pos

    track = []
    step = np.pi / 500
    r = 0.3
    for i in range(1000):
        circle = [0, r * np.sin(step * i), r - r * np.cos(step * i)]
        eef_pos = obs['robot0_eef_pos']
        action = PD_Controller(circle, eef_pos, last_eff_pos)
        track = np.append(track, eef_pos - ori_pos)
        obs, reward, done, info = env.step(action)
        last_eff_pos = eef_pos
        env.render()

    env.close()
        # plot the track
    track = track.reshape([1000, 3])
    plt.figure(0)
    plt.scatter(track[:, 1], track[:, 2])
    plt.show()