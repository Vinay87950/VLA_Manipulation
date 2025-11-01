import imageio
import argparse
import numpy as np
from stl_files_tic_tac import TicTacToeEnv
import robosuite.macros as macros


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
        has_renderer=True,
        use_camera_obs=True,
        camera_names=camera_cycle,         
        render_camera=camera_cycle[current_camera_idx],
        render_gpu_device_id=0,
        camera_heights=args.height,
        camera_widths=args.width,
    )

    env.reset()

    # Function to handle 'v' key press for camera switch
    def on_key_press(key, mods=None):
        global current_camera_idx
        if key == ord('v'):
            current_camera_idx = (current_camera_idx + 1) % len(camera_cycle)
            # Update environment camera
            env.viewer.set_camera(camera_id=env.sim.model.camera_name2id(camera_cycle[current_camera_idx]))

    # Attach keypress handler to viewer 
    env.viewer.add_keypress_callback(on_key_press)

    writer = imageio.get_writer(args.video_path, fps=20)
    frames = []

     # Main simulation loop
    for i in range(args.timesteps):
        action = np.random.uniform(env.action_spec[0], env.action_spec[1])
        obs, reward, done, info = env.step(action)
        env.render()  # Remove camera_name argument

        '''uncomment below to save videos'''
        # # Grab the frame from the camera that is currently being viewed
        # if i % args.skip_frame == 0:  # Use the skip_frame argument
            
        #     # Get the name of the camera currently shown in the viewer
        #     current_camera_name = camera_cycle[current_camera_idx]
            
        #     # Get the corresponding image from the observation dictionary
        #     #  (Note: robosuite adds "_image" to the camera name in the obs dict)
        #     frame = obs[current_camera_name + "_image"] 
            
        #     # Add the frame to the video
        #     writer.append_data(frame)
        

        if done:
            env.reset()

    # writer.close()
    # print(f"Video saved to {args.video_path}")