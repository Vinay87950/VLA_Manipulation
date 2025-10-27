import numpy as np
from stl_files_tic_tac import TicTacToeEnv  # Import your custom class

# Instantiate the environment directly
env = TicTacToeEnv(
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    horizon=3000,
)

env.reset()
for i in range(3000):
    action = np.random.uniform(env.action_spec[0], env.action_spec[1])  # Random actions for testing
    obs, reward, done, info = env.step(action)
    env.render()  # Visualize the simulation
    if done:
        env.reset()