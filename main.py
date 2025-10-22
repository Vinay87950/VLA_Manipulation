import numpy as np
from tic_tac import TicTacToe  # Import your custom class

# Instantiate the environment directly
env = TicTacToe(
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    horizon=1000,
)

env.reset()
for i in range(1000):
    action = np.random.uniform(env.action_spec[0], env.action_spec[1])  # Random actions for testing
    obs, reward, done, info = env.step(action)
    env.render()  # Visualize the simulation
    if done:
        env.reset()