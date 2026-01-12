"""
Debug script to identify observation space composition.
Run on BOTH machines (server and local) and compare output.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

# robosuite_path = os.path.join(root_dir, "robosuite")
# sys.path.insert(0, robosuite_path)

from robosuite.controllers import load_composite_controller_config

# Use the SAME path as your training script
controller_fpath = os.path.join(root_dir, "self_environment/DOF/6_dof.json")
print(f"Controller config path: {controller_fpath}")
print(f"Controller config exists: {os.path.exists(controller_fpath)}")

config = load_composite_controller_config(controller=controller_fpath)

from robosuite.wrappers.gym_wrapper import GymWrapper
from self_environment.envs.test_tic_tac import TicTacToeEnv

# Create environment exactly like training
env = TicTacToeEnv(
    robots=["Panda"],
    has_renderer=False,
    controller_configs=config,
    use_camera_obs=False,
    use_object_obs=True,
    control_freq=20,
    horizon=500,
    ignore_done=False,
    reward_shaping=True,
)

print("\n=== RAW ROBOSUITE OBSERVATION KEYS ===")
obs = env.reset()
for key, value in obs.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: type={type(value)}")

# Wrap with GymWrapper
gym_env = GymWrapper(env, flatten_obs=True)

print(f"\n=== GYMWRAPPER OBSERVATION SPACE ===")
print(f"Total flattened shape: {gym_env.observation_space.shape}")

# Get observation and check
obs = gym_env.reset()
print(f"Actual observation shape: {obs.shape}")

# Print the keys that GymWrapper uses
print(f"\n=== GYMWRAPPER KEYS USED ===")
print(f"Keys: {gym_env.keys}")

env.close()
