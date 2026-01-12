
import os
import sys
import numpy as np
import time

# --- Path Setup ---
# Add parent directory to path to allow importing self_environment
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

# # Add local robosuite to path to use the custom source version
# robosuite_path = os.path.join(root_dir, "robosuite")
# sys.path.insert(0, robosuite_path)

from robosuite.controllers import load_composite_controller_config
# Load Controller Config (Must match training!)
controller_fpath = os.path.join(root_dir, "self_environment/DOF/6_dof.json")
config = load_composite_controller_config(controller=controller_fpath)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from robosuite.wrappers.gym_wrapper import GymWrapper
    from self_environment.envs.test_tic_tac import TicTacToeEnv
except ImportError as e:
    print(f"Error importing libraries: {e}")
    sys.exit(1)

# --- Configuration ---
MODEL_PATH = os.path.join(current_dir, "models/tic_tac_ppo_final.zip")
NUM_EPISODES = 5

def evaluate():
    # Use a local variable to avoid UnboundLocalError
    model_path = MODEL_PATH
    
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print("Model file not found!")
        # Try finding a checkpoint if final model is missing
        model_dir = os.path.join(current_dir, "models")
        if os.path.exists(model_dir):
            files = sorted([f for f in os.listdir(model_dir) if f.endswith(".zip")])
            if files:
                model_path = os.path.join(model_dir, files[-1])
                print(f"Falling back to latest checkpoint: {model_path}")
            else:
                print("No model found.")
                return

    # Load Model
    model = PPO.load(model_path)

    # Create Environment (With Renderer!)
    # We need to recreate the exact GymWrapper -> DummyVecEnv -> VecNormalize stack
    
    def make_eval_env():
        env = TicTacToeEnv(
            robots=["Panda"],
            has_renderer=True,           # ENABLE RENDERER
            render_camera="robot0_eye_in_hand",   # Optional: "frontview", "agentview"
            controller_configs=config,
            use_camera_obs=False,
            use_object_obs=True,
            control_freq=20,
            horizon=500,
            ignore_done=False,
            reward_shaping=True,
        )
        return GymWrapper(env, flatten_obs=True)

    # 1. Create VecEnv
    # We create a dummy env to inspect/set render_mode if needed
    dummy_env = make_eval_env()
    dummy_env.render_mode = "human" # Fixes the warning
    env = DummyVecEnv([lambda: dummy_env])

    # 2. Load Normalization Stats (If available)
    # The stats file usually has the same name style but with another extension or suffix
    # In train_ppo.py we explicitly named it:
    norm_path = model_path.replace(".zip", "_vecnormalize.pkl")
    
    if os.path.exists(norm_path):
        print(f"Loading normalization stats from: {norm_path}")
        env = VecNormalize.load(norm_path, env)
        # IMPORTANT: Disable training mode during evaluation and reward normalization
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: No normalization stats found. Running without normalization (performance might be poor if model was trained with it).")

    print("Starting Evaluation...")
    print("Press Ctrl+C to stop.")

    for episode in range(NUM_EPISODES):
        obs = env.reset() # VecEnv reset returns just obs
        done = False
        total_reward = 0
        step = 0
        
        print(f"--- Episode {episode + 1} ---")
        
        # VecEnv loop usually handles things differently (array of obs), but with DummyVecEnv size 1:
        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step env
            # VecEnv returns: obs, rewards, dones, infos
            obs, reward, dones, infos = env.step(action)
            
            # Since we have 1 env, extract scalars
            done = dones[0]
            total_reward += reward[0]
            step += 1
            
            # Render
            env.render()
            
        print(f"Episode finished. Steps: {step}, Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate()
