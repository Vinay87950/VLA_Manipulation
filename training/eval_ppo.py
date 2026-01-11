
import os
import sys
import numpy as np
import time

# --- Path Setup ---
# Add parent directory to path to allow importing self_environment
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

# Add local robosuite to path to use the custom source version
robosuite_path = os.path.join(root_dir, "robosuite")
sys.path.insert(0, robosuite_path)

from robosuite.controllers import load_composite_controller_config
# Load Controller Config (Must match training!)
controller_fpath = os.path.join(root_dir, "self_environment/DOF/6_dof.json")
config = load_composite_controller_config(controller=controller_fpath)

try:
    from stable_baselines3 import PPO
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
    env = TicTacToeEnv(
        robots=["Panda"],
        has_renderer=True,           # ENABLE RENDERER
        render_camera="frontview",   # Optional: "frontview", "agentview"
        controller_configs=config,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        reward_shaping=True,
    )
    
    # Wrap environment exactly as in training
    env = GymWrapper(env, flatten_obs=True)

    print("Starting Evaluation...")
    print("Press Ctrl+C to stop.")

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"--- Episode {episode + 1} ---")
        
        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step env
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            # Render is handled by the env's internal viewer since has_renderer=True
            env.render()
            
            # Slow down slightly for viewing pleasure if needed (Robosuite viewer usually handles sync)
            # time.sleep(0.01) 
            
        print(f"Episode finished. Steps: {step}, Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate()
