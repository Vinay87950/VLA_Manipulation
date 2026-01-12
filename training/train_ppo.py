
import os
import sys
import numpy as np
import gymnasium as gym

# --- Path Setup ---
# Add parent directory to path to allow importing self_environment
# Assumes this script is in self_environment/training/
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

# Add local robosuite to path to use the custom source version
robosuite_path = os.path.join(root_dir, "robosuite")
sys.path.insert(0, robosuite_path)
from robosuite.controllers import load_composite_controller_config
controller_fpath = "/Users/killuaa/Desktop/mujoco_sim/self_environment/DOF/6_dof.json"
config = load_composite_controller_config(controller=controller_fpath)
# Check imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed
except ImportError:
    print("Error: stable-baselines3 is not installed. Please run: pip install stable-baselines3")
    sys.exit(1)

try:
    from robosuite.wrappers.gym_wrapper import GymWrapper
    from self_environment.envs.test_tic_tac import TicTacToeEnv
except ImportError as e:
    print(f"Error importing environment or robosuite: {e}")
    sys.exit(1)

# --- Configuration ---
TRAIN_TIMESTEPS = 100_000
SAVE_FREQ = 10_000
SEED = 42
NUM_ENVS = 1 # Set to >1 for parallel training (Requires SubprocVecEnv, which is faster)
LOG_DIR = os.path.join(current_dir, "logs")
MODEL_DIR = os.path.join(current_dir, "models")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        # 1. Create Robosuite Env
        env = TicTacToeEnv(
            robots=["Panda"],
            has_renderer=False,  
            controller_configs=config,
            use_camera_obs=False,        # State-based training is faster
            use_object_obs=True,
            control_freq=20,
            horizon=500,                 # Max steps per episode
            ignore_done=False,           # Allow episodes to finish
            reward_shaping=True,         # Dense rewards for learning
        )
        
        # 2. Wrap with GymWrapper
        # flatten_obs=True creates a Box observation space (good for PPO MlpPolicy)
        # keys=None defaults to ["robot0_proprio-state", "object-state"]
        env = GymWrapper(env, flatten_obs=True)
        
        # 3. Monitor wrapper for internal SB3 logging
        env = Monitor(env, os.path.join(LOG_DIR, str(rank)))
        
        # 4. Seed
        env.reset(seed=seed + rank)
        return env
        
    return _init

if __name__ == "__main__":
    print("--- Starting Training Setup ---")
    set_random_seed(SEED)
    
    # Create Vectorized Environment
    vec_env_cls = SubprocVecEnv if NUM_ENVS > 1 else DummyVecEnv
    env = vec_env_cls([make_env(i, SEED) for i in range(NUM_ENVS)])
    
    # Normalize observations and rewards (Crucial for PPO performance)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Initialize PPO
    # MlpPolicy is standard for state-based observations
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu",
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_DIR,
        name_prefix="tic_tac_ppo"
    )
    
    print(f"Training for {TRAIN_TIMESTEPS} timesteps...")
    try:
        model.learn(
            total_timesteps=TRAIN_TIMESTEPS, 
            callback=checkpoint_callback,
            progress_bar=True
        )
        print("Training Complete!")
        
        # Save Final Model
        final_path = os.path.join(MODEL_DIR, "tic_tac_ppo_final")
        model.save(final_path)
        print(f"Model saved to {final_path}")
        
        # Save Normalization Stats
        final_norm_path = os.path.join(MODEL_DIR, "tic_tac_ppo_final_vecnormalize.pkl")
        env.save(final_norm_path)
        print(f"Normalization stats saved to {final_norm_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving current model...")
        model.save(os.path.join(MODEL_DIR, "tic_tac_ppo_interrupted"))
        env.save(os.path.join(MODEL_DIR, "tic_tac_ppo_interrupted_vecnormalize.pkl"))

