
import numpy as np
import sys
import os

# Add parent directory to path to allow importing self_environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# Add local robosuite to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../robosuite")))

try:
    from self_environment.envs.test_tic_tac import TicTacToeEnv
    
    # Initialize Environment
    # Assuming "Panda" robot exists in robosuite registry or passed directly if needed
    # Using 'Panda' as a safe default for robosuite envs
    env = TicTacToeEnv(
        robots=["Panda"],
        has_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        ignore_done=True
    )
    
    print("Environment initialized successfully.")
    
    # Test Reset Randomization
    print("\n--- Testing Randomization ---")
    positions = []
    for i in range(3):
        env.reset()
        # Check first X piece position
        # accessing internal piece list
        p0 = env.x_pieces[0]
        # getting body id
        bid = env.sim.model.body_name2id(p0.root_body)
        pos = env.sim.data.body_xpos[bid]
        print(f"Reset {i}: X-Piece-0 Position = {pos}")
        positions.append(pos)
        
    if not np.allclose(positions[0], positions[1]):
        print("SUCCESS: Positions vary between resets.")
    else:
        print("WARNING: Positions identical between resets (might be deterministic or sampler fail).")

    # Test Staged Rewards
    print("\n--- Testing Staged Rewards ---")
    staged = env.staged_rewards()
    print(f"Staged Rewards (Initial): {staged}")
    # Tuple: reach, grasp, lift, transport, place
    
    # Sanity check reward computation
    total_reward = env.reward()
    print(f"Total Reward (Initial): {total_reward}")
    
except Exception as e:
    print(f"FAILED with error: {e}")
    import traceback
    traceback.print_exc()
