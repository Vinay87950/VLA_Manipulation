import robosuite as suite
import numpy as np
from test_tic_tac import TicTacToeEnv

def explore_environment(env_name="PickPlace", robot="Panda"):
    """
    Complete script to explore and find positions in any robosuite environment
    """
    # Create environment
    env = suite.make(
        env_name=env_name,
        robots=robot,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
    )
    
    obs = env.reset()
    sim = env.sim
    
    print("\n" + "="*60)
    print(f"ENVIRONMENT: {env_name}")
    print("="*60)
    
    # 1. Print observation keys
    print("\n📊 OBSERVATION KEYS:")
    for key, value in obs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    # 2. Print all bodies
    print("\n🔲 ALL BODIES:")
    for i in range(sim.model.nbody):
        name = sim.model.body_id2name(i)
        pos = sim.data.body_xpos[i]
        print(f"  {name:30s}: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")
    
    # 3. Print all sites
    print("\n📍 ALL SITES:")
    for i in range(sim.model.nsite):
        name = sim.model.site_id2name(i)
        pos = sim.data.site_xpos[i]
        print(f"  {name:30s}: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")
    
    # 4. Calculate safe target positions
    try:
        table_id = sim.model.body_name2id("table")
        table_pos = sim.data.body_xpos[table_id]
        table_size = env.table_full_size if hasattr(env, 'table_full_size') else (0.8, 0.8, 0.05)
        table_top = table_pos[2] + table_size[2]/2
        
        print("\n🎯 SUGGESTED TARGET POSITIONS:")
        positions = {
            "Center": [table_pos[0], table_pos[1], table_top + 0.02],
            "Front": [table_pos[0] - 0.15, table_pos[1], table_top + 0.02],
            "Back": [table_pos[0] + 0.15, table_pos[1], table_top + 0.02],
            "Left": [table_pos[0], table_pos[1] - 0.15, table_top + 0.02],
            "Right": [table_pos[0], table_pos[1] + 0.15, table_top + 0.02],
        }
        for name, pos in positions.items():
            print(f"  {name:10s}: {pos}")
    except:
        print("\n⚠️ Could not find table - check body names above")
    
    # 5. Interactive viewing
    print("\n👁️ Rendering environment... (close window to exit)")
    for _ in range(2000):
        action = np.zeros(env.action_dim)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    
    env.close()

# Run the explorer
explore_environment("PickPlace", "Panda")  # Change env_name as needed