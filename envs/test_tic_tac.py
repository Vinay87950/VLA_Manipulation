import numpy as np
from gymnasium.spaces import Box
import os
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
import robosuite.utils.transform_utils as T

class TicTacToeEnv(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.6, 0.6, 0.05), 
        table_friction=(1.0, 5e-3, 1e-4),
        z_offset=0.0,
        z_rotation=None,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=0,
        object_type=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=0,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="frontview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        seed=None,
    ):
        # task settings
        self.single_object_mode = single_object_mode
        self.object_to_id = {"cube": 0, "cylinder": 1} 
        self.object_id_to_sensors = {}
        self.obj_names = ["Cube", "Cylinder"]
 
        # Validate input or set default
        if object_type is None:
            object_type = "cube"
            
        if object_type is not None:
            assert object_type in self.object_to_id, "Invalid object type- choose one of the following {}".format(
                list(self.object_to_id.keys())
            )
        self.object_id = self.object_to_id[object_type]
        
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        #setting the table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        self.z_offset = z_offset
        self.z_rotation = z_rotation
        
        #table settings for 2 tables
        self.board_table_offset = np.array([0.4, 0.3, 0.8]) 
        self.piece_table_offset = np.array([0.4, -0.3, 0.8])
        
        # reward settings
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.grid_centers = []
        
        # placement initializer
        self.placement_initializer = placement_initializer


        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

        self.unplaced_x = list(range(5))  # Indices of available X pieces
        self.unplaced_o = list(range(5))  # For O
        self.current_player = 1  # Start with agent (X)
        self.timestep = 0

    def reward(self, action=None):
        """
        Reward function for the task.
        """
        r_reach, r_orient, r_open, r_grasp, r_lift, r_transport, r_place = self.staged_rewards()
        reward = r_reach + r_orient + r_open + r_grasp + r_lift + r_transport + r_place
        
        if self.reward_scale:
            reward *= self.reward_scale / 5.0
            
        return reward

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states.
        stages consist of reaching, grasping, lifting, and placing.

        Returns:
            7-tuple:
                - (float) reaching reward
                - (float) orientation reward
                - (float) gripper open reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) transport reward
                - (float) placing reward
        """
        
        reach_mult = 0.3
        orient_mult = 0.1
        open_mult = 0.1
        grasp_mult = 0.5
        lift_mult = 1.0
        transport_mult = 1.0
        place_mult = 2.0
        
        r_reach = 0.0
        r_orient = 0.0
        r_open = 0.0
        r_grasp = 0.0
        r_lift = 0.0
        r_transport = 0.0
        r_place = 0.0

        current_piece = self._get_current_piece()
        if current_piece is None:
            return r_reach, r_orient, r_open, r_grasp, r_lift, r_transport, r_place

        gripper = self._get_gripper()

        # STAGE 1: Reaching
        reach_dist = self._gripper_to_target(
            gripper=gripper,
            target=current_piece.root_body,
            target_type="body",
            return_distance=True
        )
        # Relaxed scaling (5.0 instead of 10.0)
        r_reach = (1 - np.tanh(5.0 * reach_dist)) * reach_mult

        # STAGE 1.5: Orientation (Encourage gripper to point down)
        # Get gripper Z axis in world frame
        # We assume "down" is -Z (0, 0, -1)
        gripper_site_id = self.sim.model.site_name2id(gripper.important_sites["grip_site"])
        gripper_mat = self.sim.data.site_xmat[gripper_site_id].reshape(3, 3)
        gripper_z = gripper_mat[:, 2] # Z-axis of gripper
        # Dot product with world -Z (0, 0, -1) -> this should be 1 if perfectly aligned
        z_dot = -gripper_z[2] 
        if z_dot > 0:
            r_orient = (z_dot ** 2) * orient_mult

        # STAGE 1.6: Gripper Open (Encourage opening when near but not grasping)
        if reach_dist < 0.1 and not self._check_grasp(gripper=gripper, object_geoms=current_piece):
            # Proportional to how open the fingers are
            # gripper.current_action contains the last action sent, normalized [-1, 1] usually
            # But better to check joint positions if possible. 
            # Robosuite grippers have `current_action` property which tracks input
            
            # Simple heuristic: Reward if current control input is "open" (-1)
            # Depending on gripper config, -1 might be open or closed. Usually -1 is open for Robosuite grippers (Standard).
            # Let's assume -1 is open.
            
            # Actually, let's look at joint positions for robustness if we can, but checking action is easier.
            # Giving a small reward constant if we "want" to be open
            r_open = open_mult

        # STAGE 2: Grasping
        if self._check_grasp(gripper=gripper, object_geoms=current_piece):
            r_grasp = grasp_mult

        # STAGE 3: Lifting
        piece_height = self._get_piece_height(current_piece)
        if r_grasp > 0.0:
            r_lift = min(1.0, piece_height / 0.1) * lift_mult

        # STAGE 4: Transport to Valid Grid
        if r_lift > 0.0 and piece_height > 0.05:
            # Find closest EMPTY grid center
            board_state = self._get_board_state().flatten()
            empty_indices = [i for i, state in enumerate(board_state) if state == 0]
            
            if empty_indices:
                grip_pos = self.sim.data.site_xpos[self.sim.model.site_name2id(gripper.important_sites["grip_site"])]
                
                min_dist = float("inf")
                for idx in empty_indices:
                    target = self.grid_centers[idx].copy()
                    target[2] += 0.05 # Target slightly above
                    d = np.linalg.norm(grip_pos - target)
                    if d < min_dist:
                        min_dist = d
                        
                r_transport = (1 - np.tanh(5.0 * min_dist)) * transport_mult

        # STAGE 5: Placing
        if self._piece_placed_in_valid_cell(current_piece):
            r_place = place_mult

        return r_reach, r_orient, r_open, r_grasp, r_lift, r_transport, r_place

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # Define bounds for Table 1 (Piece Table) where we want to spawn pieces
        # self.piece_table_offset is center of Table 1
        # Table size is self.table_full_size (x, y, z)
        
        # We want to spawn on the surface of Table 1
        # Let's define a region on Table 1 for X pieces and O pieces
        
        # X pieces area (left side of table 1)
        x_x_range = [-0.15, -0.05] # Relative to table center
        x_y_range = [-0.2, 0.2]
        
        # O pieces area (right side of table 1)
        o_x_range = [0.05, 0.15]
        o_y_range = [-0.2, 0.2]
        
        # Create sampler for X pieces
        for i, piece in enumerate(self.x_pieces):
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"X_Piece_{i}_Sampler",
                    mujoco_objects=piece,
                    x_range=x_x_range,
                    y_range=x_y_range,
                    rotation=0.0,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.piece_table_offset,
                    z_offset=0.02, # Small offset to sit on table
                )
            )

        # Create sampler for O pieces
        for i, piece in enumerate(self.o_pieces):
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"O_Piece_{i}_Sampler",
                    mujoco_objects=piece,
                    x_range=o_x_range,
                    y_range=o_y_range,
                    rotation=0.0,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.piece_table_offset,
                    z_offset=0.02,
                )
            )

    def _load_model(self):
        super()._load_model()

        # Adjust robot base position
        self.robots[0].robot_model.set_base_xpos([0, 0, 0])

        # Create Two-Table Arena
        self.mujoco_arena = MultiTableArena(
            table_offsets=[self.board_table_offset, self.piece_table_offset],
            table_rots=[0, 0],
            table_full_sizes=[self.table_full_size, self.table_full_size],
            table_frictions=[self.table_friction, self.table_friction],
            has_legs=[True, True]
        )
        self.mujoco_arena.set_origin([0, 0, 0])

        # Calc absolute top positions for helper properties
        self.board_table_top_z = self.board_table_offset[2]
        self.piece_table_top_z = self.piece_table_offset[2]

        # Create procedural board with grid lines
        self.board_components = self._create_procedural_board()

        # Create Pieces
        tex_attrib = {
        "type": "cube",
        }

        # Material attributes for shiny/reflective look
        mat_attrib_shiny = {
            "texrepeat": "1 1",
            "specular": "0.6",
            "shininess": "0.3",
        }

        # Create custom materials for X pieces (Red/Wood theme)
        x_material = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="x_piece_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib_shiny,
        )

        # Create custom materials for O pieces (Blue/Ceramic theme)
        o_material = CustomMaterial(
            texture="WoodBlue",  # or try "WoodBlue", "Metal", "Plastic"
            tex_name="blueceramic",
            mat_name="o_piece_mat",
            mat_attrib=mat_attrib_shiny,
        )

        # X pieces with material
        self.x_pieces = [
            BoxObject(
                name=f"cube_{i}",
                size=[0.02, 0.02, 0.02],
                rgba=[0.9, 0.2, 0.1, 1],  # Slightly adjusted red
                material=x_material,
                joints=[dict(type="free", damping="0.0005")],
            ) for i in range(5)
        ]

        # O pieces with material
        self.o_pieces = [
            CylinderObject(
                name=f"cyl_{i}",
                size=[0.02, 0.01],
                rgba=[0.1, 0.3, 0.9, 1],  # Slightly adjusted blue
                material=o_material,
                joints=[dict(type="free", damping="0.0005")],
            ) for i in range(5)
        ]

            # Create Task
        all_objects = self.board_components + self.x_pieces + self.o_pieces
        self.model = ManipulationTask(
                mujoco_arena=self.mujoco_arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=all_objects,
            )

        # Generate placement initializer
        self._get_placement_initializer()

    def _setup_references(self):
        super()._setup_references()
        self.x_body_ids = [self.sim.model.body_name2id(p.root_body) for p in self.x_pieces]
        self.o_body_ids = [self.sim.model.body_name2id(p.root_body) for p in self.o_pieces]
        # Store reference to board base for grid center calculations
        self.board_body_id = self.sim.model.body_name2id(self.board_components[0].root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"
            sensors = []
            names = []

            # Standard Panda Setup
            gripper = self.robots[0].gripper
            
            if isinstance(gripper, dict): 
                gripper = gripper['right']

            gripper_site_name = gripper.important_sites["grip_site"]
            gripper_site_id = self.sim.model.site_name2id(gripper_site_name)
            pf = self.robots[0].robot_model.naming_prefix

            def create_pos_quat_sensors(pieces, prefix):
                for i, piece in enumerate(pieces):
                    body_id = self.sim.model.body_name2id(piece.root_body)
                    
                    @sensor(modality=modality)
                    def piece_pos_sensor(obs_cache, bid=body_id):
                        return np.array(self.sim.data.body_xpos[bid])
                    
                    @sensor(modality=modality)
                    def piece_quat_sensor(obs_cache, bid=body_id):
                        return convert_quat(np.array(self.sim.data.body_xquat[bid]), to="xyzw")

                    @sensor(modality=modality)
                    def grip_dist_sensor(obs_cache, gid=gripper_site_id, bid=body_id):
                        return self.sim.data.site_xpos[gid] - self.sim.data.body_xpos[bid]

                    sensors.extend([piece_pos_sensor, piece_quat_sensor, grip_dist_sensor])
                    names.extend([
                        f"{prefix}_{i}_pos", 
                        f"{prefix}_{i}_quat", 
                        f"{pf}gripper_to_{prefix}_{i}"
                    ])

            create_pos_quat_sensors(self.x_pieces, "cube")
            create_pos_quat_sensors(self.o_pieces, "cyl")
            
            # Map object IDs to their relevant sensors
            self.object_id_to_sensors[0] = [n for n in names if "cube" in n]
            self.object_id_to_sensors[1] = [n for n in names if "cyl" in n]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        @sensor(modality="object")
        def board_state_obs(obs_cache):
            return self._get_board_state().flatten().astype(np.float32)
        
        observables["board_state"] = Observable(
            name="board_state",
            sensor=board_state_obs,
            sampling_rate=self.control_freq,
        )

        return observables
    
    def _reset_internal(self):
        super()._reset_internal()

        self.board_pos = self.sim.data.body_xpos[self.board_body_id]

        board_width = 0.15  # This is half-size, actual board is 0.3m wide
        self.grid_spacing = (board_width * 2) / 3.0  # 0.3 / 3 = 0.1m per cell
        z_top = self.board_pos[2] + 0.01  # Board top surface
        
        self.grid_centers = []
        for i in range(3): 
            for j in range(3): 
                x_offset = (i - 1) * self.grid_spacing
                y_offset = (j - 1) * self.grid_spacing
                center = np.array([self.board_pos[0] + x_offset, self.board_pos[1] + y_offset, z_top])
                self.grid_centers.append(center)

        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # Set the collision object joints
                # Note: This assumes pieces have a single free joint (which they do in _load_model)
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        self.sim.forward()

        self.unplaced_x = list(range(5))
        self.unplaced_o = list(range(5))
        self.current_player = 1
        self.timestep = 0

    def _create_procedural_board(self):
        """Creates a procedural TicTacToe board with grid lines"""
        components = []
        
        # Board base plate (dark grey)
        board_base = BoxObject(
            name="board_base",
            size=[0.15, 0.15, 0.005],  # 0.3 x 0.3 x 0.01 m
            rgba=[1.0, 1.0, 1.0, 1],
            joints=None
        )
        # Set position on Board Table (Table 0)
        # We need to manually calculate the position because this runs before collision/physics
        # board_table_offset is the center. 
        # Table height is offset[2] (top surface) - we need to add half board thickness
        board_pos = self.board_table_offset + np.array([0, 0, 0.005]) 
        
        board_base.get_obj().set("pos", array_to_string(board_pos))
        components.append(board_base)
        
        # Grid lines (thin black boxes)
        line_thickness = 0.01
        line_height = 0.002
        
        # Grid line positions (relative to board center)
        # For a 0.3m board with 3 cells, lines are at ±0.05m from center
        grid_positions = [
            # Horizontal lines (y offset)
            (0, -0.05),  # bottom horizontal
            (0, 0.05),   # top horizontal
            # Vertical lines (x offset)
            (-0.05, 0),  # left vertical
            (0.05, 0),   # right vertical
        ]
        
        line_names = ["grid_h_line_1", "grid_h_line_2", "grid_v_line_1", "grid_v_line_2"]
        
        for i, (line_name, (x_offset, y_offset)) in enumerate(zip(line_names, grid_positions)):
            # Horizontal lines have full width, thin depth
            # Vertical lines have thin width, full depth
            if i < 2:  # Horizontal lines
                size = [0.15, line_thickness / 2, line_height / 2]
            else:  # Vertical lines
                size = [line_thickness / 2, 0.15, line_height / 2]
            
            line = BoxObject(
                name=line_name,
                size=size,
                rgba=[0.05, 0.05, 0.05, 1],  # Dark lines
                joints=None
            )
            
            # Position on top of board (board height is 0.005, so top surface is at +0.005)
            # board_pos already includes the table height + base half height
            line_pos = board_pos + np.array([x_offset, y_offset, 0.005 + line_height / 2])
            line.get_obj().set("pos", array_to_string(line_pos))
            components.append(line)
        
        return components




    def _get_board_state(self):
        board = np.zeros(9, dtype=int)
        if not self.grid_centers: return board.reshape(3,3)

        for idx, center in enumerate(self.grid_centers):
            for piece in self.x_pieces:
                pid = self.sim.model.body_name2id(piece.root_body)
                pos = self.sim.data.body_xpos[pid]
                if np.linalg.norm(pos[:2] - center[:2]) < 0.04: 
                    board[idx] = 1
            for piece in self.o_pieces:
                pid = self.sim.model.body_name2id(piece.root_body)
                pos = self.sim.data.body_xpos[pid]
                if np.linalg.norm(pos[:2] - center[:2]) < 0.04:
                    board[idx] = 2
        return board.reshape(3, 3)

    def _check_win(self, player):
        b = self._get_board_state()
        wins = [
            np.all(b[0,:] == player), np.all(b[1,:] == player), np.all(b[2,:] == player),
            np.all(b[:,0] == player), np.all(b[:,1] == player), np.all(b[:,2] == player),
            np.all(np.diag(b) == player), np.all(np.diag(np.fliplr(b)) == player)
        ]
        return any(wins)

    def _check_draw(self):
        board = self._get_board_state().flatten()
        return np.all(board != 0) and not self._check_win(1) and not self._check_win(2)

    def _check_success(self):
        """
        Check if the task is successfully completed.
        For manipulation/pick-place, we defined success as placing a valid piece.
        """
        current_piece = self._get_current_piece()
        if current_piece and self._piece_placed_in_valid_cell(current_piece):
            return True
        return False

    def _check_done(self):
        """Helper to calculate if the episode is over."""
        # Done if board is full (Draw) or horizon reached
        # In Solitaire mode, we keep going until no pieces left or board full
        board_full = np.all(self._get_board_state().flatten() != 0)
        return board_full or (self.timestep >= self.horizon and not self.ignore_done)

    def step(self, action):
        # 1. Constrain action (fix orientation)
        action = np.array(action)
        action[3:6] = 0.0 # Zero out rotation actions if you want fixed orientation

        # 2. Run the standard Robosuite step
        obs, reward, done, info = super().step(action)

        # 3. Custom Logic
        game_reward = self.reward(action)
        
        # Check for placement success
        current_piece = self._get_current_piece()
        if current_piece and self._piece_placed_in_valid_cell(current_piece):
            # SUCCESS!
            game_reward += 5.0 # Big bonus for placing
            
            # Remove from list (STATE MODIFICATION HAPPENS HERE NOW)
            if self.current_player == 1:
                if current_piece in self.x_pieces:
                     idx = self.x_pieces.index(current_piece)
                     if idx in self.unplaced_x:
                         self.unplaced_x.remove(idx)
            else:
                if current_piece in self.o_pieces:
                    idx = self.o_pieces.index(current_piece)
                    if idx in self.unplaced_o:
                        self.unplaced_o.remove(idx)
            
            # Switch Turns
            self.current_player = 3 - self.current_player
            
        # Combine dones
        done = self._check_done()

        # Update info
        info.update({
            "board_state": self._get_board_state(),
            "current_player": self.current_player
        })

        return obs, game_reward, done, info
    
    def _get_gripper(self):
        """Helper to get the actual gripper object, handling the dictionary case."""
        gripper = self.robots[0].gripper
        if isinstance(gripper, dict):
            # Usually 'right', but we can fallback to the first value if 'right' is missing
            return gripper.get('right', list(gripper.values())[0])
        return gripper

    MAX_TRIES = 100
    def _place_opponent_piece(self):
        """Deprecated for Solitaire mode."""
        pass

    def _get_current_piece(self):
        """
        Determine the piece the gripper should interact with based on self.current_player.
        """
        if self.current_player == 1:
            target_list = self.unplaced_x
            pieces = self.x_pieces
        else:
            target_list = self.unplaced_o
            pieces = self.o_pieces
        
        if not target_list:
            return None
            
        next_idx = target_list[0]
        return pieces[next_idx]

    def _check_grasp(self, gripper, object_geoms):
        """
        Check if the gripper is grasping the object (from Robosuite utils).
        """
        contacts = self.sim.data.contact
        for contact in contacts:
            if contact.geom1 in gripper.contact_geoms and contact.geom2 in object_geoms.contact_geoms:
                return True
            if contact.geom2 in gripper.contact_geoms and contact.geom1 in object_geoms.contact_geoms:
                return True
        return False

    def _get_piece_height(self, piece):
        """
        Get z-height of the piece above the board.
        """
        pid = self.sim.model.body_name2id(piece.root_body)
        return self.sim.data.body_xpos[pid][2] - self.board_pos[2]

    def _gripper_to_target(self, gripper, target, target_type="body", return_distance=False):
        """
        Distance from gripper site to target (body or site).
        """
        grip_site_pos = self.sim.data.site_xpos[self.sim.model.site_name2id(gripper.important_sites["grip_site"])]
        if target_type == "body":
            target_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(target)]
        elif target_type == "site":
            target_pos = self.sim.data.site_xpos[self.sim.model.site_name2id(target)]
        else:
            raise ValueError("Invalid target_type")
        dist = np.linalg.norm(grip_site_pos - target_pos)
        if return_distance:
            return dist
        return target_pos  # Or vector if needed

    def _piece_placed_in_valid_cell(self, piece):
        """
        Check if piece is placed (not grasped, low velocity, in a grid cell).
        PURE FUNCTION: Does not modify state.
        """
        gripper = self._get_gripper()
        if self._check_grasp(gripper, piece):  # Still grasped?
            return False
            
        pid = self.sim.model.body_name2id(piece.root_body)
        pos = self.sim.data.body_xpos[pid]
        # vel = self.sim.data.get_body_xvelp(piece.root_body) # this might be buggy in some versions
        
        # Simplified check: just height and position
        if pos[2] > self.board_pos[2] + 0.05:  # Moving or too high?
            return False
            
        for idx, center in enumerate(self.grid_centers):
            # Check if close to a grid center
            if np.linalg.norm(pos[:2] - center[:2]) < self.grid_spacing / 2:
                # Check if cell empty (from board state)
                board = self._get_board_state().flatten()
                if board[idx] == 0:
                    return True
        return False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest object.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest object
        if vis_settings.get("grippers"):
            gripper = self._get_gripper()
            current_piece = self._get_current_piece()
            
            if current_piece:
                self._visualize_gripper_to_target(
                    gripper=gripper,
                    target=current_piece.root_body,
                    target_type="body",
                )