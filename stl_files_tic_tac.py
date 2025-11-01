# Read this for objects: https://robosuite.ai/docs/modules/objects.html
# Especially this section: https://robosuite.ai/docs/modules/objects.html#creating-a-procedurally-generated-object
# this to 'https://robosuite.ai/docs/source/robosuite.environments.manipulation.html#module-robosuite.environments.manipulation.manipulation_env'
# 'https://community.latenode.com/t/mujoco-custom-object-mesh-flickering-through-table-surface-in-gym-environment/27486'
# for tuning camer look over 'https://github.com/quantumiracle/robolite/blob/zihan/robosuite/scripts/tune_camera.py'
# see here for prompt template 'https://github.com/shaunck96/arc_agi/blob/main/solver.py#L169'
'''
To design this environment, the idea is taken from '/mujoco_sim/robosuite/robosuite/environments/manipulation/lift.py'
'''

from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, xml_path_completion
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class XObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(xml_path_completion("/Users/killuaa/Desktop/mujoco_sim/self_environment/objects/x.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")], # damping means how quickly the object slows down and eventually stops moving
                         obj_type="all", duplicate_collision_geoms=True) 


class OObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(xml_path_completion("/Users/killuaa/Desktop/mujoco_sim/self_environment/objects/o.xml"),
                         name=name, joints=[dict(type="free", damping="0.0001")],
                         obj_type="all", duplicate_collision_geoms=True)


class BoardObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(xml_path_completion("/Users/killuaa/Desktop/mujoco_sim/self_environment/objects/board.xml"),
                         name=name, joints=None,
                         obj_type="all", duplicate_collision_geoms=True)


class TicTacToeEnv(ManipulationEnv):
    """
    This class corresponds to a Tic Tac Toe task for a single robot arm (Franka Panda).

    The environment sets up a table with custom X and O pieces from XML meshes, placed on a static custom board.
    Pieces are initially placed on the sides for the robot to pick and place on a virtual 3x3 grid on the board.
    Rewards encourage grasping and placing; success checks for placement on the grid.
    Game logic (turns, wins) can be extended in overrides.

    Args: Similar to Lift environment; see docstring for details.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(1.1, 1.1, 0.05),  # Larger table for grid and pieces
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_eye_in_hand",
        # render_camera="frontview", 
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20, # Default control frequency
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="frontview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None, # No segmentation by default
        renderer="mujoco",
        renderer_config=None,
        seed=None,
    ):
        # Tic Tac Toe specific settings
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.table_top = self.table_offset + np.array([0, 0, self.table_full_size[2] / 2])

        # Grid will be computed dynamically based on board sites
        self.grid_spacing = None
        self.grid_centers = None

        # Reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # Object observations
        self.use_object_obs = use_object_obs

        # Placement initializer
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

    def reward(self, action=None):
        """
        Reward function: Sparse for placement on grid, shaped for reaching/grasping if enabled.
        """
        reward = 0.0

        # Check if any piece is placed on a grid center
        if self._check_success():
            reward = 2.0  # Sparse reward for successful placement

        if self.reward_shaping:
            # Example shaping: Distance to nearest unused piece
            min_dist = np.inf
            for piece in self.x_pieces:  # Assuming robot places X
                dist = self._gripper_to_target(
                    gripper=self.robots[0].gripper, target=piece.root_body, target_type="body", return_distance=True
                )
                min_dist = min(min_dist, dist)
            reaching_reward = 1 - np.tanh(10.0 * min_dist)
            reward += reaching_reward

            # Grasping bonus
            grasped = False
            for piece in self.x_pieces:
                if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=piece.contact_geoms):
                    grasped = True
                    break
            if grasped:
                reward += 0.25

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def _load_model(self):
        """
        Loads the model: Arena, robot, static board, and multiple pieces for X and O.
        """
        super()._load_model()

        # Adjust base pose
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Create board (static)
        self.board = BoardObject(name="board")

        # Merge board into arena as static object
        mujoco_arena.merge_assets(self.board)
        board_obj = self.board.get_obj()
        # Calculate position to be on top of the table
        # self.table_top is the absolute z-coordinate of the table surface
        # We subtract the board's bottom_offset to place it flush on the table
        pos = self.table_top - self.board.bottom_offset
        board_obj.set("pos", array_to_string(pos))
        mujoco_arena.worldbody.append(board_obj)

        # Create pieces (dynamic)
        self.x_pieces = [
            XObject(name=f"x_piece_{i}") for i in range(5)
        ]
        self.o_pieces = [
            OObject(name=f"o_piece_{i}") for i in range(5)
        ]

        # Placement initializer: Separate areas for X and O (board is fixed, not sampled)
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="PieceSampler")
            # Left side for X
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="XSampler",
                    mujoco_objects=self.x_pieces,
                    x_range=[-0.45, -0.25],
                    y_range=[-0.25, 0.25],
                    rotation=None,
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.03  # Slightly above table to avoid sinking
                )
            )
            # Right side for O
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="OSampler",
                    mujoco_objects=self.o_pieces,
                    x_range=[0.25, 0.45],
                    y_range=[-0.25, 0.25],
                    rotation=None,
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.03  # Slightly above table to avoid sinking
                )
            )

        # Task model (pieces only, board is part of arena)
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.x_pieces + self.o_pieces,
        )

    def _setup_references(self):
        """
        Sets up references to piece bodies and board.
        """
        super()._setup_references()
        self.x_body_ids = [self.sim.model.body_name2id(p.root_body) for p in self.x_pieces]
        self.o_body_ids = [self.sim.model.body_name2id(p.root_body) for p in self.o_pieces]
        self.board_body_id = self.sim.model.body_name2id(self.board.root_body)

    def _setup_observables(self):
        """
        Observables include positions and quaternions for all pieces.
        """
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"
            sensors = []
            names = []
            all_pieces = [(self.x_pieces, "x_piece"), (self.o_pieces, "o_piece")]

            for pieces, prefix in all_pieces:
                for idx, piece in enumerate(pieces):
                    @sensor(modality=modality)
                    def piece_pos(obs_cache, p=piece):  # Closure over piece
                        return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(p.root_body)])

                    @sensor(modality=modality)
                    def piece_quat(obs_cache, p=piece):
                        return convert_quat(np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(p.root_body)]), to="xyzw")

                    sensors.extend([piece_pos, piece_quat])
                    names.extend([f"{prefix}_{idx}_pos", f"{prefix}_{idx}_quat"])

            # Add gripper-to-piece sensors (example for X pieces)
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes):
                for idx, piece in enumerate(self.x_pieces):
                    sensors.append(self._get_obj_eef_sensor(full_pf, piece.root_body, f"{arm_pf}gripper_to_{prefix}_{idx}_pos", modality))
                    names.append(f"{arm_pf}gripper_to_{prefix}_{idx}_pos")

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets object positions using the initializer and computes dynamic grid centers based on board.
        """
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Compute grid centers dynamically based on board and piece sites
        self.board_pos = self.sim.data.body_xpos[self.board_body_id]
        self.board_top_offset = self.board.top_offset
        self.piece_bottom_offset = self.x_pieces[0].bottom_offset  # Assume same for O
        self.board_horizontal_radius = self.board.horizontal_radius
        self.grid_spacing = (2 * self.board_horizontal_radius) / 3
        self.grid_centers = [
            self.board_pos + np.array([(i - 1) * self.grid_spacing, (j - 1) * self.grid_spacing, 0]) +
            self.board_top_offset - self.piece_bottom_offset
            for i in range(3) for j in range(3)
        ]

    def visualize(self, vis_settings):
        """
        Visualizes gripper distance to pieces.
        """
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            for piece in self.x_pieces:  # Visualize for X pieces
                self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=piece)

    def _check_success(self):
        """
        Checks if any piece is placed near a grid center (extend to check for three in a row).
        """
        placed = False
        all_body_ids = self.x_body_ids + self.o_body_ids
        for body_id in all_body_ids:
            pos = self.sim.data.body_xpos[body_id]
            for grid_pos in self.grid_centers:
                if np.linalg.norm(pos - grid_pos) < 0.03:  # Tolerance
                    placed = True
                    break
        return placed

    # Extension point for game logic
    def _get_board_state(self):
        """
        Computes current board state by matching piece positions to grid.
        Returns: 3x3 numpy array (0: empty, 1: X, 2: O)
        """
        board = np.zeros((3, 3), dtype=int)
        for idx, grid_pos in enumerate(self.grid_centers):
            i, j = divmod(idx, 3)
            for x_id in self.x_body_ids:
                if np.linalg.norm(self.sim.data.body_xpos[x_id] - grid_pos) < 0.03:
                    board[i, j] = 1
                    break
            for o_id in self.o_body_ids:
                if np.linalg.norm(self.sim.data.body_xpos[o_id] - grid_pos) < 0.03:
                    board[i, j] = 2
                    break
        return board

    def _check_win(self, player):
        """
        Checks if player (1: X, 2: O) has three in a row.
        """
        board = self._get_board_state()
        # Rows, columns, diagonals
        for i in range(3):
            if all(board[i, :] == player) or all(board[:, i] == player):
                return True
        if all(np.diag(board) == player) or all(np.diag(np.fliplr(board)) == player):
            return True
        return False