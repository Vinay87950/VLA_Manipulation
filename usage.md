Reference from here - 'https://robosuite.ai/docs/modules/robots.html'

1. During initilization of environment (suite.make(....)), individual robots are both instantiated and initialized

2. During a given simulation call (env.step(...)), the environment will receive a set of actions and distribute them accordingly to each robot, according to their respective action spaces. Each robot then converts these actions into low-level torques via their respective controllers, and directly executes these torques in the simulation

Now for the Robosuite Objects - 'https://robosuite.ai/docs/modules/objects.html'

the stl files are downloaded from 'https://www.thingiverse.com/thing:7115554'

to run the simulation use python simulation_run.py


observations of the positions this is for my env 
'============================================================
ENVIRONMENT: TicTacToeEnv
============================================================

📊 OBSERVATION KEYS:
  robot0_joint_pos: shape=(7,)
  robot0_joint_pos_cos: shape=(7,)
  robot0_joint_pos_sin: shape=(7,)
  robot0_joint_vel: shape=(7,)
  robot0_eef_pos: shape=(3,)
  robot0_eef_quat: shape=(4,)
  robot0_eef_quat_site: shape=(4,)
  robot0_gripper_qpos: shape=(2,)
  robot0_gripper_qvel: shape=(2,)
  cube_0_pos: shape=(3,)
  cube_0_quat: shape=(4,)
  robot0_gripper_to_cube_0: shape=(3,)
  cube_1_pos: shape=(3,)
  cube_1_quat: shape=(4,)
  robot0_gripper_to_cube_1: shape=(3,)
  cube_2_pos: shape=(3,)
  cube_2_quat: shape=(4,)
  robot0_gripper_to_cube_2: shape=(3,)
  cube_3_pos: shape=(3,)
  cube_3_quat: shape=(4,)
  robot0_gripper_to_cube_3: shape=(3,)
  cube_4_pos: shape=(3,)
  cube_4_quat: shape=(4,)
  robot0_gripper_to_cube_4: shape=(3,)
  cyl_0_pos: shape=(3,)
  cyl_0_quat: shape=(4,)
  robot0_gripper_to_cyl_0: shape=(3,)
  cyl_1_pos: shape=(3,)
  cyl_1_quat: shape=(4,)
  robot0_gripper_to_cyl_1: shape=(3,)
  cyl_2_pos: shape=(3,)
  cyl_2_quat: shape=(4,)
  robot0_gripper_to_cyl_2: shape=(3,)
  cyl_3_pos: shape=(3,)
  cyl_3_quat: shape=(4,)
  robot0_gripper_to_cyl_3: shape=(3,)
  cyl_4_pos: shape=(3,)
  cyl_4_quat: shape=(4,)
  robot0_gripper_to_cyl_4: shape=(3,)
  board_state: shape=(9,)
  robot0_proprio-state: shape=(43,)
  object-state: shape=(109,)

🔲 ALL BODIES:
  world                         : [  0.000,   0.000,   0.000]
  left_eef_target               : [  0.000,   0.000,  -1.000]
  right_eef_target              : [  0.000,   0.000,  -1.000]
  table0                        : [  0.400,   0.300,   0.775]
  table1                        : [  0.400,  -0.300,   0.775]
  robot0_base                   : [  0.000,   0.000,   0.912]
  robot0_link0                  : [  0.000,   0.000,   0.912]
  robot0_link1                  : [  0.000,   0.000,   1.245]
  robot0_link2                  : [  0.000,   0.000,   1.245]
  robot0_link3                  : [  0.056,  -0.002,   1.556]
  robot0_link4                  : [  0.137,  -0.004,   1.542]
  robot0_link5                  : [  0.345,  -0.011,   1.209]
  robot0_link6                  : [  0.345,  -0.011,   1.209]
  robot0_link7                  : [  0.432,  -0.014,   1.224]
  robot0_right_hand             : [  0.451,  -0.015,   1.119]
  gripper0_right_right_gripper  : [  0.451,  -0.015,   1.119]
  gripper0_right_eef            : [  0.468,  -0.015,   1.024]
  gripper0_right_leftfinger     : [  0.460,  -0.036,   1.068]
  gripper0_right_finger_joint1_tip: [  0.469,  -0.045,   1.013]
  gripper0_right_rightfinger    : [  0.460,   0.006,   1.068]
  gripper0_right_finger_joint2_tip: [  0.470,   0.014,   1.013]
  fixed_mount0_base             : [  0.000,   0.000,   0.922]
  fixed_mount0_controller_box   : [  0.000,   0.000,   0.922]
  fixed_mount0_pedestal_feet    : [  0.000,   0.000,   0.922]
  fixed_mount0_torso            : [  0.000,   0.000,   0.922]
  fixed_mount0_pedestal         : [  0.000,   0.000,   0.922]
  board_base_main               : [  0.400,   0.300,   0.805]
  grid_h_line_1_main            : [  0.400,   0.250,   0.811]
  grid_h_line_2_main            : [  0.400,   0.350,   0.811]
  grid_v_line_1_main            : [  0.350,   0.300,   0.811]
  grid_v_line_2_main            : [  0.450,   0.300,   0.811]
  cube_0_main                   : [  0.300,  -0.450,   0.850]
  cube_1_main                   : [  0.300,  -0.390,   0.850]
  cube_2_main                   : [  0.300,  -0.330,   0.850]
  cube_3_main                   : [  0.300,  -0.270,   0.850]
  cube_4_main                   : [  0.300,  -0.210,   0.850]
  cyl_0_main                    : [  0.500,  -0.450,   0.850]
  cyl_1_main                    : [  0.500,  -0.390,   0.850]
  cyl_2_main                    : [  0.500,  -0.330,   0.850]
  cyl_3_main                    : [  0.500,  -0.270,   0.850]
  cyl_4_main                    : [  0.500,  -0.210,   0.850]

📍 ALL SITES:
  table0_top                    : [  0.400,   0.300,   0.800]
  table1_top                    : [  0.400,  -0.300,   0.800]
  robot0_right_center           : [  0.000,   0.000,   0.912]
  gripper0_right_ft_frame       : [  0.451,  -0.015,   1.119]
  gripper0_right_grip_site      : [  0.468,  -0.015,   1.024]
  gripper0_right_ee_x           : [  0.469,   0.085,   1.023]
  gripper0_right_ee_y           : [  0.566,  -0.016,   1.042]
  gripper0_right_ee_z           : [  0.485,  -0.016,   0.925]
  gripper0_right_grip_site_cylinder: [  0.468,  -0.015,   1.024]
  board_base_default_site       : [  0.400,   0.300,   0.805]
  grid_h_line_1_default_site    : [  0.400,   0.250,   0.811]
  grid_h_line_2_default_site    : [  0.400,   0.350,   0.811]
  grid_v_line_1_default_site    : [  0.350,   0.300,   0.811]
  grid_v_line_2_default_site    : [  0.450,   0.300,   0.811]
  cube_0_default_site           : [  0.300,  -0.450,   0.850]
  cube_1_default_site           : [  0.300,  -0.390,   0.850]
  cube_2_default_site           : [  0.300,  -0.330,   0.850]
  cube_3_default_site           : [  0.300,  -0.270,   0.850]
  cube_4_default_site           : [  0.300,  -0.210,   0.850]
  cyl_0_default_site            : [  0.500,  -0.450,   0.850]
  cyl_1_default_site            : [  0.500,  -0.390,   0.850]
  cyl_2_default_site            : [  0.500,  -0.330,   0.850]
  cyl_3_default_site            : [  0.500,  -0.270,   0.850]
  cyl_4_default_site            : [  0.500,  -0.210,   0.850]

⚠️ Could not find table - check body names above'

this is for original pickplace env 
============================================================
ENVIRONMENT: PickPlace
============================================================

📊 OBSERVATION KEYS:
  robot0_joint_pos: shape=(7,)
  robot0_joint_pos_cos: shape=(7,)
  robot0_joint_pos_sin: shape=(7,)
  robot0_joint_vel: shape=(7,)
  robot0_eef_pos: shape=(3,)
  robot0_eef_quat: shape=(4,)
  robot0_eef_quat_site: shape=(4,)
  robot0_gripper_qpos: shape=(2,)
  robot0_gripper_qvel: shape=(2,)
  Milk_to_robot0_eef_pos: shape=(3,)
  Milk_to_robot0_eef_quat: shape=(4,)
  Milk_pos: shape=(3,)
  Milk_quat: shape=(4,)
  Bread_to_robot0_eef_pos: shape=(3,)
  Bread_to_robot0_eef_quat: shape=(4,)
  Bread_pos: shape=(3,)
  Bread_quat: shape=(4,)
  Cereal_to_robot0_eef_pos: shape=(3,)
  Cereal_to_robot0_eef_quat: shape=(4,)
  Cereal_pos: shape=(3,)
  Cereal_quat: shape=(4,)
  Can_to_robot0_eef_pos: shape=(3,)
  Can_to_robot0_eef_quat: shape=(4,)
  Can_pos: shape=(3,)
  Can_quat: shape=(4,)
  robot0_proprio-state: shape=(43,)
  object-state: shape=(56,)

🔲 ALL BODIES:
  world                         : [  0.000,   0.000,   0.000]
  bin1                          : [  0.100,  -0.250,   0.800]
  bin2                          : [  0.100,   0.280,   0.800]
  left_eef_target               : [  0.000,   0.000,  -1.000]
  right_eef_target              : [  0.000,   0.000,  -1.000]
  robot0_base                   : [ -0.500,  -0.100,   0.912]
  robot0_link0                  : [ -0.500,  -0.100,   0.912]
  robot0_link1                  : [ -0.500,  -0.100,   1.245]
  robot0_link2                  : [ -0.500,  -0.100,   1.245]
  robot0_link3                  : [ -0.436,  -0.102,   1.554]
  robot0_link4                  : [ -0.355,  -0.105,   1.538]
  robot0_link5                  : [ -0.161,  -0.111,   1.197]
  robot0_link6                  : [ -0.161,  -0.111,   1.197]
  robot0_link7                  : [ -0.073,  -0.112,   1.203]
  robot0_right_hand             : [ -0.065,  -0.113,   1.097]
  gripper0_right_right_gripper  : [ -0.065,  -0.113,   1.097]
  gripper0_right_eef            : [ -0.058,  -0.114,   1.001]
  gripper0_right_leftfinger     : [ -0.061,  -0.135,   1.045]
  gripper0_right_finger_joint1_tip: [ -0.056,  -0.144,   0.989]
  gripper0_right_rightfinger    : [ -0.061,  -0.093,   1.045]
  gripper0_right_finger_joint2_tip: [ -0.057,  -0.085,   0.989]
  fixed_mount0_base             : [ -0.500,  -0.100,   0.922]
  fixed_mount0_controller_box   : [ -0.500,  -0.100,   0.922]
  fixed_mount0_pedestal_feet    : [ -0.500,  -0.100,   0.922]
  fixed_mount0_torso            : [ -0.500,  -0.100,   0.922]
  fixed_mount0_pedestal         : [ -0.500,  -0.100,   0.922]
  VisualMilk_main               : [  0.003,   0.158,   0.885]
  VisualBread_main              : [  0.198,   0.158,   0.845]
  VisualCereal_main             : [  0.003,   0.403,   0.900]
  VisualCan_main                : [  0.198,   0.403,   0.860]
  Milk_main                     : [  0.206,  -0.326,   0.885]
  Bread_main                    : [ -0.011,  -0.250,   0.845]
  Cereal_main                   : [  0.172,  -0.233,   0.900]
  Can_main                      : [ -0.007,  -0.156,   0.860]

📍 ALL SITES:
  robot0_right_center           : [ -0.500,  -0.100,   0.912]
  gripper0_right_ft_frame       : [ -0.065,  -0.113,   1.097]
  gripper0_right_grip_site      : [ -0.058,  -0.114,   1.001]
  gripper0_right_ee_x           : [ -0.059,  -0.014,   1.000]
  gripper0_right_ee_y           : [  0.042,  -0.113,   1.008]
  gripper0_right_ee_z           : [ -0.050,  -0.115,   0.901]
  gripper0_right_grip_site_cylinder: [ -0.058,  -0.114,   1.001]
  VisualMilk_default_site       : [  0.003,   0.158,   0.885]
  VisualBread_default_site      : [  0.198,   0.158,   0.845]
  VisualCereal_default_site     : [  0.003,   0.403,   0.900]
  VisualCan_default_site        : [  0.198,   0.403,   0.860]
  Milk_default_site             : [  0.206,  -0.326,   0.885]
  Bread_default_site            : [ -0.011,  -0.250,   0.845]
  Cereal_default_site           : [  0.172,  -0.233,   0.900]
  Can_default_site              : [ -0.007,  -0.156,   0.860]
