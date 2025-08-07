from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique


import os
import pickle

TRAIN = True

@configclass
class UREnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333 #s  = 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 23 - 2 + 4
    state_space = 0
    num_envs = 4
    action_scale = 7.5
    dof_velocity_scale = 0.1

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Marker visualization
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1) 

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    action_penalty_scale = 0.05


class UREnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: UREnvCfg

    def __init__(self, cfg: UREnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs) # call 

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.hand_link_idx = self._robot.find_bodies("panda_hand")[0][0]

        self._resample_command(torch.arange(self.num_envs))
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])

        if TRAIN:
            base_path = "logs/skrl/reach_franka"
            self.folder_path = self.find_latest_folder(base_path) + '/checkpoints/mydata/'
            
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)

            self.data = {
                "env_interaction": [],
                "rewards": [],
                "dist_reward": [],
                "rot_reward": [],
                "action_penalty": [],
            }
            self.episode_counter = 0


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


        # -- goal pose
        self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
        # -- current body pose
        self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)

        # variables
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.pos_x=(0.35, 0.65)
        self.pos_y=(-0.2, 0.2)
        self.pos_z=(0.15, 0.5)
        self.roll=(0.0, 0.0)
        self.pitch=(-3.14, 3.14)
        self.yaw=(-3.14, 3.14)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.tensor([False], dtype=torch.bool)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        if TRAIN:
            if truncated.all():
                if self.folder_path:
                    file_path = os.path.join(self.folder_path, f"data_ep{self.episode_counter}.pkl")
                    print("\n\n",file_path)
                    if not os.path.exists(file_path): # never override data
                        with open(file_path, "wb") as f:
                            pickle.dump(self.data, f)
                        print(f"Data saved to {file_path}")                    
                    self.data = {
                        "env_interaction": [],
                        "rewards": [],
                        "dist_reward": [],
                        "rot_reward": [],
                        "action_penalty": [],
                    }
                    self.episode_counter += 1
                else:
                    print("No valid folder found to save data.")
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:

        # -- current body pose
        hand_pos = self._robot.data.body_state_w[:, self.hand_link_idx, :3]#.cpu().numpy()
        hand_quat = self._robot.data.body_state_w[:, self.hand_link_idx, 3:7]#.cpu().numpy()
        self.current_pose_visualizer.visualize(hand_pos,hand_quat)

        return self._compute_rewards(
            self.actions,
            self.pose_command_w[:, :3],#.cpu().numpy(),
            hand_pos,
            self.pose_command_w[:, 3:],#.cpu().numpy(),
            hand_quat,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.action_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self._resample_command(torch.arange(self.num_envs))
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        hand_pos = self._robot.data.body_state_w[:, self.hand_link_idx, :3]#.cpu().numpy()
        to_target = self.pose_command_w[:, :3] - hand_pos

        ur_rot = self._robot.data.body_state_w[:, self.hand_link_idx, 3:7]
        des_rot = self.pose_command_w[:, 3:]
        dot = ur_rot * des_rot

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                dot
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_rewards(
        self,
        actions,
        ur_pos,
        des_pos,
        ur_rot,
        des_rot,
        dist_reward_scale,
        rot_reward_scale,
        action_penalty_scale,
    ):

        # distance from hand to the drawer
        d = torch.norm(ur_pos - des_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        dot = torch.sum(ur_rot * des_rot, dim=1)
        # rot_reward = 1.0 / (1.0 + dot**2)
        # rot_reward *= rot_reward
        # rot_reward = torch.where(dot <= 0.02, rot_reward * 2, rot_reward)
        rot_reward = 0.5 * torch.sign(dot) * dot**2

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            - action_penalty_scale * action_penalty
        )
        # self.extras["log"] = {
        #     "dist_reward": (dist_reward_scale * dist_reward).mean(),
        #     "rot_reward": (rot_reward_scale * rot_reward).mean(),
        #     "action_penalty": (-action_penalty_scale * action_penalty).mean(),
        # }
        # print('\n\n here')
        if TRAIN:
            self.data["env_interaction"].append(self.episode_length_buf.cpu().tolist()[0])
            self.data["rewards"].append(rewards.mean().cpu().tolist())
            self.data["dist_reward"].append((dist_reward_scale * dist_reward).mean().cpu().tolist())
            self.data["rot_reward"].append((rot_reward_scale * rot_reward).mean().cpu().tolist())
            self.data["action_penalty"].append((-action_penalty_scale * action_penalty).mean().cpu().tolist())
        return rewards

    def _resample_command(self, env_ids):
        # not fully random

        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.roll)
        euler_angles[:, 1].uniform_(*self.pitch)
        euler_angles[:, 2].uniform_(*self.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat)

        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )

        

    def find_latest_folder(self, base_path):
        folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        latest_folder = sorted(folders)[-1] if folders else None
        return os.path.join(base_path, latest_folder) if latest_folder else None


# save in case it is useful

        # def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
        #     """Compute pose in env-local coordinates"""
        #     world_transform = xformable.ComputeLocalToWorldTransform(0)
        #     world_pos = world_transform.ExtractTranslation()
        #     world_quat = world_transform.ExtractRotationQuat()

        #     px = world_pos[0] - env_pos[0]
        #     py = world_pos[1] - env_pos[1]
        #     pz = world_pos[2] - env_pos[2]
        #     qx = world_quat.imaginary[0]
        #     qy = world_quat.imaginary[1]
        #     qz = world_quat.imaginary[2]
        #     qw = world_quat.real

        #     return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)
        # stage = get_current_stage()

        # hand_pose = get_env_local_pose(
        #     self.scene.env_origins[0],
        #     UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
        #     self.device,
        # )