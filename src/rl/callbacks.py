"""Small callbacks used during SAC training."""

import csv
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingDataCallback(BaseCallback):
    """Persist one diagnostic row for every environment transition."""

    FIELDS = (
        "timestep",
        "episode",
        "episode_step",
        "reward",
        "reward_shell",
        "reward_delta_v",
        "reward_terminal",
        "action_normalized",
        "dv_base_m_s",
        "dv_residual_m_s",
        "dv_command_m_s",
        "dv_executed_m_s",
        "thruster_fired",
        "dv_off_threshold_m_s",
        "radius_m",
        "cumulative_dv_m_s",
        "mass_fraction_1",
        "mass_fraction_2",
        "initial_position_x_m",
        "initial_position_y_m",
        "initial_position_z_m",
        "initial_velocity_x_m_s",
        "initial_velocity_y_m_s",
        "initial_velocity_z_m_s",
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "velocity_z_m_s",
        "terminated",
        "truncated",
    )

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.episode = 1
        self.episode_step = 0
        self.file = None
        self.writer = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.file.flush()

    def _on_step(self):
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        rewards = np.asarray(self.locals["rewards"]).reshape(-1)
        actions = np.asarray(self.locals["actions"]).reshape(-1)

        self.episode_step += 1
        self.writer.writerow({
            "timestep": self.num_timesteps,
            "episode": self.episode,
            "episode_step": self.episode_step,
            "reward": float(rewards[0]),
            "reward_shell": info.get("reward_shell", ""),
            "reward_delta_v": info.get("reward_delta_v", ""),
            "reward_terminal": info.get("reward_terminal", ""),
            "action_normalized": float(actions[0]),
            "dv_base_m_s": info.get("dv_base_m_s", ""),
            "dv_residual_m_s": info.get("dv_residual_m_s", ""),
            "dv_command_m_s": info.get("dv_command_m_s", ""),
            "dv_executed_m_s": info.get("dv_executed_m_s", ""),
            "thruster_fired": info.get("thruster_fired", ""),
            "dv_off_threshold_m_s": info.get("dv_off_threshold_m_s", ""),
            "radius_m": info.get("radius_m", ""),
            "cumulative_dv_m_s": info.get("cumulative_dv_m_s", ""),
            "mass_fraction_1": info.get("mass_fraction_1", ""),
            "mass_fraction_2": info.get("mass_fraction_2", ""),
            "initial_position_x_m": info.get("initial_position_x_m", ""),
            "initial_position_y_m": info.get("initial_position_y_m", ""),
            "initial_position_z_m": info.get("initial_position_z_m", ""),
            "initial_velocity_x_m_s": info.get("initial_velocity_x_m_s", ""),
            "initial_velocity_y_m_s": info.get("initial_velocity_y_m_s", ""),
            "initial_velocity_z_m_s": info.get("initial_velocity_z_m_s", ""),
            "position_x_m": info.get("position_x_m", ""),
            "position_y_m": info.get("position_y_m", ""),
            "position_z_m": info.get("position_z_m", ""),
            "velocity_x_m_s": info.get("velocity_x_m_s", ""),
            "velocity_y_m_s": info.get("velocity_y_m_s", ""),
            "velocity_z_m_s": info.get("velocity_z_m_s", ""),
            "terminated": info.get("terminated", ""),
            "truncated": info.get("truncated", ""),
        })
        self.file.flush()

        dones = np.asarray(self.locals["dones"]).reshape(-1)
        if dones[0]:
            self.episode += 1
            self.episode_step = 0
        return True

    def _on_training_end(self):
        if self.file is not None:
            self.file.close()


class RewardPrint(BaseCallback):
    """Print episode reward and rolling statistics every N episodes."""

    def __init__(self, every_episodes=10):
        super().__init__()
        self.every = every_episodes
        self.n_episodes = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self.n_episodes += 1
            if self.n_episodes % self.every:
                continue

            episode_stats = self.model.ep_info_buffer
            mean_reward = np.mean([item["r"] for item in episode_stats])
            mean_length = np.mean([item["l"] for item in episode_stats])
            episode = info["episode"]
            print(
                f"ep {self.n_episodes:6d}  this_r={episode['r']:.3f}  "
                f"this_l={episode['l']:.0f}  mean={mean_reward:.3f}  "
                f"len_mean={mean_length:.1f}",
                flush=True,
            )
        return True
