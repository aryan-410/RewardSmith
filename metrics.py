import numpy as np
import time
from typing import List, Optional
import torch
import torch.nn as nn

class RewardMetrics:
    """
    Tracks and computes RL metrics: 
    - Best Episode Reward
    - Return Variance (over last N episodes)
    """
    def __init__(self, window: int = 100):
        self.ep_returns: List[float] = []
        self.window = window
        self.best_reward = -np.inf

    def update(self, ep_return: float):
        """
        Call this after each episode with the episode's total reward.
        """
        self.ep_returns.append(ep_return)
        if ep_return > self.best_reward:
            self.best_reward = ep_return

    def get_best_reward(self) -> float:
        """
        Returns the highest episode reward seen so far.
        """
        return self.best_reward

    def get_return_variance(self) -> float:
        """
        Returns variance of the last `window` episode rewards.
        If fewer than `window` episodes, uses all available.
        """
        if not self.ep_returns:
            return 0.0
        recent = self.ep_returns[-self.window:] if len(self.ep_returns) >= self.window else self.ep_returns
        return float(np.var(recent))

    def get_last_returns(self, n: int = 10) -> List[float]:
        """
        Optional: Returns last n episode returns for logging/plotting.
        """
        return self.ep_returns[-n:]


class TrainingTimeProfiler:
    """
    Tracks training time metrics:
    - Total training time
    - Time per episode
    - Average time per episode (over last N episodes)
    """
    def __init__(self, window: int = 100):
        self.start_time: Optional[float] = None
        self.episode_times: List[float] = []
        self.window = window
        self.current_ep_start: Optional[float] = None

    def start_training(self):
        """Call at the start of training."""
        self.start_time = time.time()

    def start_episode(self):
        """Call at the start of each episode."""
        self.current_ep_start = time.time()

    def end_episode(self):
        """Call at the end of each episode."""
        if self.current_ep_start is not None:
            ep_time = time.time() - self.current_ep_start
            self.episode_times.append(ep_time)
            self.current_ep_start = None

    def get_total_time(self) -> float:
        """Returns total training time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_last_episode_time(self) -> float:
        """Returns time taken for the last episode."""
        if not self.episode_times:
            return 0.0
        return self.episode_times[-1]

    def get_avg_episode_time(self) -> float:
        """Returns average time per episode over the last `window` episodes."""
        if not self.episode_times:
            return 0.0
        recent = self.episode_times[-self.window:] if len(self.episode_times) >= self.window else self.episode_times
        return float(np.mean(recent))

    def get_estimated_remaining_time(self, total_episodes: int, current_episode: int) -> float:
        """Estimates remaining training time based on average episode time."""
        if current_episode >= total_episodes:
            return 0.0
        avg_time = self.get_avg_episode_time()
        remaining_eps = total_episodes - current_episode
        return avg_time * remaining_eps


class GradientStatsProfiler:
    """
    Tracks gradient statistics for neural networks:
    - Gradient norms (L2 norm) for policy and value networks
    - Average gradient norms over last N updates
    """
    def __init__(self, window: int = 100):
        self.policy_grad_norms: List[float] = []
        self.value_grad_norms: List[float] = []
        self.window = window

    def compute_grad_norm(self, model: nn.Module) -> float:
        """Computes L2 norm of all gradients in the model."""
        total_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        if param_count == 0:
            return 0.0
        total_norm = total_norm ** (1. / 2)
        return float(total_norm)

    def update_policy_grad(self, policy: nn.Module):
        """Call after policy backward pass to record gradient norm."""
        grad_norm = self.compute_grad_norm(policy)
        self.policy_grad_norms.append(grad_norm)

    def update_value_grad(self, value: nn.Module):
        """Call after value network backward pass to record gradient norm."""
        grad_norm = self.compute_grad_norm(value)
        self.value_grad_norms.append(grad_norm)

    def get_last_policy_grad_norm(self) -> float:
        """Returns the last recorded policy gradient norm."""
        if not self.policy_grad_norms:
            return 0.0
        return self.policy_grad_norms[-1]

    def get_last_value_grad_norm(self) -> float:
        """Returns the last recorded value gradient norm."""
        if not self.value_grad_norms:
            return 0.0
        return self.value_grad_norms[-1]

    def get_avg_policy_grad_norm(self) -> float:
        """Returns average policy gradient norm over last `window` updates."""
        if not self.policy_grad_norms:
            return 0.0
        recent = self.policy_grad_norms[-self.window:] if len(self.policy_grad_norms) >= self.window else self.policy_grad_norms
        return float(np.mean(recent))

    def get_avg_value_grad_norm(self) -> float:
        """Returns average value gradient norm over last `window` updates."""
        if not self.value_grad_norms:
            return 0.0
        recent = self.value_grad_norms[-self.window:] if len(self.value_grad_norms) >= self.window else self.value_grad_norms
        return float(np.mean(recent))
