import numpy as np
import time
from typing import List, Optional
import torch
import torch.nn as nn

#tracks best episode reward and return variance over last n episodes
class RewardMetrics:
    def __init__(self, window: int = 100):
        self.ep_returns: List[float] = []
        self.window = window
        self.best_reward = -np.inf

    #updates metrics w episode return, tracks best reward seen so far
    def update(self, ep_return: float):
        self.ep_returns.append(ep_return)
        if ep_return > self.best_reward:
            self.best_reward = ep_return

    #returns highest episode reward seen so far
    def get_best_reward(self) -> float:
        return self.best_reward

    #returns variance of last window episode rewards, uses all if fewer than window
    def get_return_variance(self) -> float:
        if not self.ep_returns:
            return 0.0
        recent = self.ep_returns[-self.window:] if len(self.ep_returns) >= self.window else self.ep_returns
        return float(np.var(recent))

    #returns last n episode returns for logging or plotting
    def get_last_returns(self, n: int = 10) -> List[float]:
        return self.ep_returns[-n:]


#tracks total training time, time per episode, and average time over last n episodes
class TrainingTimeProfiler:
    def __init__(self, window: int = 100):
        self.start_time: Optional[float] = None
        self.episode_times: List[float] = []
        self.window = window
        self.current_ep_start: Optional[float] = None

    #starts timer at beginning of training
    def start_training(self):
        self.start_time = time.time()

    #starts timer at beginning of each episode
    def start_episode(self):
        self.current_ep_start = time.time()

    #ends episode timer and records episode time
    def end_episode(self):
        if self.current_ep_start is not None:
            ep_time = time.time() - self.current_ep_start
            self.episode_times.append(ep_time)
            self.current_ep_start = None

    #returns total training time in seconds
    def get_total_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    #returns time taken for last episode
    def get_last_episode_time(self) -> float:
        if not self.episode_times:
            return 0.0
        return self.episode_times[-1]

    #returns average time per episode over last window episodes
    def get_avg_episode_time(self) -> float:
        if not self.episode_times:
            return 0.0
        recent = self.episode_times[-self.window:] if len(self.episode_times) >= self.window else self.episode_times
        return float(np.mean(recent))

    #estimates remaining training time based on average episode time
    def get_estimated_remaining_time(self, total_episodes: int, current_episode: int) -> float:
        if current_episode >= total_episodes:
            return 0.0
        avg_time = self.get_avg_episode_time()
        remaining_eps = total_episodes - current_episode
        return avg_time * remaining_eps


#tracks gradient norms for policy and value networks, computes averages over last n updates
class GradientStatsProfiler:
    def __init__(self, window: int = 100):
        self.policy_grad_norms: List[float] = []
        self.value_grad_norms: List[float] = []
        self.window = window

    #computes l2 norm of all gradients in model
    def compute_grad_norm(self, model: nn.Module) -> float:
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

    #records policy gradient norm after backward pass
    def update_policy_grad(self, policy: nn.Module):
        grad_norm = self.compute_grad_norm(policy)
        self.policy_grad_norms.append(grad_norm)

    #records value network gradient norm after backward pass
    def update_value_grad(self, value: nn.Module):
        grad_norm = self.compute_grad_norm(value)
        self.value_grad_norms.append(grad_norm)

    #returns last recorded policy gradient norm
    def get_last_policy_grad_norm(self) -> float:
        if not self.policy_grad_norms:
            return 0.0
        return self.policy_grad_norms[-1]

    #returns last recorded value gradient norm
    def get_last_value_grad_norm(self) -> float:
        if not self.value_grad_norms:
            return 0.0
        return self.value_grad_norms[-1]

    #returns average policy gradient norm over last window updates
    def get_avg_policy_grad_norm(self) -> float:
        if not self.policy_grad_norms:
            return 0.0
        recent = self.policy_grad_norms[-self.window:] if len(self.policy_grad_norms) >= self.window else self.policy_grad_norms
        return float(np.mean(recent))

    #returns average value gradient norm over last window updates
    def get_avg_value_grad_norm(self) -> float:
        if not self.value_grad_norms:
            return 0.0
        recent = self.value_grad_norms[-self.window:] if len(self.value_grad_norms) >= self.window else self.value_grad_norms
        return float(np.mean(recent))
