import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Environment: Gymnasium (not an RL helper lib, only the simulator)
try:
    import gymnasium as gym
except ImportError:
    raise SystemExit("Please install gymnasium: pip install gymnasium")

# -------------------------------
# (Optional) TinyCartPole stub
# -------------------------------
# Placeholder for from-scratch CartPole if avoiding gym completely.
class TinyCartPole:
    def __init__(self, seed: int = 0):
        self.np_random = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.state = None
        self.t = 0

    def reset(self, seed=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.state = self.np_random.normal(0.0, 0.05, size=(4,)).astype(np.float32)
        self.t = 0
        return self.state, {}

    def step(self, action: int):
        s = self.state
        s = s + self.np_random.normal(0.0, 0.02, size=(4,)).astype(np.float32) + (1 if action == 1 else -1)*0.005
        self.state = s
        self.t += 1
        reward = 1.0
        terminated = self.t >= 200
        truncated = False
        return s, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass


# -------------------------------
# Networks
# -------------------------------

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits

class ValueNet(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class Trajectory:
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    logps: List[float]
    dones: List[bool]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def discount_cumsum(rewards: List[float], gamma: float) -> np.ndarray:
    """Compute reward-to-go (discounted) returns G_t for one episode."""
    G = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        G[t] = g
    return G

def collect_episodes(env, policy: PolicyNet, episodes: int, render_every: int, device: torch.device) -> Tuple[List[Trajectory], float]:
    policy.eval()
    trajs: List[Trajectory] = []
    ep_returns = []

    for ep in range(episodes):
        s, _ = env.reset()
        states, actions, rewards, logps, dones = [], [], [], [], []
        ep_ret = 0.0

        while True:
            if render_every and ((ep + 1) % render_every == 0):
                try:
                    env.render()
                except Exception:
                    pass

            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(st)
            dist = Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)

            s2, r, terminated, truncated, _ = env.step(a.item())
            done = terminated or truncated

            states.append(s)
            actions.append(int(a.item()))
            rewards.append(float(r))
            logps.append(float(logp.item()))
            dones.append(done)

            ep_ret += float(r)
            s = s2

            if done:
                break

        trajs.append(Trajectory(states, actions, rewards, logps, dones))
        ep_returns.append(ep_ret)

    avg_return = float(np.mean(ep_returns)) if ep_returns else 0.0
    return trajs, avg_return

def compute_batch_tensors(trajs: List[Trajectory], gamma: float, device: torch.device, reward_norm: bool):
    states = np.concatenate([np.array(tr.states, dtype=np.float32) for tr in trajs], axis=0)
    actions = np.concatenate([np.array(tr.actions, dtype=np.int64) for tr in trajs], axis=0)
    logps  = np.concatenate([np.array(tr.logps,  dtype=np.float32) for tr in trajs], axis=0)
    returns = np.concatenate([discount_cumsum(tr.rewards, gamma) for tr in trajs], axis=0).astype(np.float32)

    if reward_norm:
        m, s = returns.mean(), returns.std() + 1e-8
        returns = (returns - m) / s

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
    logps_t   = torch.tensor(logps,  dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    return states_t, actions_t, logps_t, returns_t

def train(args):
    set_seed(args.seed)
    print("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env = gym.make("CartPole-v1", render_mode="human" if args.render_every else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim).to(device)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=args.lr)

    if args.use_baseline:
        value_net = ValueNet(obs_dim).to(device)
        opt_v = torch.optim.Adam(value_net.parameters(), lr=args.v_lr)
    else:
        value_net = None
        opt_v = None

    best_avg = -1e9
    total_steps = 0

    for episode in range(1, args.episodes + 1):
        trajs, avg_epret = collect_episodes(env, policy, episodes=1, render_every=args.render_every, device=device)
        total_steps += sum(len(tr.rewards) for tr in trajs)

        states, actions, old_logps, returns = compute_batch_tensors(trajs, args.gamma, device, args.reward_norm)

        policy.train()
        logits = policy(states)
        dist = Categorical(logits=logits)
        new_logps = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        if args.use_baseline:
            value_net.train()
            with torch.no_grad():
                V = value_net(states)
            adv = returns - V
        else:
            adv = returns.clone()

        if args.adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        loss_pi = -(adv.detach() * new_logps).mean() - args.entropy_coef * entropy
        opt_pi.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_pi.step()

        if args.use_baseline:
            V_pred = value_net(states)
            loss_v = F.mse_loss(V_pred, returns)
            opt_v.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            opt_v.step()
        else:
            loss_v = torch.tensor(0.0)

        if avg_epret > best_avg:
            best_avg = avg_epret

        if episode % max(1, args.log_every) == 0:
            print(f"Ep {episode:4d} | avg_return {avg_epret:7.2f} | "
                  f"loss_pi {float(loss_pi):.3f} | loss_v {float(loss_v):.3f} | "
                  f"steps {total_steps} | best_avg {best_avg:.2f}")

    env.close()


def parse_args():
    p = argparse.ArgumentParser(description="Vanilla REINFORCE on CartPole (no RL libs)")
    p.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--lr", type=float, default=3e-3, help="Policy learning rate")
    p.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient")
    p.add_argument("--use_baseline", action="store_true", help="Use value function baseline")
    p.add_argument("--v_lr", type=float, default=3e-3, help="Value net learning rate")
    p.add_argument("--reward_norm", action="store_true", help="Normalize returns within batch")
    p.add_argument("--adv_norm", action="store_true", help="Normalize advantages within batch")
    p.add_argument("--render_every", type=int, default=0, help="Render an episode every N episodes (0 = never)")
    p.add_argument("--log_every", type=int, default=10, help="Print logs every N episodes")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
