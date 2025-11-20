
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
from minigrid_adapter import make_minigrid_env
from metrics import RewardMetrics, TrainingTimeProfiler, GradientStatsProfiler

#takes what agent sees and outputs score for each action, then, agent picks action w highest score
class PolicyNet(nn.Module):
    #builds network w 2 hidden layers of 128 nodes each, then output layer w one score per action
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, act_dim)
        )
    #runs observation through network and returns action scores
    def forward(self, x):
        return self.net(x)


#takes what agent sees and outputs single number predicting how good this state is
class ValueNet(nn.Module):
    #builds network w 2 hidden layers of 128 nodes each, then output layer w single value
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )
    #runs observation through network and returns predicted state value
    def forward(self, x):
        return self.net(x).squeeze(-1)


#runs agent through one episode, collects all observations, actions, and rewards
def collect_trajectory(env, policy, device, render=False):
    policy.eval()
    obs, _ = env.reset()
    done = False
    obs_list, act_list, rew_list = [], [], []
    ep_len = 0
    while not done:
        if render and env.render_mode == "human":
            env.render()
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = policy(x)
            dist = Categorical(logits=logits)
            action = dist.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        obs_list.append(obs)
        act_list.append(action.item())
        rew_list.append(reward)
        obs = next_obs
        ep_len += 1
    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "acts": np.array(act_list, dtype=np.int64),
        "rews": np.array(rew_list, dtype=np.float32),
        "len": ep_len,
        "ret": float(np.sum(rew_list)),
    }


#takes list of rewards and computes discounted return for each step, later rewards worth less
def compute_returns(rews, gamma):
    G = 0.0
    out = []
    for r in reversed(rews):
        G = r + gamma * G
        out.append(G)
    return np.array(list(reversed(out)), dtype=np.float32)


#main training loop, runs episodes, collects data, updates policy and value networks
def train(args):
    #creates environment, can turn rendering on or off
    def make_env(render=False):
        return make_minigrid_env(
            args.env,
            seed=args.seed,
            render_mode=("human" if render else None),
        )

    env = make_env(render=False)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    policy = PolicyNet(obs_dim, act_dim).to(device)
    value = ValueNet(obs_dim).to(device) if args.use_baseline else None

    opt_pi = optim.Adam(policy.parameters(), lr=args.lr)
    opt_v = optim.Adam(value.parameters(), lr=args.v_lr) if value else None

    reward_metrics = RewardMetrics(window=100)
    time_profiler = TrainingTimeProfiler(window=100)
    grad_profiler = GradientStatsProfiler(window=100)
    
    time_profiler.start_training()
    best_ret = -1e9
    for ep in range(1, args.episodes + 1):
        time_profiler.start_episode()
        render_now = args.render_every > 0 and (ep % args.render_every == 0)

        if render_now and (getattr(env, "render_mode", None) != "human"):
            env.close()
            env = make_env(render=True)

        if (not render_now) and (getattr(env, "render_mode", None) == "human"):
            env.close()
            env = make_env(render=False)

        traj = collect_trajectory(env, policy, device, render=render_now)
        time_profiler.end_episode()
        
        reward_metrics.update(traj["ret"])

        G = compute_returns(traj["rews"], args.gamma)
        if args.reward_norm:
            G = (G - G.mean()) / (G.std() + 1e-8)
        obs_t = torch.as_tensor(traj["obs"], dtype=torch.float32, device=device)
        acts_t = torch.as_tensor(traj["acts"], dtype=torch.int64, device=device)

        policy.train()
        logits = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        logps_t = dist.log_prob(acts_t)
        entropy = dist.entropy().mean()

        if value is None:
            advantages = torch.as_tensor(G, dtype=torch.float32, device=device)
        else:
            value.train()
            with torch.no_grad():
                V = value(obs_t).detach().cpu().numpy()
            adv = G - V
            if args.adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            advantages = torch.as_tensor(adv, dtype=torch.float32, device=device)

        loss_pi = -(advantages.detach() * logps_t).mean() - args.entropy_coef * entropy
        opt_pi.zero_grad(set_to_none=True)
        loss_pi.backward()
        grad_profiler.update_policy_grad(policy)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_pi.step()

        if value is not None:
            target = torch.as_tensor(G, dtype=torch.float32, device=device)
            pred_v = value(obs_t)
            loss_v = torch.nn.functional.mse_loss(pred_v, target)
            opt_v.zero_grad(set_to_none=True)
            loss_v.backward()
            grad_profiler.update_value_grad(value)
            torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
            opt_v.step()
        else:
            loss_v = torch.tensor(0.0)

        if ep % args.log_every == 0:
            avg_r = traj["ret"]
            best_ret = max(best_ret, avg_r)
            
            return_var = reward_metrics.get_return_variance()
            total_time = time_profiler.get_total_time()
            avg_ep_time = time_profiler.get_avg_episode_time()
            policy_grad_norm = grad_profiler.get_last_policy_grad_norm()
            value_grad_norm = grad_profiler.get_last_value_grad_norm() if value else 0.0
            
            print(
                f"ep {ep:5d} | len {traj['len']:3d} | ret {avg_r:6.2f} | "
                f"L_pi {loss_pi.item():.3f} | L_v {loss_v.item():.3f} | H {entropy.item():.3f} | "
                f"best {best_ret:.2f} | var {return_var:.3f} | "
                f"time {total_time:.1f}s | ep_time {avg_ep_time:.3f}s | "
                f"grad_pi {policy_grad_norm:.3f} | grad_v {value_grad_norm:.3f}"
            )

    env.close()


#reads command line arguments like number of episodes, learning rate, etc
def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE on MiniGrid via flat-obs adapter")
    p.add_argument("--env", type=str, default="MiniGrid-DoorKey-5x5-v0")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--lr", type=float, default=3e-4, help="Policy learning rate")
    p.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient")
    p.add_argument("--use_baseline", action="store_true", help="Use value function baseline")
    p.add_argument("--v_lr", type=float, default=3e-3, help="Value net learning rate")
    p.add_argument("--reward_norm", action="store_true", help="Normalize returns within batch")
    p.add_argument("--adv_norm", action="store_true", help="Normalize advantages within batch")
    p.add_argument("--render_every", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
