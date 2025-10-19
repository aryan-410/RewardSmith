
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
from minigrid_adapter import make_minigrid_env

# ---------- Simple REINFORCE matching your flags ----------

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, act_dim)
        )
    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def collect_trajectory(env, policy, device, render=False):
    obs, _ = env.reset()
    done = False
    obs_list, act_list, logp_list, rew_list = [], [], [], []
    ep_len = 0
    while not done:
        if render and env.render_mode == "human":
            env.render()
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        obs_list.append(obs)
        act_list.append(action.item())
        logp_list.append(logp)
        rew_list.append(reward)
        obs = next_obs
        ep_len += 1
    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "acts": np.array(act_list, dtype=np.int64),
        "logps": torch.stack(logp_list),
        "rews": np.array(rew_list, dtype=np.float32),
        "len": ep_len,
        "ret": float(np.sum(rew_list)),
    }


def compute_returns(rews, gamma):
    G = 0.0
    out = []
    for r in reversed(rews):
        G = r + gamma * G
        out.append(G)
    return np.array(list(reversed(out)), dtype=np.float32)


def train(args):
    # --- helpers to (re)make envs with/without rendering ---
    def make_env(render=False):
        return make_minigrid_env(
            args.env,
            seed=args.seed,
            render_mode=("human" if render else None),
        )

    # start headless
    env = make_env(render=False)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    policy = PolicyNet(obs_dim, act_dim).to(device)
    value = ValueNet(obs_dim).to(device) if args.use_baseline else None

    opt_pi = optim.Adam(policy.parameters(), lr=args.lr)
    opt_v = optim.Adam(value.parameters(), lr=args.v_lr) if value else None

    best_ret = -1e9
    for ep in range(1, args.episodes + 1):
        # toggle rendering by recreating the env exactly when needed
        render_now = args.render_every > 0 and (ep % args.render_every == 0)

        # if we want to render this episode but current env is headless -> recreate as human
        if render_now and (getattr(env, "render_mode", None) != "human"):
            env.close()
            env = make_env(render=True)

        # if we do NOT want to render but current env is human -> recreate headless
        if (not render_now) and (getattr(env, "render_mode", None) == "human"):
            env.close()
            env = make_env(render=False)

        traj = collect_trajectory(env, policy, device, render=render_now)

        G = compute_returns(traj["rews"], args.gamma)
        if args.reward_norm:
            G = (G - G.mean()) / (G.std() + 1e-8)
        obs_t = torch.as_tensor(traj["obs"], dtype=torch.float32, device=device)
        logps_t = traj["logps"].to(device)

        if value is None:
            advantages = torch.as_tensor(G, dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                V = value(obs_t).detach().cpu().numpy()
            adv = G - V
            if args.adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            advantages = torch.as_tensor(adv, dtype=torch.float32, device=device)

        loss_pi = -(advantages * logps_t).sum()
        opt_pi.zero_grad(set_to_none=True); loss_pi.backward(); opt_pi.step()

        if value is not None:
            target = torch.as_tensor(G, dtype=torch.float32, device=device)
            pred_v = value(obs_t)
            loss_v = torch.nn.functional.mse_loss(pred_v, target)
            opt_v.zero_grad(set_to_none=True); loss_v.backward(); opt_v.step()
        else:
            loss_v = torch.tensor(0.0)

        if ep % args.log_every == 0:
            avg_r = traj["ret"]; best_ret = max(best_ret, avg_r)
            print(
                f"ep {ep:5d} | len {traj['len']:3d} | ret {avg_r:6.2f} | "
                f"L_pi {loss_pi.item():.3f} | L_v {loss_v.item():.3f} | best {best_ret:.2f}"
            )

    env.close()


def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE on MiniGrid via flat-obs adapter")
    p.add_argument("--env", type=str, default="MiniGrid-DoorKey-5x5-v0")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--use_baseline", action="store_true")
    p.add_argument("--v_lr", type=float, default=3e-3)
    p.add_argument("--reward_norm", action="store_true")
    p.add_argument("--adv_norm", action="store_true")
    p.add_argument("--render_every", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
