
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import minigrid  # noqa: F401  # ensure package available for env registry
except Exception as e:
    raise SystemExit(
        "MiniGrid not found. Install with:\n"
        "  pip install minigrid gymnasium[other]  # or gymnasium==0.29.*\n"
        f"Import error: {e}"
    )


class FlattenMiniGridObs(gym.ObservationWrapper):
    """Converts MiniGrid dict observation {image(7x7x3), direction, mission(str)}
    into a single flat float32 vector suitable for MLP policies.
    - image is integer ids in [0..max], we scale to [0,1] by dividing by 10.
    - direction is 0..3 -> one-hot(4)
    - optional: ignore mission text for the network (you can still prompt an LLM with it externally)
    This keeps your REINFORCE code unchanged if it expects a Box(obs_dim,) observation.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Peek space to define new Box
        img_space = env.observation_space["image"]
        H, W, C = img_space.shape
        self._img_size = H * W * C
        # New observation: [flatten(image), onehot_dir(4)]
        low = np.zeros(self._img_size + 4, dtype=np.float32)
        high = np.ones(self._img_size + 4, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        img = obs["image"].astype(np.float32) / 10.0  # scale ids
        flat_img = img.flatten()
        dir_oh = np.zeros(4, dtype=np.float32)
        dir_oh[int(obs["direction"]) % 4] = 1.0
        out = np.concatenate([flat_img, dir_oh], axis=0).astype(np.float32)
        return out


def make_minigrid_env(env_id: str = "MiniGrid-DoorKey-5x5-v0", seed: int = 42, render_mode: str | None = None):
    """Factory that returns a MiniGrid env with flattened observations and discrete actions."""
    env = gym.make(env_id, render_mode=render_mode, max_steps=200)  # keep short horizon
    env.reset(seed=seed)
    env = FlattenMiniGridObs(env)
    return env
