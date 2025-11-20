
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import minigrid  # noqa: F401
except Exception as e:
    raise SystemExit(
        "MiniGrid not found. Install with:\n"
        "  pip install minigrid gymnasium[other]  # or gymnasium==0.29.*\n"
        f"Import error: {e}"
    )


#converts minigrid dict observation into flat float vector, scales image ids and one-hots direction
class FlattenMiniGridObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        img_space = env.observation_space["image"]
        H, W, C = img_space.shape
        self._img_size = H * W * C
        low = np.zeros(self._img_size + 4, dtype=np.float32)
        high = np.ones(self._img_size + 4, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    #flattens image and converts direction to one-hot, scales image values
    def observation(self, obs):
        img = obs["image"].astype(np.float32) / 10.0
        flat_img = img.flatten()
        dir_oh = np.zeros(4, dtype=np.float32)
        dir_oh[int(obs["direction"]) % 4] = 1.0
        out = np.concatenate([flat_img, dir_oh], axis=0).astype(np.float32)
        return out


#creates minigrid environment w flattened observations and discrete actions
def make_minigrid_env(env_id: str = "MiniGrid-DoorKey-5x5-v0", seed: int = 42, render_mode: str | None = None):
    env = gym.make(env_id, render_mode=render_mode, max_steps=200)
    env.reset(seed=seed)
    env = FlattenMiniGridObs(env)
    return env
