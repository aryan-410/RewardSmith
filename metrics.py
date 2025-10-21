import numpy as np

def return_var(ep_rewards, window_size=None):
    """
    Computes the variance of total episode rewards.
    Used to assess consistency of policy performance over training.
    
    Args:
        ep_rewards (list or np.array): total reward per episode
        window_size (int, optional): number of recent episodes to include. 
                                     If None, uses all episodes.
    Returns:
        float: variance of episode returns
    """
    if len(ep_rewards) == 0:
        return 0.0
    
    # Optionally limit to recent eps for smoother tracking
    rewards_window = ep_rewards[-window_size:] if window_size else ep_rewards
    return np.var(rewards_window)


def best_ep_reward(ep_rewards):
    """
    Returns the highest episode reward achieved so far.
    Used to track the peak performance of the current policy.
    
    Args:
        ep_rewards (list or np.array): total reward per ep
    Returns:
        float: maximum total reward
    """
    if len(ep_rewards) == 0:
        return 0.0
    return np.max(ep_rewards)

def run_episode(env, policy, render=False, max_steps=1000):
    """
    Runs one full episode of Pacman using the current policy.

    Args:
        env: The Pacman game environment (must support reset() and step(action))
        policy: Function or model that selects actions given a state
        render (bool): Whether to show the game visually (slows down training)
        max_steps (int): Safety cap for steps per episode

    Returns:
        float: Total reward collected during this episode
    """
    state = env.reset()         # Reset environment at start of episode
    done = False
    total_reward = 0.0

    for step in range(max_steps):
        if render:
            env.render()        # Optional: display game window
        
        # Get action from policy
        action = policy(state)
        
        # Step the environment
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        state = next_state

        if done:
            break

    return total_reward

ep_rewards = []  # stores total reward per ep
env = ...      # your Pacman environment
policy = ...   # your policy function or model

# Example usage in training loop
num_eps = 1000
for ep in range(num_eps):
    total_reward = run_episode(env, policy)  # your episode execution
    ep_rewards.append(total_reward)

    # Compute tracking metrics
    var_return = return_var(ep_rewards, window_size=100)
    best_reward =best_ep_reward(ep_rewards)
    
    print(f"Episode {ep+1} | Total Reward: {total_reward:.2f} | "
          f"Return Variance (100ep): {var_return:.2f} | "
          f"Best Reward: {best_reward:.2f}")