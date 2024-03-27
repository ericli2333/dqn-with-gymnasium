import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack


def make_env(env_name):
    """
    Create and configure an environment for reinforcement learning.

    Parameters:
    - env_name (str): The name of the environment to create.

    Returns:
    - env (gym.Env): The configured environment.
    """

    env = gym.make(env_name)
    env = AtariPreprocessing(env, scale_obs=False, terminal_on_life_loss=True)
    env = FrameStack(env, num_stack=4)
    return env
