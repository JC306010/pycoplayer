import gym
import gym_chrome_dino
import gym_chrome_dino.utils
import gym_chrome_dino.utils.wrappers

env = gym.make("ChromeDino-v0")
env = gym_chrome_dino.utils.wrappers.make_dino(env, True, True)