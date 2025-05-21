import gymnasium as gym
import keyboard

env = gym.make("CarRacing-v3",
               render_mode="human",
               domain_randomize=False,
               continuous=False) 

observation, info = env.reset()

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
    
    if keyboard.is_pressed("q"):
        break
    
print(observation, "\n", reward, "\n", terminated, "\n", truncated, "\n", info)
env.close()