import gymnasium as gym

env = gym.make("CarRacing-v3",
               render_mode="human",
               domain_randomize=False,
               continuous=False) 

observation, info = env.reset()

while True:
    action = env.action_space.sample()  # Random actions
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
    
env.close()