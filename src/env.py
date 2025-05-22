import gymnasium as gym
import keyboard
import ai

learning_rate = 0.01
episodes = 100_000
start_epsilon = 1
epsilon_decay = start_epsilon / (episodes / 2)
end_epsilon = 0.1

env = gym.make("CarRacing-v3",
               render_mode="human",
               domain_randomize=False,
               continuous=False) 
agent = ai.DeepQLearning(env, learning_rate, start_epsilon, end_epsilon, epsilon_decay, 0.95)

for episode in range(episodes):
    observation, info = env.reset()
    done = False
    pressed = False
    
    while (not done):
        action = agent.do_action(observation)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.step(observation, action, reward, next_obs)

        done = terminated or truncated
        
        if keyboard.is_pressed("q"):
            pressed = True
            break
        
    agent.decay_epsilon()
        
    if pressed == True:
        break
    
env.close()