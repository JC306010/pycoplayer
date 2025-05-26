import gymnasium as gym
import keyboard
import ai
import cv2

class ImageEnv():
    def __init__(self):
        pass
    
    def preprocess(self, image):
        image = cv2.resize(image, (84, 84))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0
        
        return image

env = gym.make("CarRacing-v3",
               render_mode="human",
               lap_complete_percent=0.95,
               domain_randomize=False,
               continuous=True) 
max_epoch = 500
learning_rate = 0.01
episodes = 100_000
start_epsilon = 1
epsilon_decay = start_epsilon / (episodes / 2)
end_epsilon = 0.1
input_dims = 96*96*1
hidden_dims = 128
output_dims = 5
dropout = 0.5

agent = ai.DeepQLearning(env, learning_rate, start_epsilon, end_epsilon, epsilon_decay, 0.95)
neural_network = ai.NeuralNetwork(input_dims, hidden_dims, output_dims, dropout)

def car_race():
    for episode in range(episodes):
        observation, info = env.reset()
        done = False
        pressed = False
        
        while (not done):
            action = agent.do_action(observation)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.step(observation, action, reward, next_obs)

            done = terminated or truncated
            
            # if keyboard.is_pressed("q"):
            #     pressed = True
            #     break
            
        agent.decay_epsilon()
            
        if pressed == True:
            break
        
    env.close()