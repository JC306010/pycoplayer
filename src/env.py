import IPython.display
import gymnasium as gym
import ai
import torch
import cv2
import numpy
import IPython
import matplotlib.pyplot as plt
import string
import random

class ImageEnv(gym.Wrapper):
    def __init__(self, env, skip_frames=4, stack_frames=4, **kwargs):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
    
    def preprocess(self, image):
        image = cv2.resize(image, dsize=(84, 84))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0
        
        return image
    
    def reset(self):
        state, info = self.env.reset()
        
        for i in range(50):
            state, reward, terminated, truncated, info = self.env.step(0)

        state = self.preprocess(state)
        self.stacked_frames = numpy.tile(state, (self.stack_frames, 1, 1))
        
        return self.stacked_frames, info
    
    def step(self, action):
        reward = 0

        for i in range(self.skip_frames):
            state, r, terminated, truncated, info = self.env.step(action)
            reward += r

            if (terminated or truncated):
                break

        state = self.preprocess(state)
        self.stacked_frames = numpy.concatenate((self.stacked_frames[1:], state[numpy.newaxis]), axis=0)

        return self.stacked_frames, reward, terminated, truncated, info

env = gym.make("CarRacing-v3",
               continuous=False,
               render_mode="human") 
env = ImageEnv(env)

max_steps = int(2e6)
eval_interval = 10000
state_dims = (4, 84, 84)
action_dims = env.action_space.n

deep_q = ai.DeepQLearning(state_dims, action_dims)
frames = []

def evaulate(n_evals=5):
    eval_env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    eval_env = ImageEnv(eval_env)
    
    scores = 0
    for i in range(n_evals):
        state, _ = eval_env.reset()
        done = False
        reward = 0
        
        scores = 0
        while not done:
            frames.append(eval_env.render())
            action = deep_q.act(state, training=False)
            next_state, r, terminated, truncated, info = eval_env.step(action)
            state = next_state
            reward += r
            
            done = terminated or truncated
            
        scores += reward
        
    return numpy.round(scores / n_evals, 4)

def animate(imgs, video_name, _return=False):
    if video_name is None:
        video_name = ''.join(random.choice(string.ascii_letters) for i in range(18)) + '.webm'
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter.fourcc(*"VP90")
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))
    
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)
    video.release()
    if _return:
        from IPython.display import Video
        return Video(video_name)

history = {
    "Steps": [],
    "AvgReturn": []
}

(state, _) = env.reset()

while True:
    action = deep_q.act(state)
    next_state, r, terminated, truncated, info = env.step(action)
    result = deep_q.process((state, action, r, next_state, terminated))
    
    state = next_state
    if terminated or truncated:
        state, _ = env.reset()
        
    if deep_q.total_steps % eval_interval == 0:
        ret = evaulate()
        history['Steps'].append(deep_q.total_steps)
        history["AvgReturn"].append(ret)
        
        IPython.display.clear_output()
        plt.figure(figsize=(8,5))
        plt.plot(history["Steps"], history["AvgReturn"], 'r-')
        plt.xlabel("Steps", fontsize=16)
        plt.ylabel("AvgReturn", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis='y')
        
        torch.save(deep_q.network.state_dict(), "dqn.pt")
        
    if deep_q.total_steps > max_steps:
        plt.show()
        break
    
animate(frames)