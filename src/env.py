import gymnasium as gym
import torch
import keyboard
import ai
import cv2
import numpy

class ImageEnv(gym.Wrapper):
    def __init__(self, env, skip_frames=4, stack_frames=4, **kwargs):
        super.__init__(ImageEnv, self).__init__(env, **kwargs)
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
    
    def preprocess(self, image):
        image = cv2.resize(image, (84, 84))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0
        
        return image
    
    def reset(self):
        state, info = env.reset()
        
        for i in range(50):
            state, reward, terminated, truncated, info = self.env.step(0)

        state = self.preprocess(state)
        self.stack_frames = numpy.tile(state, (self.stack_frames, 1, 1))
        
        return self.stack_frames, info
    
    def step(self, action):
        reward = 0

        for i in range(self.skip_frames):
            state, r, terminated, truncated, info = self.env.step(action)
            reward += r

            if (terminated or truncated):
                break

        state = self.preprocess(state)
        self.stack_frames = numpy.concatenate((self.stack_frames[1:], state[numpy.newaxis]), axis=0)

        return self.stack_frames, reward, terminated, truncated, info

env = gym.make("CarRacing-v3",
               continuous=False) 
env = ImageEnv(env)

state, info = env.reset()
print(state.shape)
