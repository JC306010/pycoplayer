from enum import Enum
import gymnasium
import pygame

class Actions(Enum):
    Up = 0,
    Down = 1
    
class Window():
    def __init__(self, width=600, height=150):
        self.fps = 60
        self.width = width
        self.height = height
        self.drop_velocity = -5
    
class DinosaurGame(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        self.t_rex = None
        self.distance = None
        self.time = 0
        self.score = 0
        self.gravity = 0.6
        self.initial_jump_velocity = 1.2
        self.clear_time = 3000
        self.speed = 6
        self.max_speed = 13
        self.acceleration = 0.001
        self.max_jump_height = 35
        self.gameover_clear_time = 750
        self.max_obstacle_duplication = 2
        self.max_obstacle_length = 3
        self.cloud_speed = 0.2
        self.gap_coefficient = 0.6
        self.max_clouds = 6
        self.speed_drop_coefficient = 3
        