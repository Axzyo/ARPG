import pygame
from constants import *

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = PLAYER_SPEED
        
    def update(self, keys, dt):
        # Handle WASD movement
        dx = 0
        dy = 0
        
        if keys[pygame.K_w]:
            dy -= self.speed * dt
        if keys[pygame.K_s]:
            dy += self.speed * dt
        if keys[pygame.K_a]:
            dx -= self.speed * dt
        if keys[pygame.K_d]:
            dx += self.speed * dt
        
        # Normalize diagonal movement
        if dx != 0 and dy != 0:
            dx *= 0.707  # 1/sqrt(2)
            dy *= 0.707
        
        # Update position
        self.x += dx
        self.y += dy
    
    def render(self, screen, camera_x, camera_y):
        # Calculate screen position relative to camera
        screen_x = self.x - camera_x
        screen_y = self.y - camera_y
        
        # Only render if on screen
        if (-PLAYER_SIZE <= screen_x <= SCREEN_WIDTH + PLAYER_SIZE and 
            -PLAYER_SIZE <= screen_y <= SCREEN_HEIGHT + PLAYER_SIZE):
            
            # Draw player as a circle
            pygame.draw.circle(screen, PLAYER_COLOR, 
                             (int(screen_x), int(screen_y)), 
                             PLAYER_SIZE // 2)
            
            # Draw a small white center dot
            pygame.draw.circle(screen, WHITE, 
                             (int(screen_x), int(screen_y)), 
                             3) 