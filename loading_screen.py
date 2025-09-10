import pygame
import math
from constants import *

class LoadingScreen:
    def __init__(self, screen):
        self.screen = screen
        self.font_title = pygame.font.Font(None, FONT_TITLE)
        self.font_text = pygame.font.Font(None, FONT_MEDIUM)
        self.font_small = pygame.font.Font(None, FONT_SMALL)
        
        # Loading state
        self.progress = 0.0  # 0.0 to 1.0
        self.current_task = "Initializing..."
        self.tasks_completed = 0
        self.total_tasks = 5  # Adjust based on actual loading tasks
        
        # Animation
        self.time = 0.0
        self.dot_count = 0
        
    def update(self, dt):
        """Update loading screen animations"""
        self.time += dt
        self.dot_count = int(self.time * 2) % 4  # Animate dots
        
    def set_progress(self, progress, task_description="Loading..."):
        """Set loading progress (0.0 to 1.0) and current task"""
        self.progress = max(0.0, min(1.0, progress))
        self.current_task = task_description
        
    def set_task_progress(self, completed_tasks, task_description="Loading..."):
        """Set progress based on completed tasks"""
        self.tasks_completed = completed_tasks
        self.progress = completed_tasks / self.total_tasks
        self.current_task = task_description
        
    def render(self):
        """Render the loading screen"""
        # Clear screen with dark background
        self.screen.fill((20, 20, 30))
        
        # Calculate center positions
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        
        # Draw title
        title_text = "Infinite Rouge"
        title_surface = self.font_title.render(title_text, True, (200, 200, 255))
        title_rect = title_surface.get_rect(center=(center_x, center_y - 100))
        self.screen.blit(title_surface, title_rect)
        
        # Draw subtitle
        subtitle_text = "Procedural RPG with AI NPCs"
        subtitle_surface = self.font_small.render(subtitle_text, True, (150, 150, 200))
        subtitle_rect = subtitle_surface.get_rect(center=(center_x, center_y - 70))
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Draw loading bar background
        bar_width = 400
        bar_height = 20
        bar_x = center_x - bar_width // 2
        bar_y = center_y - 10
        
        # Background bar
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.screen, (60, 60, 80), bg_rect)
        pygame.draw.rect(self.screen, (100, 100, 120), bg_rect, 2)
        
        # Progress bar
        if self.progress > 0:
            progress_width = int(bar_width * self.progress)
            progress_rect = pygame.Rect(bar_x, bar_y, progress_width, bar_height)
            
            # Gradient effect for progress bar
            for i in range(progress_width):
                progress = i / bar_width
                color_intensity = int(100 + 100 * progress)
                color = (color_intensity, color_intensity // 2, 255)
                line_rect = pygame.Rect(bar_x + i, bar_y, 1, bar_height)
                pygame.draw.rect(self.screen, color, line_rect)
        
        # Draw percentage
        percentage = int(self.progress * 100)
        percent_text = f"{percentage}%"
        percent_surface = self.font_text.render(percent_text, True, WHITE)
        percent_rect = percent_surface.get_rect(center=(center_x, bar_y + bar_height + 30))
        self.screen.blit(percent_surface, percent_rect)
        
        # Draw current task with animated dots
        dots = "." * self.dot_count
        task_text = f"{self.current_task}{dots}"
        task_surface = self.font_text.render(task_text, True, (180, 180, 180))
        task_rect = task_surface.get_rect(center=(center_x, bar_y + bar_height + 60))
        self.screen.blit(task_surface, task_rect)
        
        # Draw task counter
        task_counter_text = f"({self.tasks_completed}/{self.total_tasks} tasks completed)"
        task_counter_surface = self.font_small.render(task_counter_text, True, (120, 120, 120))
        task_counter_rect = task_counter_surface.get_rect(center=(center_x, bar_y + bar_height + 85))
        self.screen.blit(task_counter_surface, task_counter_rect)
        
        # Draw spinning loading indicator
        spinner_radius = 15
        spinner_x = center_x - 200
        spinner_y = center_y
        
        # Draw spinner dots
        for i in range(8):
            angle = (self.time * 2 + i * math.pi / 4) % (2 * math.pi)
            dot_x = spinner_x + math.cos(angle) * spinner_radius
            dot_y = spinner_y + math.sin(angle) * spinner_radius
            
            # Fade dots based on position
            alpha = (math.sin(angle + self.time * 2) + 1) / 2
            color_value = int(100 + 100 * alpha)
            color = (color_value, color_value, 255)
            
            pygame.draw.circle(self.screen, color, (int(dot_x), int(dot_y)), 3)
        
        # Draw loading tips
        tips = [
            "Tip: Press 'E' to interact with NPCs",
            "Tip: NPCs have memories and will remember you",
            "Tip: Press 'V' to toggle NPC vision display",
            "Tip: NPCs use AI to respond intelligently",
            "Tip: The world is procedurally generated"
        ]
        
        tip_index = int(self.time * 0.5) % len(tips)
        tip_text = tips[tip_index]
        tip_surface = self.font_small.render(tip_text, True, (100, 150, 100))
        tip_rect = tip_surface.get_rect(center=(center_x, SCREEN_HEIGHT - 50))
        self.screen.blit(tip_surface, tip_rect)
        
        pygame.display.flip()
    
    def is_complete(self):
        """Check if loading is complete"""
        return self.progress >= 1.0 