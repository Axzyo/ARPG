import pygame
from constants import *

class InteractionPopup:
    def __init__(self):
        """Initialize interaction popup system"""
        self.font = pygame.font.Font(None, FONT_MEDIUM)
        self.small_font = pygame.font.Font(None, FONT_NORMAL)
        self.active_popups = []  # List of currently active popups
        
    def add_popup(self, text, world_x, world_y, popup_type="interact"):
        """
        Add a popup at the specified world position
        
        Args:
            text: The text to display (e.g., "Press E to talk")
            world_x, world_y: World coordinates where popup should appear
            popup_type: Type of popup ("interact", "info", etc.)
        """
        popup = {
            "text": text,
            "world_x": world_x,
            "world_y": world_y,
            "type": popup_type,
            "alpha": 255,
            "bob_offset": 0,  # For floating animation
            "timer": 0
        }
        
        # Remove existing popup at similar location to avoid duplicates
        min_distance = NPC_INTERACTION_DISTANCE
        self.active_popups = [p for p in self.active_popups 
                            if abs(p["world_x"] - world_x) > min_distance or abs(p["world_y"] - world_y) > min_distance]
        
        self.active_popups.append(popup)
    
    def clear_popups(self):
        """Clear all active popups"""
        self.active_popups.clear()
    
    def update(self, dt):
        """Update popup animations and lifetime"""
        for popup in self.active_popups[:]:  # Copy list to allow removal during iteration
            popup["timer"] += dt
            
            # Floating bob animation
            popup["bob_offset"] = int(5 * pygame.math.Vector2(0, 1).rotate(popup["timer"] * 180).y)
            
            # Fade out after 3 seconds (optional - can be removed for persistent popups)
            # if popup["timer"] > 3.0:
            #     popup["alpha"] = max(0, popup["alpha"] - 255 * dt)
            #     if popup["alpha"] <= 0:
            #         self.active_popups.remove(popup)
    
    def render(self, screen, camera_x, camera_y):
        """
        Render all active popups
        
        Args:
            screen: Pygame surface to render to
            camera_x, camera_y: Camera offset for world-to-screen conversion
        """
        for popup in self.active_popups:
            # Convert world coordinates to screen coordinates
            screen_x = popup["world_x"] - camera_x
            screen_y = popup["world_y"] - camera_y + popup["bob_offset"]
            
            # Skip if popup is off-screen
            if screen_x < -100 or screen_x > screen.get_width() + 100:
                continue
            if screen_y < -50 or screen_y > screen.get_height() + 50:
                continue
            
            # Choose colors based on popup type
            if popup["type"] == "interact":
                bg_color = (50, 50, 50, 200)  # Dark background
                text_color = (255, 255, 100)  # Yellow text
                border_color = (255, 255, 255, 150)  # White border
            else:
                bg_color = (40, 40, 60, 200)  # Slightly blue background
                text_color = (200, 200, 255)  # Light blue text
                border_color = (150, 150, 255, 150)  # Blue border
            
            # Render text
            text_surface = self.font.render(popup["text"], True, text_color)
            text_rect = text_surface.get_rect()
            
            # Create background with padding
            padding = UI_PADDING
            bg_rect = pygame.Rect(
                screen_x - text_rect.width // 2 - padding,
                screen_y - text_rect.height - padding,
                text_rect.width + padding * 2,
                text_rect.height + padding * 2
            )
            
            # Create surfaces with alpha for transparency
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill((*bg_color[:3], int(bg_color[3] * popup["alpha"] / 255)))
            
            # Draw background
            screen.blit(bg_surface, bg_rect.topleft)
            
            # Draw border
            pygame.draw.rect(screen, (*border_color[:3], int(border_color[3] * popup["alpha"] / 255)), 
                           bg_rect, 2)
            
            # Draw text
            text_surface.set_alpha(popup["alpha"])
            text_rect.center = (screen_x, screen_y - text_rect.height // 2)
            screen.blit(text_surface, text_rect)
    
    def get_nearby_popup(self, world_x, world_y, max_distance=50):
        """
        Check if there's a popup near the specified world coordinates
        
        Returns:
            The popup dict if found, None otherwise
        """
        for popup in self.active_popups:
            distance = ((popup["world_x"] - world_x) ** 2 + (popup["world_y"] - world_y) ** 2) ** 0.5
            if distance <= max_distance:
                return popup
        return None 