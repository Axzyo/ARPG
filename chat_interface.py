import pygame
from constants import *

class ChatInterface:
    def __init__(self):
        self.active = False
        self.input_text = ""
        self.cursor_position = 0  # Position of cursor in text
        self.target_npc = None
        self.font = pygame.font.Font(None, FONT_MEDIUM)
        
    def open_chat(self, npc):
        """Open chat interface for a specific NPC"""
        self.active = True
        self.input_text = ""
        self.cursor_position = 0
        self.target_npc = npc
        
    def close_chat(self):
        """Close the chat interface"""
        self.active = False
        self.input_text = ""
        self.cursor_position = 0
        self.target_npc = None
        
    def handle_event(self, event):
        """Handle input events for the chat interface"""
        if not self.active:
            return None
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Submit message
                if self.input_text.strip():
                    message = self.input_text.strip()
                    target_npc = self.target_npc  # Capture before closing
                    self.close_chat()
                    return {"message": message, "target_npc": target_npc}
            elif event.key == pygame.K_ESCAPE:
                # Cancel chat
                self.close_chat()
            elif event.key == pygame.K_LEFT:
                # Move cursor left
                self.cursor_position = max(0, self.cursor_position - 1)
            elif event.key == pygame.K_RIGHT:
                # Move cursor right
                self.cursor_position = min(len(self.input_text), self.cursor_position + 1)
            elif event.key == pygame.K_HOME:
                # Move cursor to beginning
                self.cursor_position = 0
            elif event.key == pygame.K_END:
                # Move cursor to end
                self.cursor_position = len(self.input_text)
            elif event.key == pygame.K_BACKSPACE:
                # Delete character before cursor
                if self.cursor_position > 0:
                    self.input_text = (self.input_text[:self.cursor_position-1] + 
                                     self.input_text[self.cursor_position:])
                    self.cursor_position -= 1
            elif event.key == pygame.K_DELETE:
                # Delete character after cursor
                if self.cursor_position < len(self.input_text):
                    self.input_text = (self.input_text[:self.cursor_position] + 
                                     self.input_text[self.cursor_position+1:])
            else:
                # Add character at cursor position
                if event.unicode.isprintable():
                    self.input_text = (self.input_text[:self.cursor_position] + 
                                     event.unicode + 
                                     self.input_text[self.cursor_position:])
                    self.cursor_position += 1
                    
        return None
        
    def render(self, screen):
        """Render the chat interface"""
        if not self.active:
            return
            
        # Draw input box background
        input_box_height = max(30, int(40 * SCALE))
        input_box_y = SCREEN_HEIGHT - input_box_height - UI_LARGE_SPACING
        input_box = pygame.Rect(UI_LARGE_SPACING, input_box_y, SCREEN_WIDTH - (UI_LARGE_SPACING * 2), input_box_height)
        
        pygame.draw.rect(screen, (40, 40, 40), input_box)
        pygame.draw.rect(screen, WHITE, input_box, width=2)
        
        # Draw prompt text
        prompt_text = f"Talking to {self.target_npc.name if self.target_npc else 'NPC'}:"
        prompt_surface = self.font.render(prompt_text, True, WHITE)
        screen.blit(prompt_surface, (UI_LARGE_SPACING + UI_SMALL_SPACING, input_box_y - UI_LARGE_SPACING))
        
        # Draw input text
        display_text = self.input_text
        if len(display_text) > 60:  # Limit visible characters
            display_text = "..." + display_text[-57:]
            
        text_surface = self.font.render(display_text, True, WHITE)
        screen.blit(text_surface, (UI_LARGE_SPACING + UI_SMALL_SPACING, input_box_y + UI_MARGIN))
        
        # Draw cursor at correct position
        if pygame.time.get_ticks() % 1000 < 500:  # Blinking cursor
            # Calculate cursor x position
            cursor_text_x = UI_LARGE_SPACING + UI_SMALL_SPACING
            if len(display_text) > 60:
                # If text is truncated, cursor is at the end
                cursor_x = cursor_text_x + text_surface.get_width()
            else:
                # Calculate actual cursor position
                text_before_cursor = self.input_text[:self.cursor_position]
                cursor_text_surface = self.font.render(text_before_cursor, True, WHITE)
                cursor_x = cursor_text_x + cursor_text_surface.get_width()
            
            pygame.draw.line(screen, WHITE, 
                           (cursor_x, input_box_y + UI_PADDING), 
                           (cursor_x, input_box_y + input_box_height - UI_PADDING), 2)
        
        # Draw instructions
        instructions = "Press ENTER to send, ESC to cancel, Arrow keys to navigate"
        inst_surface = pygame.font.Font(None, FONT_SMALL).render(instructions, True, (150, 150, 150))
        screen.blit(inst_surface, (UI_LARGE_SPACING + UI_SMALL_SPACING, input_box_y + input_box_height + UI_SMALL_SPACING)) 