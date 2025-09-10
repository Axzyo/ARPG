import pygame
import sys
import threading
import time
from player import Player
from chunk_manager import ChunkManager
from npc import NPC
from loading_screen import LoadingScreen
from constants import *

class Game:
    def __init__(self):
        # Initialize Pygame first
        pygame.init()
        
        # Set up fullscreen display
        self.fullscreen = True
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Infinite Rouge - Procedural RPG")
        self.clock = pygame.time.Clock()
        
        # Initialize loading screen
        self.loading_screen = LoadingScreen(self.screen)
        self.loading_complete = False
        self.loading_tasks = {
            'pygame': False,
            'player': False,
            'chunks': False,
            'npcs': False,
            'llm': False
        }
        
        # Game objects (will be initialized during loading)
        self.player = None
        self.chunk_manager = None
        self.npcs = []
        
        
        # Game state
        self.camera_x = 0
        self.camera_y = 0
        self.show_npc_vision = True
        
        # Start asynchronous loading
        self._start_async_loading()
    
    def _start_async_loading(self):
        """Start asynchronous loading of game components"""
        def _loading_worker():
            try:
                # Task 1: Pygame (already done)
                self.loading_tasks['pygame'] = True
                self._update_loading_progress("Pygame initialized", 1)
                
                # Task 2: Initialize player
                time.sleep(0.1)  # Small delay to show loading
                self.player = Player(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                self.loading_tasks['player'] = True
                self._update_loading_progress("Player created", 2)
                
                # Task 3: Initialize chunk manager
                time.sleep(0.1)
                self.chunk_manager = ChunkManager()
                self.loading_tasks['chunks'] = True
                self._update_loading_progress("World generator ready", 3)
                
                # Task 4: Create NPCs
                time.sleep(0.1)
                test_npc = NPC("Bob", SCREEN_WIDTH // 2 + 150, SCREEN_HEIGHT // 2 + 100)
                test_npc.set_facing_direction(225)
                test_npc.add_goal("patrol area", priority=3)
                self.npcs = [test_npc]
                self.loading_tasks['npcs'] = True
                self._update_loading_progress("NPCs ready", 4)
                self.loading_complete = True
                
            except Exception as e:
                print(f"[LOADING] Error during loading: {e}")
                # Continue with loading even if there's an error
                self.loading_complete = True
        
        # Start loading in background thread
        loading_thread = threading.Thread(target=_loading_worker, daemon=True)
        loading_thread.start()
    
    def _update_loading_progress(self, task_description, completed_tasks):
        """Update loading progress safely from background thread"""
        if hasattr(self.loading_screen, 'set_task_progress'):
            self.loading_screen.set_task_progress(completed_tasks, task_description)
    
    def _generate_npc_backstories(self):
        """Deprecated: backstories are now represented as memories. Keeping function for compatibility."""
        print(f"[GAME] Skipping backstory generation (memory-driven NPCs)")
    
    def _run_loading_screen(self):
        """Run the loading screen loop"""
        while not self.loading_complete:
            dt = self.clock.tick(60) / 1000.0  # 60 FPS for smooth loading screen
            
            # Handle events (mainly to prevent Windows from thinking the app is frozen)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Update and render loading screen
            self.loading_screen.update(dt)
            self.loading_screen.render()
        
        # Loading complete, show final screen briefly
        self.loading_screen.set_task_progress(4, "Ready!")
        self.loading_screen.render()
        time.sleep(0.5)  # Brief pause to show completion
        
    def handle_events(self):
        # If still loading, don't process game events
        if not self.loading_complete:
            return True
            
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_v:
                    # Toggle NPC vision display
                    self.show_npc_vision = not self.show_npc_vision
                    print(f"NPC Vision Display: {'ON' if self.show_npc_vision else 'OFF'}")
                elif event.key == pygame.K_F11:
                    # Toggle fullscreen
                    self._toggle_fullscreen()
                
        return True
    
    def _toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        self.fullscreen = not self.fullscreen
        
        if self.fullscreen:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
            print("[DISPLAY] Switched to fullscreen mode")
        else:
            # Use reference resolution for windowed mode
            windowed_width = min(REFERENCE_WIDTH, SCREEN_WIDTH)
            windowed_height = min(REFERENCE_HEIGHT, SCREEN_HEIGHT)
            self.screen = pygame.display.set_mode((windowed_width, windowed_height))
            print(f"[DISPLAY] Switched to windowed mode ({windowed_width}x{windowed_height})")
    
    
    def update(self, dt):
        # Don't update game state while loading
        if not self.loading_complete:
            return
            
        # Handle player input and movement
        keys = pygame.key.get_pressed()
        self.player.update(keys, dt)
        
        # Update camera to follow player
        self.camera_x = self.player.x - SCREEN_WIDTH // 2
        self.camera_y = self.player.y - SCREEN_HEIGHT // 2
        
        # Update chunk manager based on player position
        self.chunk_manager.update_chunks(self.player.x, self.player.y)
        
        # (Interaction/chat removed)
        
        # Update NPCs
        for npc in self.npcs:
            npc.update(dt)
            
            # Process NPC vision (includes player and other NPCs)
            all_entities = [self.player] + [other_npc for other_npc in self.npcs if other_npc != npc]
            npc.see(self.chunk_manager, all_entities)
    
    def render(self):
        # Show loading screen if not loaded yet
        if not self.loading_complete:
            return  # Loading screen renders itself
            
        self.screen.fill(BLACK)
        
        # Render chunks (terrain)
        self.chunk_manager.render(self.screen, self.camera_x, self.camera_y)
        
        # Render NPCs (before player so player appears on top)
        for npc in self.npcs:
            npc.render(self.screen, self.camera_x, self.camera_y, show_vision=self.show_npc_vision)
        
        # Render player
        self.player.render(self.screen, self.camera_x, self.camera_y)
        
        # Draw debug info
        font = pygame.font.Font(None, FONT_MEDIUM)
        player_chunk_x = self.player.x // (CHUNK_SIZE * TILE_SIZE)
        player_chunk_y = self.player.y // (CHUNK_SIZE * TILE_SIZE)
        debug_text = f"Player: ({self.player.x:.0f}, {self.player.y:.0f}) Chunk: ({player_chunk_x}, {player_chunk_y})"
        text_surface = font.render(debug_text, True, WHITE)
        self.screen.blit(text_surface, (UI_MARGIN, UI_MARGIN))
        
        # Draw NPC info
        y_offset = UI_LARGE_SPACING + UI_MARGIN
        for i, npc in enumerate(self.npcs):
            npc_info = f"{npc.name}: ({npc.x:.0f}, {npc.y:.0f}) Facing: {npc.facing_direction:.0f}° Action: {npc.current_action}"
            npc_text = font.render(npc_info, True, WHITE)
            self.screen.blit(npc_text, (UI_MARGIN, y_offset))
            y_offset += UI_LARGE_SPACING
            
            # Show memory count
            memory_info = f"  Memories: {len(npc.memory_bank)} | Recent saw: {len(npc.memory_bank.get_memories_by_sense('saw'))}"
            memory_text = font.render(memory_info, True, (200, 200, 200))
            self.screen.blit(memory_text, (UI_MARGIN, y_offset))
            y_offset += UI_LARGE_SPACING
        
        # Draw controls
        controls_y = SCREEN_HEIGHT - UI_LARGE_SPACING
        controls_text = f"Controls: WASD to move, V to toggle NPC vision, F11 to toggle fullscreen, ESC to quit"
        controls_surface = font.render(controls_text, True, (150, 150, 150))
        self.screen.blit(controls_surface, (UI_MARGIN, controls_y))
        
        # (Chat UI removed)
        
        pygame.display.flip()
    
    def run(self):
        # First, run the loading screen
        self._run_loading_screen()
        
        # Then run the main game loop
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            
            running = self.handle_events()
            self.update(dt)
            self.render()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run() 