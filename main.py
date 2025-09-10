import pygame
import sys
import threading
import time
from player import Player
from chunk_manager import ChunkManager
from npc import NPC
from chat_interface import ChatInterface
from llm_client import LLMClient
from interaction_popup import InteractionPopup
from loading_screen import LoadingScreen
from npc_prompts import get_conversation_prompt
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
        self.chat_interface = None
        self.llm_client = None
        self.interaction_popup = None
        
        # Game state
        self.camera_x = 0
        self.camera_y = 0
        self.show_npc_vision = True
        self.nearby_npc = None
        
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
                
                # Initialize other systems
                self.chat_interface = ChatInterface()
                self.interaction_popup = InteractionPopup()
                self.loading_tasks['npcs'] = True
                self._update_loading_progress("NPCs and systems ready", 4)
                
                # Task 5: Initialize and preload LLM (this is the slow part)
                self.llm_client = LLMClient()
                
                def on_llm_preload_complete(success):
                    if success:
                        self._update_loading_progress("AI models loaded", 5)
                    
                    self.loading_tasks['llm'] = True
                    if success:
                        self._update_loading_progress("AI models loaded", 6)
                    else:
                        self._update_loading_progress("AI models ready (fallback mode)", 6)
                    self.loading_complete = True
                
                # Start LLM preload asynchronously
                self.llm_client.preload_model_async(on_llm_preload_complete)
                
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
        self.loading_screen.set_task_progress(5, "Ready!")
        self.loading_screen.render()
        time.sleep(0.5)  # Brief pause to show completion
        
    def handle_events(self):
        # If still loading, don't process game events
        if not self.loading_complete:
            return True
            
        for event in pygame.event.get():
            chat_just_opened = False  # Track if we just opened chat with this event
            
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_v and not self.chat_interface.active:
                    # Toggle NPC vision display (disabled while chat is active)
                    self.show_npc_vision = not self.show_npc_vision
                    print(f"NPC Vision Display: {'ON' if self.show_npc_vision else 'OFF'}")
                elif event.key == pygame.K_e:
                    # Interact with nearby NPC using E key
                    if not self.chat_interface.active and self.nearby_npc:
                        self._handle_npc_interaction(self.nearby_npc)
                        chat_just_opened = True  # Mark that we just opened chat
                elif event.key == pygame.K_F11:
                    # Toggle fullscreen
                    self._toggle_fullscreen()
            
            # Handle chat interface events (only if we didn't just open chat with this event)
            if self.chat_interface and not chat_just_opened:
                chat_result = self.chat_interface.handle_event(event)
                if chat_result:
                    message = chat_result["message"]
                    target_npc = chat_result["target_npc"]
                    print(f"[GAME] Chat interface returned message: '{message}' for NPC: {target_npc.name if target_npc else 'None'}")
                    self._send_message_to_llm(message, target_npc)
                
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
    
    def _handle_npc_interaction(self, npc):
        """Handle keyboard interaction with NPC"""
        if npc:
            distance = npc.get_distance_to(self.player.x, self.player.y)
            print(f"[CHAT] Opening chat with {npc.name} (distance: {distance:.1f})")
            self.chat_interface.open_chat(npc)
        else:
            print(f"[CHAT] No NPC to interact with")
    
    def _update_nearby_npc_detection(self):
        """Check for NPCs in interaction range and manage popups"""
        previous_nearby_npc = self.nearby_npc
        self.nearby_npc = None
        
        # Find the closest NPC within interaction distance
        closest_distance = float('inf')
        interaction_distance = NPC_INTERACTION_DISTANCE  # Same as used in NPC.is_player_nearby
        
        for npc in self.npcs:
            if npc.is_player_nearby(self.player.x, self.player.y, interaction_distance):
                distance = npc.get_distance_to(self.player.x, self.player.y)
                if distance < closest_distance:
                    closest_distance = distance
                    self.nearby_npc = npc
        
        # Manage interaction popups
        if self.nearby_npc and not self.chat_interface.active and not (self.nearby_npc.chat_response or self.nearby_npc.is_thinking):
            # Show interaction popup for nearby NPC
            popup_y = self.nearby_npc.y - (UI_LARGE_SPACING + UI_MARGIN)  # Show popup above NPC
            self.interaction_popup.add_popup(
                f"Press E to talk to {self.nearby_npc.name}",
                self.nearby_npc.x,
                popup_y,
                "interact"
            )
        else:
            # Clear popups when no NPC is nearby
            self.interaction_popup.clear_popups()
    
    def _send_message_to_llm(self, message, target_npc):
        """Send message to LLM and display response"""
        print(f"[DEBUG] _send_message_to_llm called with message: '{message}'")
        print(f"[DEBUG] Target NPC: {target_npc}")
        
        if target_npc:
            print(f"[CHAT] Player message to {target_npc.name}: '{message}'")
            
            # Store the player's message as a memory
            target_npc.add_memory("heard", "player", f"Player said: '{message}'", significance=7.0)
            
            # Check if backstory is available - if not, use preset responses immediately
            # Always attempt AI response; fall back to presets only on error
            
            try:
                # Start thinking animation for real LLM requests (covers actual processing time)
                target_npc.start_thinking()
                
                # Construct the structured prompt for LLM processing
                structured_prompt = self._build_npc_prompt(message, target_npc)
                
                # Define callback functions for streaming
                def on_chunk_received(chunk_text):
                    """Called when a chunk of response arrives"""
                    print(f"[STREAM] Chunk received: '{chunk_text}'")
                    target_npc.add_to_response(chunk_text)
                
                def on_response_complete(full_response):
                    """Called when the full response is complete"""
                    print(f"[STREAM] Response complete: '{full_response}'")
                    target_npc.complete_response(full_response)
                
                # Send async streaming request
                print(f"[DEBUG] Starting async streaming request...")
                self.llm_client.send_message_async(
                    structured_prompt, 
                    on_chunk_received, 
                    on_response_complete
                )
                print(f"[DEBUG] Async request started, game continues running...")
                
            except Exception as e:
                print(f"[ERROR] Exception in _send_message_to_llm: {e}")
                import traceback
                traceback.print_exc()
                # Start timed preset responses instead of error message
                target_npc.is_thinking = False
                target_npc._start_timed_preset_responses()
        else:
            print(f"[ERROR] No target NPC found!")
    
    def _build_npc_prompt(self, player_message, target_npc):
        """Build a simplified prompt for faster NPC interaction
        
        Note: The actual prompt template is defined in npc_prompts.py - edit that file to change how NPCs respond"""
        
        # Get only the most recent memories for speed (avoid expensive filtering)
        recent_memories = target_npc.memory_bank.get_recent_memories(60.0)  # Last 60 seconds only
        memory_text = "No recent memories"
        if recent_memories:
            # Only use the 2 most recent memories for speed
            memory_entries = [f"- {memory.data}" for memory in recent_memories[-2:]]
            memory_text = "\n".join(memory_entries)
        
        # Build memory-driven context only (no backstory summary)
        prompt = get_conversation_prompt(
            npc_name=target_npc.name,
            backstory_summary="",  # deprecated: backstory now lives in memories
            memory_text=memory_text,
            player_message=player_message
        )
        
        # Debug output to see what's being sent to LLM
        print(f"[DEBUG] Prompt being sent to LLM:")
        print(f"--- PROMPT START ---")
        print(prompt)
        print(f"--- PROMPT END ---")
        
        return prompt
    
    def update(self, dt):
        # Don't update game state while loading
        if not self.loading_complete:
            return
            
        # Handle player input and movement (only if chat is not open)
        if not self.chat_interface.active:
            keys = pygame.key.get_pressed()
            self.player.update(keys, dt)
        
        # Update camera to follow player
        self.camera_x = self.player.x - SCREEN_WIDTH // 2
        self.camera_y = self.player.y - SCREEN_HEIGHT // 2
        
        # Update chunk manager based on player position
        self.chunk_manager.update_chunks(self.player.x, self.player.y)
        
        # Check for nearby NPCs for interaction
        self._update_nearby_npc_detection()
        
        # Update interaction popup system
        self.interaction_popup.update(dt)
        
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
        controls_text = f"Controls: WASD to move, V to toggle NPC vision, E to interact with NPCs, F11 to toggle fullscreen, ESC to quit"
        controls_surface = font.render(controls_text, True, (150, 150, 150))
        self.screen.blit(controls_surface, (UI_MARGIN, controls_y))
        
        # Render interaction popups
        self.interaction_popup.render(self.screen, self.camera_x, self.camera_y)
        
        # Render chat interface
        self.chat_interface.render(self.screen)
        
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