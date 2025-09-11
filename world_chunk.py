import pygame
import random
from constants import *
from world_gen.terrain_generator import TerrainGenerator

class Chunk:
    def __init__(self, chunk_x, chunk_y, terrain_gen: TerrainGenerator | None = None):
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.world_x = chunk_x * CHUNK_SIZE * TILE_SIZE
        self.world_y = chunk_y * CHUNK_SIZE * TILE_SIZE
        self.terrain_gen = terrain_gen or TerrainGenerator()
        self._tile_images = None  # Lazy-loaded tile images
        
        # Generate terrain for this chunk
        self.generate_terrain()
        
        # Pre-render the chunk to a surface for better performance
        self.surface = pygame.Surface((CHUNK_SIZE * TILE_SIZE, CHUNK_SIZE * TILE_SIZE))
        self.render_to_surface()
    
    def generate_terrain(self):
        """Generate terrain data for this chunk"""
        # Use vectorized generator for oceans and rivers
        self.tiles = self.terrain_gen.generate_water_and_land(self.chunk_x, self.chunk_y)

    def _load_tiles_if_needed(self):
        """Load and scale tile images once (grass, water)."""
        if self._tile_images is not None:
            return
        self._tile_images = {}
        try:
            grass_img = pygame.image.load('assets/tiles/tilesets/grass.png').convert()
            water_img = pygame.image.load('assets/tiles/tilesets/water.png').convert()
            grass_img = pygame.transform.scale(grass_img, (TILE_SIZE, TILE_SIZE))
            water_img = pygame.transform.scale(water_img, (TILE_SIZE, TILE_SIZE))
            self._tile_images['grass'] = grass_img
            self._tile_images['water'] = water_img
        except Exception:
            # Fallback to colored fills if images are missing
            self._tile_images['grass'] = None
            self._tile_images['water'] = None
    
    def render_to_surface(self):
        """Pre-render the chunk to a surface"""
        self._load_tiles_if_needed()
        # Draw tiles
        for ty, row in enumerate(self.tiles):
            for tx, tile in enumerate(row):
                dest = (tx * TILE_SIZE, ty * TILE_SIZE)
                img = self._tile_images.get(tile)
                if img is not None:
                    self.surface.blit(img, dest)
                else:
                    # Fallback solid fill
                    color = WATER_COLOR if tile == 'water' else GRASS_COLOR
                    rect = pygame.Rect(dest[0], dest[1], TILE_SIZE, TILE_SIZE)
                    pygame.draw.rect(self.surface, color, rect)
        
        # Create temporary surface for transparent grid lines
        grid_surface = pygame.Surface((CHUNK_SIZE * TILE_SIZE, CHUNK_SIZE * TILE_SIZE), pygame.SRCALPHA)
        
        # Draw internal tile grid lines (lighter grey)
        for x in range(1, CHUNK_SIZE):  # Skip borders (0 and CHUNK_SIZE)
            start_pos = (x * TILE_SIZE, 0)
            end_pos = (x * TILE_SIZE, CHUNK_SIZE * TILE_SIZE)
            pygame.draw.line(grid_surface, GRID_COLOR, start_pos, end_pos, 1)
        
        for y in range(1, CHUNK_SIZE):  # Skip borders (0 and CHUNK_SIZE)
            start_pos = (0, y * TILE_SIZE)
            end_pos = (CHUNK_SIZE * TILE_SIZE, y * TILE_SIZE)
            pygame.draw.line(grid_surface, GRID_COLOR, start_pos, end_pos, 1)
        
        # Draw chunk borders (darker grey)
        # Top border
        pygame.draw.line(grid_surface, CHUNK_BORDER_COLOR, 
                        (0, 0), (CHUNK_SIZE * TILE_SIZE, 0), 1)
        # Bottom border
        pygame.draw.line(grid_surface, CHUNK_BORDER_COLOR, 
                        (0, CHUNK_SIZE * TILE_SIZE - 1), 
                        (CHUNK_SIZE * TILE_SIZE, CHUNK_SIZE * TILE_SIZE - 1), 1)
        # Left border
        pygame.draw.line(grid_surface, CHUNK_BORDER_COLOR, 
                        (0, 0), (0, CHUNK_SIZE * TILE_SIZE), 1)
        # Right border
        pygame.draw.line(grid_surface, CHUNK_BORDER_COLOR, 
                        (CHUNK_SIZE * TILE_SIZE - 1, 0), 
                        (CHUNK_SIZE * TILE_SIZE - 1, CHUNK_SIZE * TILE_SIZE), 1)
        
        # Blit the grid surface onto the chunk surface
        self.surface.blit(grid_surface, (0, 0))
    
    def render(self, screen, camera_x, camera_y):
        """Render the chunk to the screen"""
        # Calculate screen position
        screen_x = self.world_x - camera_x
        screen_y = self.world_y - camera_y
        
        # Only render if chunk is visible on screen
        chunk_pixel_size = CHUNK_SIZE * TILE_SIZE
        if (screen_x < SCREEN_WIDTH and screen_x + chunk_pixel_size > 0 and
            screen_y < SCREEN_HEIGHT and screen_y + chunk_pixel_size > 0):
            
            screen.blit(self.surface, (screen_x, screen_y)) 