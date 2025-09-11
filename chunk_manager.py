import pygame
import math
import random
from constants import *
from world_chunk import Chunk
from world_gen.terrain_generator import TerrainGenerator

class ChunkManager:
    def __init__(self, seed: int | None = None):
        self.chunks = {}  # Dictionary to store loaded chunks {(chunk_x, chunk_y): Chunk}
        self.last_player_chunk = None
        # Generate a random world seed once per session
        self.world_seed = int(seed) if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        print(f"[WORLD] Seed: {self.world_seed}")
        # Single terrain generator shared across all chunks for determinism
        self.terrain_gen = TerrainGenerator(seed=self.world_seed)
        
    def get_chunk_coords(self, world_x, world_y):
        """Convert world coordinates to chunk coordinates"""
        chunk_x = math.floor(world_x / (CHUNK_SIZE * TILE_SIZE))
        chunk_y = math.floor(world_y / (CHUNK_SIZE * TILE_SIZE))
        return chunk_x, chunk_y
    
    def get_chunk(self, chunk_x, chunk_y):
        """Get or create a chunk at the given chunk coordinates"""
        chunk_key = (chunk_x, chunk_y)
        
        if chunk_key not in self.chunks:
            self.chunks[chunk_key] = Chunk(chunk_x, chunk_y, terrain_gen=self.terrain_gen)
            
        return self.chunks[chunk_key]
    
    def update_chunks(self, player_x, player_y):
        """Update loaded chunks based on player position"""
        player_chunk_x, player_chunk_y = self.get_chunk_coords(player_x, player_y)
        current_player_chunk = (player_chunk_x, player_chunk_y)
        
        # Only update if player moved to a different chunk
        if self.last_player_chunk != current_player_chunk:
            self.last_player_chunk = current_player_chunk
            
            # Calculate which chunks should be loaded
            chunks_to_load = set()
            for dx in range(-RENDER_DISTANCE, RENDER_DISTANCE + 1):
                for dy in range(-RENDER_DISTANCE, RENDER_DISTANCE + 1):
                    chunk_x = player_chunk_x + dx
                    chunk_y = player_chunk_y + dy
                    chunks_to_load.add((chunk_x, chunk_y))
            
            # Load new chunks
            for chunk_coords in chunks_to_load:
                if chunk_coords not in self.chunks:
                    self.get_chunk(chunk_coords[0], chunk_coords[1])
            
            # Unload distant chunks
            chunks_to_unload = []
            for chunk_coords in self.chunks:
                chunk_x, chunk_y = chunk_coords
                distance = max(abs(chunk_x - player_chunk_x), abs(chunk_y - player_chunk_y))
                if distance > RENDER_DISTANCE + 1:  # Keep one extra chunk buffer
                    chunks_to_unload.append(chunk_coords)
            
            for chunk_coords in chunks_to_unload:
                del self.chunks[chunk_coords]
    
    def render(self, screen, camera_x, camera_y):
        """Render all loaded chunks"""
        for chunk in self.chunks.values():
            chunk.render(screen, camera_x, camera_y) 