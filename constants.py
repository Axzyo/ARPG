# Initialize pygame to get display info
import pygame
pygame.init()

# Get native screen resolution for fullscreen scaling
display_info = pygame.display.Info()
NATIVE_WIDTH = display_info.current_w
NATIVE_HEIGHT = display_info.current_h

# Reference resolution for scaling calculations (what the game was designed for)
REFERENCE_WIDTH = 1024
REFERENCE_HEIGHT = 768
REFERENCE_TILE_SIZE = 32

# Calculate scaling factors
SCALE_W = NATIVE_WIDTH / REFERENCE_WIDTH
SCALE_H = NATIVE_HEIGHT / REFERENCE_HEIGHT
SCALE = min(SCALE_W, SCALE_H)  # Use minimum to maintain aspect ratio

# World zoom (1.0 = default scale). Lower values zoom out (smaller tiles).
WORLD_ZOOM = 0.70

# Screen settings (fullscreen uses native resolution)
SCREEN_WIDTH = NATIVE_WIDTH
SCREEN_HEIGHT = NATIVE_HEIGHT
FPS = 60

# Tile and chunk settings (dynamically scaled)
TILE_SIZE = max(12, int(REFERENCE_TILE_SIZE * SCALE * WORLD_ZOOM))  # Minimum 12px tiles
CHUNK_SIZE = 16  # 16x16 tiles per chunk
RENDER_DISTANCE = 2  # Number of chunks to render around player

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRASS_COLOR = (104, 178, 78)  # #68b24e
WATER_COLOR = (64, 105, 225)  # Deep blue for water
GRID_COLOR = (200, 200, 200, 128)  # Light grey with 50% transparency
CHUNK_BORDER_COLOR = (120, 120, 120, 128)  # Darker grey with 50% transparency
PLAYER_COLOR = (255, 100, 100)  # Red player

# Player settings (scaled based on display)
PLAYER_SPEED = int(200 * SCALE)  # pixels per second, scaled
PLAYER_SIZE = max(12, int(24 * SCALE))  # Minimum 12px player size

# UI Font sizes (scaled)
FONT_SMALL = max(12, int(16 * SCALE))
FONT_NORMAL = max(16, int(20 * SCALE))
FONT_MEDIUM = max(18, int(24 * SCALE))
FONT_LARGE = max(24, int(32 * SCALE))
FONT_TITLE = max(32, int(48 * SCALE))

# NPC settings (scaled)
NPC_SIZE = max(12, int(20 * SCALE))  # NPC visual size
NPC_VISION_RANGE = max(8, int(16 * SCALE))  # Vision range in scaled tiles
NPC_INTERACTION_DISTANCE = max(25, int(50 * SCALE))  # Interaction range
NPC_MOVEMENT_PRECISION = max(2, int(5 * SCALE))  # How close to get to movement targets

# UI spacing and dimensions (scaled)
UI_MARGIN = max(5, int(10 * SCALE))  # Standard UI margin
UI_PADDING = max(4, int(8 * SCALE))  # Standard UI padding
UI_SMALL_SPACING = max(2, int(5 * SCALE))  # Small spacing
UI_LARGE_SPACING = max(12, int(25 * SCALE))  # Large spacing

# Debug scaling info (printed when constants are loaded)
print(f"[DISPLAY] Native Resolution: {NATIVE_WIDTH}x{NATIVE_HEIGHT}")
print(f"[DISPLAY] Reference Resolution: {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}")
print(f"[DISPLAY] Scale Factors: W={SCALE_W:.2f}, H={SCALE_H:.2f}")
print(f"[DISPLAY] Applied Scale: {SCALE:.2f}")
print(f"[DISPLAY] Tile Size: {REFERENCE_TILE_SIZE} -> {TILE_SIZE}")
print(f"[DISPLAY] Player Size: 24 -> {PLAYER_SIZE}")
print(f"[DISPLAY] Player Speed: 200 -> {PLAYER_SPEED}")
print(f"[DISPLAY] Font Sizes: Small={FONT_SMALL}, Normal={FONT_NORMAL}, Medium={FONT_MEDIUM}, Large={FONT_LARGE}, Title={FONT_TITLE}")
print(f"[DISPLAY] NPC Settings: Size=20->{NPC_SIZE}, Vision=16->{NPC_VISION_RANGE}, Interaction=50->{NPC_INTERACTION_DISTANCE}, MovePrecision=5->{NPC_MOVEMENT_PRECISION}")
print(f"[DISPLAY] UI Spacing: Margin={UI_MARGIN}, Padding={UI_PADDING}, Small={UI_SMALL_SPACING}, Large={UI_LARGE_SPACING}") 