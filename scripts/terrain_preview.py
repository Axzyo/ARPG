import os
import sys
import math
import pygame

# Ensure project root is on sys.path so imports work when launched via shortcut
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chunk_manager import ChunkManager
from constants import CHUNK_SIZE, WATER_COLOR, GRASS_COLOR, TILE_SIZE


# Configuration linked variables (automatic defaults)
# - tileSize: pixels per tile in the preview image (not game TILE_SIZE)
# - previewWidth/previewHeight: target output size in pixels
tileSize = 2
previewWidth = 4096
previewHeight = 2304


def generate_preview(
    seed: int | None = None,
    output_path: str = "assets/terrain_preview.png",
    center_chunk_x: int = 0,
    center_chunk_y: int = 0,
    tile_pixels: int | None = None,
    out_width: int | None = None,
    out_height: int | None = None,
) -> str:
    # Resolve output path relative to project root if not absolute
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pygame.init()

    # Resolve effective settings from linked variables if not provided
    tile_pixels = tile_pixels or tileSize
    out_width = out_width or previewWidth
    out_height = out_height or previewHeight

    # Compute exact tile rectangle to cover requested output size
    tiles_w = math.ceil(out_width / tile_pixels)
    tiles_h = math.ceil(out_height / tile_pixels)

    # Anchor preview so center_chunk is centered
    start_tile_x = (center_chunk_x * CHUNK_SIZE) - tiles_w // 2
    start_tile_y = (center_chunk_y * CHUNK_SIZE) - tiles_h // 2

    # Working canvas exactly the requested output size
    work_w = tiles_w * tile_pixels
    work_h = tiles_h * tile_pixels
    surface = pygame.Surface((work_w, work_h))

    # Use the same pipeline as the game: shared ChunkManager with a seed
    cm = ChunkManager(seed=seed)

    # Generate tiles for the continuous area using the same generator
    tiles = cm.terrain_gen.generate_area_tiles(start_tile_x, start_tile_y, tiles_w, tiles_h)

    # Draw
    for ty in range(tiles_h):
        row = tiles[ty]
        for tx in range(tiles_w):
            tile = row[tx]
            color = WATER_COLOR if tile == "water" else GRASS_COLOR
            rx = tx * tile_pixels
            ry = ty * tile_pixels
            pygame.draw.rect(surface, color, pygame.Rect(rx, ry, tile_pixels, tile_pixels))

    # Save cropped to requested dimensions
    final = pygame.Surface((out_width, out_height))
    final.blit(surface, (0, 0))
    pygame.image.save(final, output_path)
    return os.path.abspath(output_path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate and open terrain preview image.")
    parser.add_argument("seed", nargs="?", type=int, default=None, help="World seed (int). Random if omitted.")
    parser.add_argument("--tile", dest="tile_pixels", type=int, default=None, help="Pixels per tile in preview (defaults to tileSize var).")
    parser.add_argument("--width", dest="out_width", type=int, default=None, help="Output width in pixels (defaults to previewWidth var).")
    parser.add_argument("--height", dest="out_height", type=int, default=None, help="Output height in pixels (defaults to previewHeight var).")
    parser.add_argument("--no-open", dest="open_after", action="store_false", help="Do not open the image after saving.")
    args = parser.parse_args()

    path = generate_preview(
        seed=args.seed,
        tile_pixels=args.tile_pixels,
        out_width=args.out_width,
        out_height=args.out_height,
    )
    print(f"Saved preview: {path}")

    if args.open_after:
        try:
            os.startfile(path)
        except Exception as e:
            print(f"[WARN] Could not open image automatically: {e}")


if __name__ == "__main__":
    main()


