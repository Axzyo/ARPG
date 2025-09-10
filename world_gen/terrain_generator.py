class TerrainGenerator:
    """Generate world terrain heightmaps and biome assignments.

    This is a scaffold. Implement your noise functions and rules here.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def generate_heightmap(self, chunk_x: int, chunk_y: int) -> list[list[float]]:
        """Return a 2D heightmap for the given chunk coordinates.

        Replace with actual implementation.
        """
        return []

    def assign_biomes(self, heightmap: list[list[float]]) -> list[list[str]]:
        """Return a 2D biome map based on the provided heightmap.

        Replace with actual implementation.
        """
        return []
