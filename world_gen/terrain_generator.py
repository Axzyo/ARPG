import numpy as np
from dataclasses import dataclass


@dataclass
class MacroParams:
    seed: int = 12345
    macro_freq: float = 1.0 / 1800.0
    warp_freq: float = 1.0 / 3500.0
    warp_amp: float = 90.0
    shape_exp: float = 2.0
    sea_level: float = 0.45
    coast_detail_freq: float = 1.0 / 300.0
    coast_detail_amp: float = 0.05
    close_iters: int = 1
    coast_band_width: float = 0.08


class _ValueNoise:
    def __init__(self, seed: int):
        self.seed = int(seed)

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _hash2(self, x: np.ndarray, y: np.ndarray, salt: int) -> np.ndarray:
        xi = x.astype(np.int64)
        yi = y.astype(np.int64)
        n = (xi * 374761393) ^ (yi * 668265263) ^ (self.seed * 144664) ^ (salt * 104395301)
        n = (n ^ (n >> 13)) * 1274126177
        n = (n ^ (n >> 16)) & ((1 << 32) - 1)
        return n.astype(np.float64) / 4294967295.0

    def fbm(self, gx: np.ndarray, gy: np.ndarray, freq: float, octaves: int = 5,
            lacunarity: float = 2.0, gain: float = 0.5, salt: int = 0) -> np.ndarray:
        value = np.zeros_like(gx, dtype=np.float64)
        amplitude = 1.0
        sum_amp = 0.0
        current_freq = freq
        current_salt = salt
        for _ in range(octaves):
            fx = gx * current_freq
            fy = gy * current_freq
            x0 = np.floor(fx)
            y0 = np.floor(fy)
            x1 = x0 + 1
            y1 = y0 + 1
            sx = self._fade(fx - x0)
            sy = self._fade(fy - y0)
            v00 = self._hash2(x0, y0, current_salt)
            v10 = self._hash2(x1, y0, current_salt)
            v01 = self._hash2(x0, y1, current_salt)
            v11 = self._hash2(x1, y1, current_salt)
            ix0 = v00 * (1 - sx) + v10 * sx
            ix1 = v01 * (1 - sx) + v11 * sx
            value += amplitude * (ix0 * (1 - sy) + ix1 * sy)
            sum_amp += amplitude
            amplitude *= gain
            current_freq *= lacunarity
            current_salt += 1337
        if sum_amp <= 1e-12:
            return value
        return value / sum_amp


def _shift_bool(a: np.ndarray, dy: int, dx: int, fill: bool) -> np.ndarray:
    H, W = a.shape
    out = np.full((H, W), fill, dtype=bool)
    ys = slice(max(0, dy), min(H, H + dy))
    xs = slice(max(0, dx), min(W, W + dx))
    ys_src = slice(max(0, -dy), min(H, H - dy))
    xs_src = slice(max(0, -dx), min(W, W - dx))
    out[ys, xs] = a[ys_src, xs_src]
    return out


def _dilate_cross(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    return (
        m
        | _shift_bool(m, 1, 0, False)
        | _shift_bool(m, -1, 0, False)
        | _shift_bool(m, 0, 1, False)
        | _shift_bool(m, 0, -1, False)
    )


def _erode_cross(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    return (
        m
        & _shift_bool(m, 1, 0, True)
        & _shift_bool(m, -1, 0, True)
        & _shift_bool(m, 0, 1, True)
        & _shift_bool(m, 0, -1, True)
    )


def band_limited_close(land: np.ndarray, band: np.ndarray, iters: int = 1) -> np.ndarray:
    if iters <= 0:
        return land
    closed = land.copy()
    tmp = land
    for _ in range(iters):
        tmp = _erode_cross(_dilate_cross(tmp))
    closed[band] = tmp[band]
    return closed


def geodesic_dilate(marker: np.ndarray, mask: np.ndarray, iters: int) -> np.ndarray:
    """Iterative cross-dilation constrained to a mask (geodesic reconstruction)."""
    cur = marker.copy()
    for _ in range(max(0, iters)):
        cur = _dilate_cross(cur) & mask
    return cur


def manhattan_dist_to_water(land: np.ndarray, rounds: int = 64) -> np.ndarray:
    """Approximate 4-neighbor brushfire distance to water cells (zeros on water)."""
    INF = 10 ** 6
    d = np.where(land, INF, 0).astype(np.int32)
    H, W = d.shape
    for _ in range(max(1, rounds)):
        # forward
        d[1:, :] = np.minimum(d[1:, :], d[:-1, :] + 1)
        d[:, 1:] = np.minimum(d[:, 1:], d[:, :-1] + 1)
        # backward
        d[:-1, :] = np.minimum(d[:-1, :], d[1:, :] + 1)
        d[:, :-1] = np.minimum(d[:, :-1], d[:, 1:] + 1)
    return d


def prune_tendrils(land: np.ndarray, band: np.ndarray, min_width: int = 3) -> np.ndarray:
    """Remove 1–2 tile land filaments in the coastal band while preserving interiors."""
    dist = manhattan_dist_to_water(land)
    core = land & (dist >= int(min_width))
    recon = geodesic_dilate(core, land, iters=max(0, int(min_width) - 1))
    out = land.copy()
    out[band] = recon[band]
    return out


class MacroContinents:
    def __init__(self, p: MacroParams):
        self.p = p
        self.noise = _ValueNoise(p.seed)

    def _domain_warp(self, gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        wx = self.noise.fbm(gx, gy, p.warp_freq, octaves=4, salt=901)
        wy = self.noise.fbm(gx + 137.0, gy - 89.0, p.warp_freq, octaves=4, salt=907)
        wx = (wx * 2.0 - 1.0) * p.warp_amp
        wy = (wy * 2.0 - 1.0) * p.warp_amp
        return gx + wx, gy + wy

    def continent_field(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        p = self.p
        xw, yw = self._domain_warp(gx, gy)
        base = self.noise.fbm(xw, yw, p.macro_freq, octaves=5, salt=1100)
        ridg_src = self.noise.fbm(xw * 0.9 + 37.0, yw * 0.9 - 11.0, p.macro_freq, octaves=4, salt=1200)
        ridg = 1.0 - np.abs(ridg_src - 0.5) * 2.0
        f = 0.65 * base + 0.35 * ridg
        f = np.clip(f, 0.0, 1.0) ** p.shape_exp
        return f

    def land_mask(self, gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        p = self.p
        field = self.continent_field(gx, gy)
        # Adaptive sea level (percentile) for robust ocean/land balance
        thr = float(np.percentile(field, 60.0))
        band = np.abs(field - thr) < p.coast_band_width
        detail = self.noise.fbm(gx, gy, p.coast_detail_freq, octaves=3, salt=1300)
        detail = (detail - 0.5) * p.coast_detail_amp
        shaped = field + detail * band
        land = shaped >= thr
        land_closed = band_limited_close(land, band, iters=p.close_iters)
        land_pruned = prune_tendrils(land_closed, band, min_width=3)
        return land_pruned, field, thr
from constants import CHUNK_SIZE


class TerrainGenerator:
    """Generate tile biomes with oceans and rivers (no Python loops).

    Design goals (v0):
    - No varied elevation/heightmap yet
    - Large-scale oceans ("continentalness")
    - Meandering rivers across land only
    - Fully vectorized via NumPy (no Python for/while loops)

    Output tiles: currently "water" and "grass" only.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed: int = int(seed) if seed is not None else 1337

    # -------------------------
    # Utility: deterministic hash → [0, 1]
    # -------------------------
    def _hash2(self, x: np.ndarray, y: np.ndarray, salt: int) -> np.ndarray:
        """Deterministic 2D hash that broadcasts over arrays.

        Produces reproducible pseudo-random values in [0, 1].
        """
        xi = x.astype(np.int64)
        yi = y.astype(np.int64)

        n = (xi * 374761393) ^ (yi * 668265263) ^ (self.seed * 144664) ^ (salt * 104395301)
        n = (n ^ (n >> 13)) * 1274126177
        n = (n ^ (n >> 16)) & ((1 << 32) - 1)
        return n.astype(np.float64) / 4294967295.0

    def _fade(self, t: np.ndarray) -> np.ndarray:
        """Smoothstep-like fade for interpolation (Perlin 6t^5-15t^4+10t^3)."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _value_noise(self, gx: np.ndarray, gy: np.ndarray, step: float, salt: int) -> np.ndarray:
        """Continuous value noise via lattice + bilinear interpolation (loop-free).

        Parameters
        - gx, gy: global tile-space coordinates arrays (same shape)
        - step: lattice spacing in tiles (larger → lower frequency)
        - salt: extra seed salt for independent fields
        """
        fx = gx / step
        fy = gy / step

        x0 = np.floor(fx)
        y0 = np.floor(fy)
        x1 = x0 + 1
        y1 = y0 + 1

        sx = self._fade(fx - x0)
        sy = self._fade(fy - y0)

        v00 = self._hash2(x0, y0, salt)
        v10 = self._hash2(x1, y0, salt)
        v01 = self._hash2(x0, y1, salt)
        v11 = self._hash2(x1, y1, salt)

        ix0 = v00 * (1 - sx) + v10 * sx
        ix1 = v01 * (1 - sx) + v11 * sx
        return ix0 * (1 - sy) + ix1 * sy

    # -------------------------
    # Public API (no loops)
    # -------------------------
    def generate_water_and_land(self, chunk_x: int, chunk_y: int) -> list[list[str]]:
        """Return CHUNK_SIZE x CHUNK_SIZE tiles: "water" or "grass" using area generator."""
        size = CHUNK_SIZE
        x0 = chunk_x * size
        y0 = chunk_y * size
        tiles = self.generate_area_tiles(x0, y0, size, size)
        return tiles.tolist()

    # Backwards-compat scaffolds (not used for this style)
    def generate_heightmap(self, chunk_x: int, chunk_y: int) -> list[list[float]]:
        """Placeholder to preserve interface; returns zeros (flat)."""
        return (np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=float)).tolist()

    def assign_biomes(self, heightmap: list[list[float]]) -> list[list[str]]:
        """Deprecated for this flow; mirrors generate_water_and_land with all grass."""
        size = CHUNK_SIZE
        return (np.full((size, size), "grass", dtype=object)).tolist()

    # --------- Vectorized large-area generation for previews ---------
    # Flow helpers (D8)
    _DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    _DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    _DW = np.array([np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0)])

    @staticmethod
    def _shift_edge(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """Shift with edge replication (used for slope-only ops)."""
        if dy == 0 and dx == 0:
            return a
        return np.pad(a, ((max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0))), mode='edge')[
            max(-dy, 0):a.shape[0] + max(-dy, 0), max(-dx, 0):a.shape[1] + max(-dx, 0)
        ]

    @classmethod
    def _steepest_descent(cls, height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H, W = height.shape
        drops = []
        for k in range(8):
            nb = cls._shift_edge(height, int(cls._DY[k]), int(cls._DX[k]))
            drop = (height - nb) / cls._DW[k]
            drops.append(drop)
        drops = np.stack(drops, axis=0)  # (8,H,W)
        kmax = np.argmax(drops, axis=0)
        maxdrop = drops[kmax, np.arange(H)[:, None], np.arange(W)]
        uphill = maxdrop <= 0
        if uphill.any():
            # choose neighbor with smallest uphill penalty
            kmin = np.argmin(np.abs(drops), axis=0)
            kmax = np.where(uphill, kmin, kmax)
            maxdrop = drops[kmax, np.arange(H)[:, None], np.arange(W)]
        to_y = (np.arange(H)[:, None] + cls._DY[kmax]).clip(0, H - 1)
        to_x = (np.arange(W)[None, :] + cls._DX[kmax]).clip(0, W - 1)
        to_index = (to_y * W + to_x).ravel()
        return to_index, maxdrop.ravel()
    def generate_area_water_mask(self, x0_tile: int, y0_tile: int, width_tiles: int, height_tiles: int) -> np.ndarray:
        """Return boolean mask (height_tiles x width_tiles): True for water, False for land.

        This matches the per-chunk logic but works over an arbitrary rectangle without Python loops.
        """
        # Create a HALO around requested area to eliminate boundary artifacts in morphology
        HALO = max(16, int(MacroParams.coast_band_width * 256))
        y, x = np.mgrid[-HALO:height_tiles + HALO, -HALO:width_tiles + HALO]
        gx = x0_tile + x
        gy = y0_tile + y

        # Domain warp used by continents step
        warp_step = 96.0
        warp_amp = 18.0
        wx = self._value_noise(gx, gy, warp_step, salt=900)
        wy = self._value_noise(gx + 1000, gy + 1000, warp_step, salt=901)
        gxw = gx + (wx - 0.5) * 2.0 * warp_amp
        gyw = gy + (wy - 0.5) * 2.0 * warp_amp

        macro = MacroContinents(MacroParams(seed=self.seed))
        land_mask_halo, field_halo, thr = macro.land_mask(gxw, gyw)
        ocean_mask_halo = ~land_mask_halo

        # --- Build heightfield on land (HALO region) ---
        inland = np.clip((field_halo - thr) / (1 - thr + 1e-6), 0.0, 1.0)
        relief = macro.noise.fbm(gxw * 0.8 + 77.0, gyw * 0.8 - 33.0, freq=1.0 / 280.0, octaves=4, salt=1700)
        relief = 1.0 - np.abs(relief - 0.5) * 2.0
        height = inland + 0.45 * (inland ** 1.2) * (relief - 0.5)
        height += 0.02 * (inland - 1.0)
        height = np.where(ocean_mask_halo, -10.0, height)

        # Crop back to requested area after band-limited operations
        sl = (slice(HALO, HALO + height_tiles), slice(HALO, HALO + width_tiles))
        height = height[sl]
        land_mask = land_mask_halo[sl]
        ocean_mask = ocean_mask_halo[sl]

        # --- Flow directions (D8) & accumulation on cropped area ---
        to_index, _ = self._steepest_descent(height)
        H, W = height.shape
        N = H * W
        order = np.argsort(height.ravel())[::-1]
        acc = np.ones(N, dtype=np.float32)
        np.add.at(acc, to_index[order], acc[order])
        FA = acc.reshape(H, W)

        # local slope magnitude using edge replication
        neighbors = [self._shift_edge(height, int(dy), int(dx)) for dy, dx in zip(self._DY, self._DX)]
        slopes = [np.maximum(height - nb, 0.0) for nb in neighbors]
        slope = np.maximum.reduce(slopes)
        flatness = 1.0 / (1.0 + 30.0 * slope)

        river_power = FA * flatness * land_mask
        if land_mask.any():
            thresh = np.percentile(river_power[land_mask], 97.0)
        else:
            thresh = np.inf
        rivers = river_power >= thresh

        water_mask = ocean_mask | (rivers & (~ocean_mask))
        return water_mask

    def generate_area_tiles(self, x0_tile: int, y0_tile: int, width_tiles: int, height_tiles: int) -> np.ndarray:
        mask = self.generate_area_water_mask(x0_tile, y0_tile, width_tiles, height_tiles)
        return np.where(mask, "water", "grass").astype(object)
