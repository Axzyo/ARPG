import numpy as np
from dataclasses import dataclass
from collections import deque


@dataclass
class MacroParams:
    seed: int = 12345
    macro_freq: float = 1.0 / 1800.0
    warp_freq: float = 1.0 / 3500.0
    warp_amp: float = 80.0
    shape_exp: float = 2.0
    sea_level: float = 0.45
    coast_detail_freq: float = 1.0 / 300.0
    coast_detail_amp: float = 0.04
    close_iters: int = 1
    coast_band_width: float = 0.08
    coast_band_tiles: int = 8


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


def _shift(a: np.ndarray, dy: int, dx: int, fill_value):
    H, W = a.shape
    out = np.full((H, W), fill_value, dtype=a.dtype)
    ys = slice(max(0, dy), min(H, H + dy))
    xs = slice(max(0, dx), min(W, W + dx))
    out[ys, xs] = a[max(0, -dy):min(H, H - dy), max(0, -dx):min(W, W - dx)]
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


def _dilate_8(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    fills = [_shift(m, dy, dx, False) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))]
    return m | np.logical_or.reduce(fills)


def _erode_8(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    fills = [_shift(m, dy, dx, True) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))]
    return m & np.logical_and.reduce(fills)


def band_limited_close(land: np.ndarray, band: np.ndarray, iters: int = 1) -> np.ndarray:
    if iters <= 0:
        return land
    closed = land.copy()
    tmp = land
    for _ in range(iters):
        tmp = _erode_cross(_dilate_cross(tmp))
    closed[band] = tmp[band]
    return closed


def band_limited_close_8(land: np.ndarray, band: np.ndarray, iters: int = 1) -> np.ndarray:
    if iters <= 0:
        return land
    closed = land.copy()
    tmp = land
    for _ in range(iters):
        tmp = _erode_8(_dilate_8(tmp))
    closed[band] = tmp[band]
    return closed


def geodesic_dilate(marker: np.ndarray, mask: np.ndarray, iters: int) -> np.ndarray:
    cur = marker.copy()
    for _ in range(max(0, iters)):
        cur = _dilate_cross(cur) & mask
    return cur


def manhattan_dist_to_water(land: np.ndarray, rounds: int = 64) -> np.ndarray:
    INF = 10 ** 6
    d = np.where(land, INF, 0).astype(np.int32)
    H, W = d.shape
    for _ in range(max(1, rounds)):
        d[1:, :] = np.minimum(d[1:, :], d[:-1, :] + 1)
        d[:, 1:] = np.minimum(d[:, 1:], d[:, :-1] + 1)
        d[:-1, :] = np.minimum(d[:-1, :], d[1:, :] + 1)
        d[:, :-1] = np.minimum(d[:, :-1], d[:, 1:] + 1)
    return d


def euclidish_dist(a: np.ndarray, rounds: int = 6) -> np.ndarray:
    INF = np.float32(1e6)
    d = np.where(a, INF, 0.0).astype(np.float32)
    rt2 = np.float32(1.41421356)
    for _ in range(rounds):
        d = np.minimum(d, _shift(d, 1, 0, INF) + 1)
        d = np.minimum(d, _shift(d, -1, 0, INF) + 1)
        d = np.minimum(d, _shift(d, 0, 1, INF) + 1)
        d = np.minimum(d, _shift(d, 0, -1, INF) + 1)
        d = np.minimum(d, _shift(d, 1, 1, INF) + rt2)
        d = np.minimum(d, _shift(d, 1, -1, INF) + rt2)
        d = np.minimum(d, _shift(d, -1, 1, INF) + rt2)
        d = np.minimum(d, _shift(d, -1, -1, INF) + rt2)
    return d


def prune_tendrils(land: np.ndarray, band: np.ndarray, min_width: int = 3) -> np.ndarray:
    """Remove land filaments thinner than min_width (in tiles) only in the band,
    using 8-neigh distance and reconstruction (preserves true isthmuses)."""
    dist = euclidish_dist(land)
    core = land & (dist >= float(min_width))
    recon = core.copy()
    for _ in range(max(0, int(min_width) - 1)):
        recon = _dilate_8(recon) & land
    out = land.copy()
    out[band] = recon[band]
    return out


def majority_smooth(mask: np.ndarray, band: np.ndarray, rounds: int = 1) -> np.ndarray:
    """3x3 majority filter (8-neigh) applied only inside 'band'."""
    m = mask.astype(np.uint8)
    for _ in range(max(1, rounds)):
        s = (
            m
            + _shift(m, 1, 0, 0) + _shift(m, -1, 0, 0)
            + _shift(m, 0, 1, 0) + _shift(m, 0, -1, 0)
            + _shift(m, 1, 1, 0) + _shift(m, 1, -1, 0)
            + _shift(m, -1, 1, 0) + _shift(m, -1, -1, 0)
        )
        new = (s >= 5).astype(bool)
        mask[band] = new[band]
        m = mask.astype(np.uint8)
    return mask


def prune_tendrils_cross(land: np.ndarray, band: np.ndarray, min_width: int = 3) -> np.ndarray:
    # kept for potential reference; not used
    dist = manhattan_dist_to_water(land)
    core = land & (dist >= int(min_width))
    recon = geodesic_dilate(core, land, iters=max(0, int(min_width) - 1))
    out = land.copy()
    out[band] = recon[band]
    return out


def fill_small_lakes_on_halo(land_halo: np.ndarray, max_area: int = 150) -> np.ndarray:
    H, W = land_halo.shape
    water = ~land_halo

    # Ocean flood-fill from border water (8-neigh)
    ocean = np.zeros_like(water, dtype=bool)
    q = deque()
    for x in range(W):
        if water[0, x]:
            ocean[0, x] = True
            q.append((0, x))
        if water[H - 1, x]:
            ocean[H - 1, x] = True
            q.append((H - 1, x))
    for y in range(H):
        if water[y, 0]:
            ocean[y, 0] = True
            q.append((y, 0))
        if water[y, W - 1]:
            ocean[y, W - 1] = True
            q.append((y, W - 1))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and water[ny, nx] and not ocean[ny, nx]:
                ocean[ny, nx] = True
                q.append((ny, nx))

    # Fill small lakes (8-neigh components not connected to border) under threshold
    lakes = water & (~ocean)
    seen = np.zeros_like(lakes, dtype=bool)
    for sy in range(H):
        for sx in range(W):
            if lakes[sy, sx] and not seen[sy, sx]:
                comp_cells = []
                comp_size = 0
                dq = deque([(sy, sx)])
                seen[sy, sx] = True
                small = True
                while dq:
                    cy, cx = dq.popleft()
                    comp_cells.append((cy, cx))
                    comp_size += 1
                    if comp_size > max_area:
                        small = False
                        dq.clear()
                        break
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and lakes[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            dq.append((ny, nx))
                if small:
                    for cy, cx in comp_cells:
                        land_halo[cy, cx] = True
    return land_halo


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
        f = np.clip(f, 0.0, 1.0) ** self.p.shape_exp
        return f

from constants import CHUNK_SIZE


class TerrainGenerator:
    """Generate tile biomes with oceans and rivers (no Python loops)."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed: int = int(seed) if seed is not None else 1337
        self._sea_level: float | None = None

    def _hash2(self, x: np.ndarray, y: np.ndarray, salt: int) -> np.ndarray:
        xi = x.astype(np.int64)
        yi = y.astype(np.int64)
        n = (xi * 374761393) ^ (yi * 668265263) ^ (self.seed * 144664) ^ (salt * 104395301)
        n = (n ^ (n >> 13)) * 1274126177
        n = (n ^ (n >> 16)) & ((1 << 32) - 1)
        return n.astype(np.float64) / 4294967295.0

    def _fade(self, t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _value_noise(self, gx: np.ndarray, gy: np.ndarray, step: float, salt: int) -> np.ndarray:
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

    def generate_water_and_land(self, chunk_x: int, chunk_y: int) -> list[list[str]]:
        size = CHUNK_SIZE
        x0 = chunk_x * size
        y0 = chunk_y * size
        tiles = self.generate_area_tiles(x0, y0, size, size)
        return tiles.tolist()

    def generate_heightmap(self, chunk_x: int, chunk_y: int) -> list[list[float]]:
        return (np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=float)).tolist()

    def assign_biomes(self, heightmap: list[list[float]]) -> list[list[str]]:
        size = CHUNK_SIZE
        return (np.full((size, size), "grass", dtype=object)).tolist()

    _DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    _DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    _DW = np.array([np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0)])

    @staticmethod
    def _shift_edge(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
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
        drops = np.stack(drops, axis=0)
        kmax = np.argmax(drops, axis=0)
        maxdrop = drops[kmax, np.arange(H)[:, None], np.arange(W)]
        uphill = maxdrop <= 0
        if uphill.any():
            kmin = np.argmin(np.abs(drops), axis=0)
            kmax = np.where(uphill, kmin, kmax)
            maxdrop = drops[kmax, np.arange(H)[:, None], np.arange(W)]
        to_y = (np.arange(H)[:, None] + cls._DY[kmax]).clip(0, H - 1)
        to_x = (np.arange(W)[None, :] + cls._DX[kmax]).clip(0, W - 1)
        to_index = (to_y * W + to_x).ravel()
        return to_index, maxdrop.ravel()

    def generate_area_water_mask(self, x0_tile: int, y0_tile: int, width_tiles: int, height_tiles: int) -> np.ndarray:
        p = MacroParams(seed=self.seed)
        band_tiles = int(p.coast_band_tiles)
        HALO = max(32, band_tiles + 16)

        y, x = np.mgrid[-HALO:height_tiles + HALO, -HALO:width_tiles + HALO]
        gx = x0_tile + x
        gy = y0_tile + y

        macro = MacroContinents(p)
        field_halo = macro.continent_field(gx, gy)

        if self._sea_level is None:
            self._sea_level = float(np.percentile(field_halo, 60.0))
        thr = self._sea_level

        coarse_land = field_halo >= thr
        dist_land = euclidish_dist(coarse_land)
        dist_water = euclidish_dist(~coarse_land)
        band_land_halo = (dist_land <= band_tiles)
        band_water_halo = (dist_water <= band_tiles)
        geo_band_halo = band_land_halo | band_water_halo

        detail = macro.noise.fbm(gx, gy, p.coast_detail_freq, octaves=3, salt=1300)
        detail = (detail - 0.5) * p.coast_detail_amp
        shaped = field_halo + detail * geo_band_halo
        land_halo = shaped >= thr

        # Land-side only edits (do not mutate ocean side)
        land_halo = band_limited_close_8(land_halo, band_land_halo, iters=p.close_iters)
        land_halo = prune_tendrils(land_halo, band_land_halo, min_width=3)
        land_halo = majority_smooth(land_halo, band_land_halo, rounds=1)

        # Enforce connectivity to coarse_land to prevent offshore flips
        candidate = land_halo
        connected = geodesic_reconstruct(candidate=candidate, core=coarse_land, max_iters=band_tiles + 4)
        land_halo[geo_band_halo] = connected[geo_band_halo]

        land_halo = fill_small_lakes_on_halo(land_halo, max_area=150)
        ocean_mask_halo = ~land_halo

        inland = np.clip((field_halo - thr) / (1 - thr + 1e-6), 0.0, 1.0)
        relief = macro.noise.fbm(gx * 0.8 + 77.0, gy * 0.8 - 33.0, freq=1.0 / 280.0, octaves=4, salt=1700)
        relief = 1.0 - np.abs(relief - 0.5) * 2.0
        height_halo = inland + 0.45 * (inland ** 1.2) * (relief - 0.5)

        dist_to_ocean_halo = manhattan_dist_to_water(land_halo)
        dist_norm = np.minimum(dist_to_ocean_halo, 12).astype(np.float32) / 12.0
        height_halo += 0.04 * dist_norm
        height_halo = np.where(ocean_mask_halo, -10.0, height_halo)

        to_index_h, _ = self._steepest_descent(height_halo)
        Hh, Wh = height_halo.shape
        Nh = Hh * Wh
        order_h = np.argsort(height_halo.ravel())[::-1]
        acc_h = np.ones(Nh, dtype=np.float32)
        np.add.at(acc_h, to_index_h[order_h], acc_h[order_h])
        FA_h = acc_h.reshape(Hh, Wh)

        neighbors_h = [self._shift_edge(height_halo, int(dy), int(dx)) for dy, dx in zip(self._DY, self._DX)]
        slopes_h = [np.maximum(height_halo - nb, 0.0) for nb in neighbors_h]
        slope_h = np.maximum.reduce(slopes_h)
        flatness_h = 1.0 / (1.0 + 30.0 * slope_h)

        river_power_h = FA_h * flatness_h * land_halo
        if land_halo.any():
            thresh = np.percentile(river_power_h[land_halo], 97.0)
        else:
            thresh = np.inf
        rivers_h = river_power_h >= thresh

        def touch(mask: np.ndarray) -> np.ndarray:
            return np.logical_or.reduce([self._shift_edge(mask, int(dy), int(dx)) for dy, dx in zip(self._DY, self._DX)])

        near_coast_h = dist_to_ocean_halo <= 2
        mouth_h = rivers_h & touch(ocean_mask_halo)
        rivers_h = (rivers_h & ~near_coast_h) | mouth_h
        isolated_near = near_coast_h & rivers_h & (~touch(rivers_h))
        rivers_h[isolated_near] = False

        sl = (slice(HALO, HALO + height_tiles), slice(HALO, HALO + width_tiles))
        rivers = rivers_h[sl]
        ocean_mask = ocean_mask_halo[sl]

        water_mask = ocean_mask | (rivers & ~ocean_mask)
        return water_mask

    def generate_area_tiles(self, x0_tile: int, y0_tile: int, width_tiles: int, height_tiles: int) -> np.ndarray:
        mask = self.generate_area_water_mask(x0_tile, y0_tile, width_tiles, height_tiles)
        return np.where(mask, "water", "grass").astype(object)


def geodesic_reconstruct(candidate: np.ndarray, core: np.ndarray, max_iters: int = 32) -> np.ndarray:
    """Keep only cells in `candidate` that are 8-connected to `core`."""
    rec = core.copy()
    for _ in range(max(0, int(max_iters))):
        new = _dilate_8(rec) & candidate
        if np.array_equal(new, rec):
            break
        rec = new
    return rec
