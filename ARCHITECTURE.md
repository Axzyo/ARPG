# Architecture & Interface Contracts

## Design goals
- Extremely modular: any system can be replaced if it preserves its public inputs/outputs.
- Data-first: clear types and flags; code consumes interfaces, not concrete implementations.
- Deterministic core: fixed-step updates and seeded generation to support training/headless runs.

## Interface contract template (copy into each module)

```
Module: <name> (v1)

Purpose:
- <one-paragraph summary>

Public API:
- <symbol> : <signature>
  - Inputs:
    - <name>: <type> — <desc>
  - Output:
    - <type> — <desc>
  - Errors:
    - <error type> — <when and how callers should handle>
  - Side effects:
    - <none | description>
  - Performance notes:
    - <O(...) or notes>
  - Example:
    ```python
    # minimal usage example showing inputs/outputs
    ```

Change policy:
- Backward-compatible changes allowed within v1 (internal optimizations, bug fixes).
- Breaking changes require bumping module version and documenting migration notes here.

Dependencies:
- <list any required providers (e.g., walkability query), environment variables, or assets>

Test contracts:
- Unit tests MUST validate boundary cases, error behavior, and output invariants.
```

## Current/Planned modules and stable interfaces

- World Query (v1):
  - `get_tile(x_px:int, y_px:int) -> dict | None`
  - `is_walkable(x_px:int, y_px:int) -> bool`
  - `raycast(x0:int, y0:int, x1:int, y1:int) -> dict{hit:bool, point:(int,int), tile:...}`
  - `neighbors(tile_xy:tuple[int,int]) -> list[tuple[int,int]]`
  - Notes: Consumers do NOT depend on chunk internals.

- Perception (v1):
  - `get_observation(npc) -> dict`  # compact observation of tiles/entities in FOV

- Pathfinding (v1):
  - `find_path(start:tuple[int,int], goal:tuple[int,int]) -> list[tuple[int,int]]`
  - Returns empty list if unreachable; no exceptions for “no path”.

- Collision (v1):
  - `sweep_aabb(pos:(float,float), vel:(float,float)) -> dict{pos:(float,float), collided:bool}`

- Inventory (v1):
  - `add(item_id:str, qty:int) -> int` (returns qty actually added)
  - `remove(item_id:str, qty:int) -> int`
  - `get(item_id:str) -> int`

- Combat (v1):
  - `apply_damage(entity_id:str, amount:int, source:str|None) -> dict{hp:int, dead:bool}`

## Implementation guidelines

- One module per file, e.g., `pathfinding.py`, `collision.py`.
- Avoid importing concrete game objects; depend on small provider interfaces where needed.
- Return values must be well-typed and documented; avoid implicit global state.
- Unreachable/edge outcomes should be represented in outputs rather than raising (unless invalid input).
- Provide a minimal usage example in the docblock of each public function/class.

## Versioning & migration
- Start at v1. Bump to v2 only on breaking changes; document migration steps in the module header and here.

## Headless/training mode
- Modules used during training must produce the same outputs in headless mode.
- Randomness must be seeded via passed-in seed or global seed managed by the game core.


