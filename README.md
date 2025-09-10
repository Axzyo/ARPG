# Infinite Rouge

A minimalist roguelike game built with Python and Pygame.

## Features

- Infinite scrolling world
- Smooth player movement
- Debug mode (press F3 to toggle)
- Grid-based world system

## Controls

- WASD: Move player
- F3: Toggle debug info
- ESC: Quit game

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the game:
```bash
python main.py
```

## Development

The game is built with a modular structure to allow for easy expansion. The main components are:

- `Game`: Main game loop and state management
- `Player`: Player entity with movement and rendering
- `Camera`: Handles viewport scrolling and coordinate conversion

## License

MIT License 

## Very Important: Modularity & Interface Contracts

This project is intentionally extremely modular. Every module must expose a small, stable interface with documented inputs, outputs, errors, and side effects so implementations can be hot‑swapped later without changing callers.

When adding or changing a module, include an "Interface Contract" section at the top of the file and follow these rules:
- Document Inputs, Outputs, Errors, Side Effects, and Performance Notes.
- Preserve public function/class signatures unless bumping the module version.
- Provide a minimal usage example.

Example contract (pathfinding):

- File: `pathfinding.py`
- Function: `find_path(start, goal)`
  - Inputs:
    - `start: tuple[int, int]` world position in pixels `(x, y)`
    - `goal: tuple[int, int]` world position in pixels `(x, y)`
  - Output:
    - `list[tuple[int, int]]` waypoints in pixels, including `start`/`goal` if reachable; empty list if no path
  - Errors:
    - Invalid inputs → `ValueError`
    - Unreachable is not an error; return empty list
  - Side effects: none
  - Notes: Implementation may query walkability via a provider; callers do not pass world objects here
  - Version: v1

See `ARCHITECTURE.md` for the full template and the list of stable module interfaces.