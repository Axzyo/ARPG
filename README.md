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