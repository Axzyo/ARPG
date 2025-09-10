import pygame
import math
from constants import *
from memory import Memory, MemoryBank

class NPC:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.memory_bank = MemoryBank(max_memories=200)
        self.goals = []
        self.current_action = "idle"
        
        # Visual properties
        self.color = (100, 100, 255)
        self.size = NPC_SIZE
        
        # Vision properties
        self.vision_range = NPC_VISION_RANGE
        self.field_of_view = 120
        self.facing_direction = 0
        
        # Tracking what the NPC has seen
        self.last_visible_tiles = set()
        self.last_visible_entities = set()
        
        # Backstory removed; personality can be inferred later from memories
        self.personality_traits = []
        
        self.health = 100
        self.energy = 100
        self.last_update_time = 0
        
        # Initial identity memory
        self.memory_bank.create_memory(
            sense="thought",
            location=(x, y),
            target="self",
            data=f"I am {name}, and I have awakened at this location",
            significance=8.0,
            npc_name=name
        )

    # --- Memory helpers ---
    def add_memory(self, sense: str, target: str, data: str, significance: float = 1.0):
        return self.memory_bank.create_memory(
            sense=sense,
            location=(self.x, self.y),
            target=target,
            data=data,
            significance=significance,
            npc_name=self.name
        )

    # --- Goals and actions ---
    def add_goal(self, goal, priority=1):
        self.goals.append({
            "description": goal,
            "priority": priority,
            "created_time": pygame.time.get_ticks(),
            "completed": False
        })
        self.goals.sort(key=lambda g: g["priority"], reverse=True)
    
    def complete_goal(self, goal_description):
        for goal in self.goals:
            if goal["description"] == goal_description:
                goal["completed"] = True
                self.add_memory("thought", "self", f"Completed goal: {goal_description}", 3.0)
                break
    
    def remove_completed_goals(self):
        self.goals = [goal for goal in self.goals if not goal["completed"]]
    
    def get_active_goals(self):
        return [goal for goal in self.goals if not goal["completed"]]
    
    def set_action(self, action):
        if self.current_action != action:
            self.add_memory("thought", "self", f"Changed action from {self.current_action} to {action}", 1.0)
            self.current_action = action

    # --- Update/render ---
    def update(self, dt):
        self.last_update_time = pygame.time.get_ticks()
        if self.current_action == "idle":
            pass

    def render(self, screen, camera_x, camera_y, show_vision=False):
        screen_x = self.x - camera_x
        screen_y = self.y - camera_y
        if (-self.size <= screen_x <= SCREEN_WIDTH + self.size and 
            -self.size <= screen_y <= SCREEN_HEIGHT + self.size):
            if show_vision:
                self._draw_vision_cone(screen, screen_x, screen_y)
            pygame.draw.circle(screen, self.color, (int(screen_x), int(screen_y)), self.size // 2)
            pygame.draw.circle(screen, WHITE, (int(screen_x), int(screen_y)), 3)
            direction_length = self.size // 2 + UI_SMALL_SPACING
            end_x = screen_x + math.cos(math.radians(self.facing_direction)) * direction_length
            end_y = screen_y + math.sin(math.radians(self.facing_direction)) * direction_length
            pygame.draw.line(screen, WHITE, (int(screen_x), int(screen_y)), (int(end_x), int(end_y)), 2)
            font = pygame.font.Font(None, FONT_NORMAL)
            name_surface = font.render(self.name, True, WHITE)
            name_rect = name_surface.get_rect()
            name_rect.centerx = int(screen_x)
            name_rect.bottom = int(screen_y) - self.size // 2 - UI_SMALL_SPACING
            screen.blit(name_surface, name_rect)

    # Vision helpers (unchanged)
    def _draw_vision_cone(self, screen, screen_x, screen_y):
        vision_range_pixels = self.vision_range * TILE_SIZE
        half_fov = self.field_of_view / 2
        start_angle = self.facing_direction - half_fov
        end_angle = self.facing_direction + half_fov
        points = [(screen_x, screen_y)]
        num_points = 20
        for i in range(num_points + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_points
            x = screen_x + math.cos(math.radians(angle)) * vision_range_pixels
            y = screen_y + math.sin(math.radians(angle)) * vision_range_pixels
            points.append((x, y))
        vision_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(vision_surface, (255, 255, 0, 30), points)
        screen.blit(vision_surface, (0, 0))

    # Distance/movement helpers (unchanged)
    def get_distance_to(self, x, y):
        return ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5

    def move_towards(self, target_x, target_y, speed, dt):
        distance = self.get_distance_to(target_x, target_y)
        if distance > 0:
            dx = (target_x - self.x) / distance
            dy = (target_y - self.y) / distance
            self.face_towards(target_x, target_y)
            move_distance = min(speed * dt, distance)
            self.x += dx * move_distance
            self.y += dy * move_distance
            if distance < NPC_MOVEMENT_PRECISION:
                self.set_action("idle")
                return True
            else:
                self.set_action("moving")
                return False

    def __str__(self):
        return f"NPC({self.name}) at ({self.x:.1f}, {self.y:.1f}) - Action: {self.current_action}"

    # Sensing/memories (unchanged from earlier except disabled processing hooks)
    def observe_player(self, player_x: float, player_y: float, significance: float = 5.0):
        distance = self.get_distance_to(player_x, player_y)
        return self.add_memory("saw", "player", f"Spotted the player at distance {distance:.1f}", significance)
    
    def hear_sound(self, source: str, description: str, significance: float = 2.0):
        return self.add_memory("heard", source, description, significance)

    def get_relevant_memories(self, radius: float = 100.0, max_age: float = 300.0):
        return self.memory_bank.get_relevant_memories(current_location=(self.x, self.y), radius=radius, max_age_seconds=max_age)

    def get_memories_about(self, target: str):
        return self.memory_bank.get_memories_by_target(target)

    def get_visual_memories(self):
        return self.memory_bank.get_memories_by_sense("saw")

    def get_most_important_memories(self, count: int = 5):
        return self.memory_bank.get_most_significant_memories(count)

    # Backstory-related methods removed

    def _normalize_angle(self, angle):
        return angle % 360

    def _angle_difference(self, angle1, angle2):
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)

    def _get_angle_to_point(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        angle = math.degrees(math.atan2(dy, dx))
        return self._normalize_angle(angle)

    def set_facing_direction(self, angle):
        self.facing_direction = self._normalize_angle(angle)

    def face_towards(self, target_x, target_y):
        self.facing_direction = self._get_angle_to_point(target_x, target_y)

    def _is_in_field_of_view(self, target_x, target_y):
        angle_to_target = self._get_angle_to_point(target_x, target_y)
        angle_diff = self._angle_difference(self.facing_direction, angle_to_target)
        return angle_diff <= self.field_of_view / 2

    def _has_line_of_sight(self, target_x, target_y, obstacles=None):
        if obstacles is None:
            obstacles = []
        distance = math.sqrt((target_x - self.x) ** 2 + (target_y - self.y) ** 2)
        distance_in_tiles = distance / TILE_SIZE
        return distance_in_tiles <= self.vision_range

    def _can_see_point(self, target_x, target_y, obstacles=None):
        return (self._is_in_field_of_view(target_x, target_y) and self._has_line_of_sight(target_x, target_y, obstacles))

    def _get_visible_tiles(self, chunk_manager, obstacles=None):
        visible_tiles = set()
        npc_tile_x = int(self.x // TILE_SIZE)
        npc_tile_y = int(self.y // TILE_SIZE)
        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                tile_x = npc_tile_x + dx
                tile_y = npc_tile_y + dy
                world_x = tile_x * TILE_SIZE + TILE_SIZE // 2
                world_y = tile_y * TILE_SIZE + TILE_SIZE // 2
                if self._can_see_point(world_x, world_y, obstacles):
                    chunk_x = tile_x // CHUNK_SIZE
                    chunk_y = tile_y // CHUNK_SIZE
                    if (chunk_x, chunk_y) in chunk_manager.chunks:
                        chunk = chunk_manager.chunks[(chunk_x, chunk_y)]
                        local_tile_x = tile_x % CHUNK_SIZE
                        local_tile_y = tile_y % CHUNK_SIZE
                        if (0 <= local_tile_x < CHUNK_SIZE and 0 <= local_tile_y < CHUNK_SIZE):
                            tile_type = chunk.tiles[local_tile_y][local_tile_x]
                            visible_tiles.add((tile_x, tile_y, tile_type))
        return visible_tiles

    def _get_visible_entities(self, entities, obstacles=None):
        visible_entities = set()
        for entity in entities:
            if hasattr(entity, 'x') and hasattr(entity, 'y'):
                if self._can_see_point(entity.x, entity.y, obstacles):
                    if hasattr(entity, 'name'):
                        entity_id = f"npc_{entity.name}"
                        entity_type = "npc"
                    else:
                        entity_id = "player"
                        entity_type = "player"
                    visible_entities.add((entity_id, entity_type, entity.x, entity.y))
        return visible_entities

    def see(self, chunk_manager, entities=None, obstacles=None):
        if entities is None:
            entities = []
        current_visible_tiles = self._get_visible_tiles(chunk_manager, obstacles)
        current_visible_entities = self._get_visible_entities(entities, obstacles)
        # Change processors currently disabled
        self.last_visible_tiles = current_visible_tiles
        self.last_visible_entities = current_visible_entities

 