import pygame
import math
import random
import threading
import time
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
        
        # Chat system with dialogue queue
        self.dialogue_queue = []
        self.current_dialogue_index = 0
        self.dialogue_timer = 0
        self.dialogue_duration = 3.0
        self.chat_response = ""
        self.chat_response_timer = 0
        self.chat_response_duration = 5.0
        self.is_thinking = False
        self.thinking_dots = 0
        self.preset_responses_sent = False
        
        # Backstory removed; personality can be inferred later from memories
        self.personality_traits = []
        
        # Random preset responses for backup
        self.preset_responses = [
            "...",
            "*looks confused*",
            "Huh?",
            "Ugh...",
            "I have a headache...",
            "Not now...",
            "Give me a sec",
            "*yawns*",
            "Can't think straight...",
            "What do you want?",
        ]
        
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
        if self.is_thinking:
            self.thinking_dots = (self.thinking_dots + dt * 3) % 4
        if self.dialogue_queue and self.dialogue_timer > 0:
            self.dialogue_timer -= dt
            if self.dialogue_timer <= 0:
                self._advance_dialogue()
        if self.chat_response_timer > 0 and not self.dialogue_queue:
            self.chat_response_timer -= dt
            if self.chat_response_timer <= 0:
                self.chat_response = ""
                self.is_thinking = False
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
            if self.chat_response or self.is_thinking:
                if self.is_thinking:
                    dots = "." * int(self.thinking_dots)
                    display_text = f"thinking{dots}"
                else:
                    display_text = self.chat_response
                chat_font = pygame.font.Font(None, FONT_SMALL)
                words = display_text.split(' ')
                lines = []
                current_line = []
                line_width = 0
                max_width = max(120, int(200 * SCALE))
                for word in words:
                    word_surface = chat_font.render(word + ' ', True, WHITE)
                    word_width = word_surface.get_width()
                    if line_width + word_width > max_width and current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        line_width = word_width
                    else:
                        current_line.append(word)
                        line_width += word_width
                if current_line:
                    lines.append(' '.join(current_line))
                if lines:
                    line_height = max(12, int(18 * SCALE))
                    total_height = len(lines) * line_height + UI_MARGIN
                    max_line_width = max(chat_font.size(line)[0] for line in lines)
                    bubble_width = max_line_width + (UI_PADDING * 2)
                    bubble_rect = pygame.Rect(
                        int(screen_x - bubble_width // 2),
                        int(screen_y - self.size // 2 - (UI_LARGE_SPACING + UI_MARGIN) - total_height),
                        bubble_width,
                        total_height
                    )
                    if self.is_thinking:
                        bg_color = (30, 30, 80, 200)
                        border_color = (100, 100, 255)
                    else:
                        bg_color = (50, 50, 50, 200)
                        border_color = WHITE
                    pygame.draw.rect(screen, bg_color, bubble_rect, border_radius=10)
                    pygame.draw.rect(screen, border_color, bubble_rect, width=2, border_radius=10)
                    for i, line in enumerate(lines):
                        line_surface = chat_font.render(line, True, WHITE)
                        line_rect = line_surface.get_rect()
                        line_rect.centerx = bubble_rect.centerx
                        line_rect.y = bubble_rect.y + UI_SMALL_SPACING + i * line_height
                        screen.blit(line_surface, line_rect)

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

    # --- Chat ---
    def set_chat_response(self, response):
        print(f"[NPC] {self.name} received chat response: '{response}'")
        parsed_response = self._parse_llm_response(response)
        if parsed_response.strip():
            self.add_to_dialogue_queue(parsed_response)
        self.is_thinking = False

    def start_thinking(self):
        print(f"[NPC] {self.name} started thinking...")
        self.is_thinking = True
        self.thinking_dots = 0
        self.chat_response = ""
        self.preset_responses_sent = False

    def add_to_response(self, text_chunk):
        if not hasattr(self, '_raw_response_buffer'):
            self._raw_response_buffer = ""
        self._raw_response_buffer += text_chunk
        if self.is_thinking:
            self.is_thinking = False
            self.chat_response = "..."
            self.chat_response_timer = self.chat_response_duration
            print(f"[NPC] {self.name} started receiving streaming response...")
        self.chat_response_timer = self.chat_response_duration

    def complete_response(self, full_response):
        print(f"[NPC] {self.name} completed streaming response: '{full_response}'")
        response_to_parse = getattr(self, '_raw_response_buffer', full_response)
        parsed_response = self._parse_llm_response(response_to_parse)
        if parsed_response.strip():
            self.add_to_dialogue_queue(parsed_response)
            self.add_memory("said", "player", f"I said: '{parsed_response}'", significance=6.0)
        else:
            # Clear any temporary ellipsis from streaming
            self.chat_response = ""
        self.is_thinking = False
        if hasattr(self, '_raw_response_buffer'):
            delattr(self, '_raw_response_buffer')

    def is_player_nearby(self, player_x, player_y, interaction_distance=None):
        if interaction_distance is None:
            interaction_distance = NPC_INTERACTION_DISTANCE
        distance = self.get_distance_to(player_x, player_y)
        return distance <= interaction_distance

    def _parse_llm_response(self, response):
        import re
        import json
        print(f"[NPC] {self.name}: Parsing response: '{response}'")
        response = response.strip()
        if response.startswith("[TEST]"):
            return response
        # Do not show error messages as dialogue
        if response.startswith("[ERROR]"):
            print(f"[NPC] {self.name}: LLM error; suppressing dialogue")
            return ""
        if response in {"...", "..", ".", ""}:
            return self._generate_memory_based_reply()
        # Drop prompt-echo artifacts
        lower = response.lower()
        if lower.startswith("example:") or lower.startswith("player says:") or lower.startswith("reply with:"):
            return self._generate_memory_based_reply()
        # Try to parse JSON first (kept for safety, though prompt asks plain text)
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_text = response[json_start:json_end]
                parsed_json = json.loads(json_text)
                if isinstance(parsed_json, dict) and "say" in parsed_json:
                    dialogue_text = parsed_json["say"].strip()
                    if dialogue_text and len(dialogue_text) < 200:
                        return dialogue_text
        except Exception:
            pass
        # Legacy clean
        cleaned = response
        cleaned = cleaned.replace("[Character Name]", self.name)
        cleaned = cleaned.replace("[character name]", self.name)
        cleaned = cleaned.replace("Character Name", self.name)
        cleaned = cleaned.replace("character name", self.name)
        if cleaned.lower().startswith(f"{self.name.lower()} says:"):
            cleaned = cleaned[len(f"{self.name} says:"):].strip()
        elif cleaned.lower().startswith(f"{self.name.lower()}:"):
            cleaned = cleaned[len(f"{self.name}:"):].strip()
        name_pattern = r'^[A-Za-z]+\s*(says|responds|replies)?:\s*'
        cleaned = re.sub(name_pattern, '', cleaned, flags=re.IGNORECASE)
        if cleaned and len(cleaned) < 250 and not cleaned.startswith('{'):
            return cleaned
        return self._generate_memory_based_reply()

    def add_to_dialogue_queue(self, message):
        # Keep only the first line and strip speaker labels; ignore empty
        if not message or not message.strip():
            return
        first_line = message.splitlines()[0].strip()
        import re as _re
        first_line = _re.sub(r'^\s*[A-Za-z]+\s*(says|responds|replies)?:\s*', '', first_line)
        # Strip surrounding quotes/backticks often produced by models
        first_line = first_line.strip('\'"“”‘’`')
        if not first_line:
            return
        self.dialogue_queue.append(first_line)
        print(f"[NPC] {self.name}: Added to dialogue queue: '{message}'")
        if len(self.dialogue_queue) == 1:
            self._start_displaying_dialogue()

    def _start_displaying_dialogue(self):
        if self.dialogue_queue and self.current_dialogue_index < len(self.dialogue_queue):
            self.chat_response = self.dialogue_queue[self.current_dialogue_index]
            self.dialogue_timer = self.dialogue_duration
            self.chat_response_timer = self.dialogue_duration
            print(f"[NPC] {self.name}: Now displaying: '{self.chat_response}'")

    def _advance_dialogue(self):
        self.current_dialogue_index += 1
        if self.current_dialogue_index < len(self.dialogue_queue):
            self._start_displaying_dialogue()
        else:
            self.dialogue_queue.clear()
            self.current_dialogue_index = 0
            self.chat_response = ""
            self.chat_response_timer = 0
            print(f"[NPC] {self.name}: Dialogue queue completed")

    def _generate_memory_based_reply(self) -> str:
        heard = [m for m in self.memory_bank.get_memories_by_sense("heard") if m.target == "player"]
        last_player_line = heard[-1].data if heard else ""
        text = last_player_line.lower()
        import re as _re2
        greetings_pattern = _re2.compile(r"\b(hello|hi|hey|yo|greetings)\b")
        consent = ["come closer", "come here", "follow", "approach", "help me"]
        if greetings_pattern.search(text):
            return random.choice(["Hey.", "Hi.", "Hello.", "Yo."])
        if any(c in text for c in consent):
            return random.choice(["Sure.", "Yeah.", "Okay.", "Alright."])
        questions = ["?", "where", "what", "who", "how", "why", "when"]
        if any(q in text for q in questions):
            return "I don't know."
        thanks = ["thank", "thanks", "ty"]
        if any(t in text for t in thanks):
            return random.choice(["Sure.", "Anytime.", "No problem."])
        if self.name.lower() in text:
            return random.choice(["Yeah?", "What?", "I'm listening."])
        return random.choice(["Alright.", "Okay.", "Hm.", "Right."])
 