import pygame
import uuid
import math
from typing import List, Dict, Optional, Any

class Memory:
    def __init__(
        self,
        sense: str,
        target: str,
        description: Any,
        location1: tuple,
        location2: Optional[tuple] = None,
        significance_0_to_1: float = 1.0,
        time_ms: Optional[int] = None,
        meta: Optional[Dict] = None,
    ):
        """
        New memory schema (hot-swappable, JSON-friendly):
        {
            "sense": str,
            "target": str,
            "description": Any,               # e.g., dict for items, list for state tags, or string
            "location1": [x, y],              # self position
            "location2": [x, y],              # target position (or same as location1)
            "time": int,                      # pygame ticks
            "meta": {
                "id": str,                    # unique id
                "sig": float,                 # significance in [0,1]
                "count": int,
                "first_seen": int,
                "last_seen": int,
                "links": list[str]
            }
        }
        """
        self.sense = sense
        self.target = target
        self.description = description
        self.location1 = (location1[0], location1[1]) if location1 else (0, 0)
        self.location2 = (location2[0], location2[1]) if location2 else self.location1
        self.time = time_ms if time_ms is not None else pygame.time.get_ticks()
        # Normalize significance to [0,1]
        sig = max(0.0, min(1.0, significance_0_to_1))
        self.meta = meta.copy() if meta else {}
        self.meta.setdefault('id', f"evt_{self.time}_{int(self.location2[0])}_{int(self.location2[1])}_{uuid.uuid4().hex[:2]}")
        self.meta.setdefault('sig', sig)
        self.meta.setdefault('count', 1)
        self.meta.setdefault('first_seen', self.time)
        self.meta.setdefault('last_seen', self.time)
        self.meta.setdefault('links', [])
    
    def age_in_seconds(self) -> float:
        return (pygame.time.get_ticks() - self.time) / 1000.0
    
    def distance_from(self, x: float, y: float) -> float:
        # Distance from the actor (location1) by default
        return math.sqrt((self.location1[0] - x) ** 2 + (self.location1[1] - y) ** 2)
    
    def __str__(self) -> str:
        sig = self.meta.get('sig', 0.0)
        return f"Memory({self.sense}): {self.target} - {self.description} [sig: {sig:.2f}]"
    
    def to_dict(self) -> Dict:
        return {
            'sense': self.sense,
            'target': self.target,
            'description': self.description,
            'location1': [self.location1[0], self.location1[1]],
            'location2': [self.location2[0], self.location2[1]],
            'time': self.time,
            'meta': self.meta,
        }

class MemoryBank:
    def __init__(self, max_memories: int = 500):
        """
        Memory storage and retrieval system
        
        Args:
            max_memories: Maximum number of memories to store
        """
        self.memories: List[Memory] = []
        self.max_memories = max_memories
        self.sense_index: Dict[str, List[Memory]] = {}
        self.target_index: Dict[str, List[Memory]] = {}
    
    def add_memory(self, memory: Memory) -> None:
        """Add a new memory to the bank"""
        self.memories.append(memory)
        
        # Update indexes
        if memory.sense not in self.sense_index:
            self.sense_index[memory.sense] = []
        self.sense_index[memory.sense].append(memory)
        
        if memory.target not in self.target_index:
            self.target_index[memory.target] = []
        self.target_index[memory.target].append(memory)
        
        # Remove oldest memories if we exceed the limit
        if len(self.memories) > self.max_memories:
            self._remove_oldest_memory()
    
    def create_memory(self, sense: str, location: tuple, target: str, data: Any, significance: float = 1.0, npc_name: str = "Unknown", target_location: Optional[tuple] = None, meta: Optional[Dict] = None) -> Memory:
        """Create and add a new memory (backward-compatible signature).
        - `location` maps to `location1`
        - `data` maps to `description`
        - `significance` in [0,10] is normalized to meta['sig'] in [0,1]
        - `target_location` (optional) maps to `location2` (defaults to `location1`)
        """
        sig01 = max(0.0, min(1.0, (significance / 10.0)))
        memory = Memory(
            sense=sense,
            target=target,
            description=data,
            location1=location,
            location2=target_location or location,
            significance_0_to_1=sig01,
            meta=meta,
        )
        self.add_memory(memory)
        
        # Print new memory to console with NPC identification
        print(f"[{npc_name}] NEW MEMORY: {memory}")
        
        return memory
    
    def _remove_oldest_memory(self) -> None:
        """Remove the oldest, least significant memory"""
        if not self.memories:
            return
        
        # Find the oldest memory with lowest significance (meta.sig)
        oldest_memory = min(self.memories, key=lambda m: (m.meta.get('sig', 0.0), -m.time))
        self.remove_memory(oldest_memory)
    
    def remove_memory(self, memory: Memory) -> None:
        """Remove a specific memory from the bank"""
        if memory in self.memories:
            self.memories.remove(memory)
            
            # Update indexes
            if memory.sense in self.sense_index:
                if memory in self.sense_index[memory.sense]:
                    self.sense_index[memory.sense].remove(memory)
                if not self.sense_index[memory.sense]:
                    del self.sense_index[memory.sense]
            
            if memory.target in self.target_index:
                if memory in self.target_index[memory.target]:
                    self.target_index[memory.target].remove(memory)
                if not self.target_index[memory.target]:
                    del self.target_index[memory.target]
    
    def get_memories_by_sense(self, sense: str) -> List[Memory]:
        """Get all memories of a specific sense type"""
        return self.sense_index.get(sense, []).copy()
    
    def get_memories_by_target(self, target: str) -> List[Memory]:
        """Get all memories about a specific target"""
        return self.target_index.get(target, []).copy()
    
    def get_memories_by_location(self, x: float, y: float, radius: float) -> List[Memory]:
        """Get memories within a certain radius of a location"""
        return [m for m in self.memories if m.distance_from(x, y) <= radius]
    
    def get_memories_by_significance(self, min_significance: float = 0.0) -> List[Memory]:
        """Get memories above a certain significance threshold"""
        return [m for m in self.memories if m.meta.get('sig', 0.0) >= min_significance]
    
    def get_recent_memories(self, max_age_seconds: float) -> List[Memory]:
        """Get memories newer than max_age_seconds"""
        return [m for m in self.memories if m.age_in_seconds() <= max_age_seconds]
    
    def get_relevant_memories(self, current_location: tuple, sense_filter: Optional[str] = None, 
                            target_filter: Optional[str] = None, radius: float = 100.0, 
                            max_age_seconds: float = 300.0, min_significance: float = 0.0) -> List[Memory]:
        """
        Get memories relevant to current context
        
        Args:
            current_location: Current (x, y) position
            sense_filter: Filter by sense type (optional)
            target_filter: Filter by target (optional)
            radius: Maximum distance from current location
            max_age_seconds: Maximum age of memories to consider
            min_significance: Minimum significance threshold
        """
        relevant = []
        
        for memory in self.memories:
            # Check age
            if memory.age_in_seconds() > max_age_seconds:
                continue
            
            # Check significance
            if memory.meta.get('sig', 0.0) < min_significance:
                continue
            
            # Check location proximity
            if memory.distance_from(current_location[0], current_location[1]) > radius:
                continue
            
            # Check sense filter
            if sense_filter and memory.sense != sense_filter:
                continue
            
            # Check target filter
            if target_filter and memory.target != target_filter:
                continue
            
            relevant.append(memory)
        
        # Sort by relevance (combination of significance and recency)
        def relevance_score(mem):
            age_factor = max(0.1, 1.0 - (mem.age_in_seconds() / max_age_seconds))
            distance_factor = max(0.1, 1.0 - (mem.distance_from(current_location[0], current_location[1]) / radius))
            return mem.meta.get('sig', 0.0) * age_factor * distance_factor
        
        relevant.sort(key=relevance_score, reverse=True)
        return relevant
    
    def get_most_significant_memories(self, count: int = 10) -> List[Memory]:
        """Get the most significant memories"""
        sorted_memories = sorted(self.memories, key=lambda m: m.meta.get('sig', 0.0), reverse=True)
        return sorted_memories[:count]
    
    def search_memories(self, query: str) -> List[Memory]:
        """Search memories by text content in data or target fields"""
        query_lower = query.lower()
        matching = []
        
        for memory in self.memories:
            desc_text = str(memory.description).lower()
            if (query_lower in desc_text or 
                query_lower in memory.target.lower()):
                matching.append(memory)
        
        # Sort by significance
        matching.sort(key=lambda m: m.meta.get('sig', 0.0), reverse=True)
        return matching
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about the memory bank"""
        if not self.memories:
            return {
                'total_memories': 0,
                'sense_breakdown': {},
                'avg_significance': 0,
                'oldest_memory_age': 0,
                'newest_memory_age': 0
            }
        
        sense_counts = {}
        for sense, memories in self.sense_index.items():
            sense_counts[sense] = len(memories)
        
        significances = [m.meta.get('sig', 0.0) for m in self.memories]
        ages = [m.age_in_seconds() for m in self.memories]
        
        return {
            'total_memories': len(self.memories),
            'sense_breakdown': sense_counts,
            'avg_significance': (sum(significances) / len(significances)) if significances else 0.0,
            'oldest_memory_age': max(ages),
            'newest_memory_age': min(ages)
        }
    
    def clear_old_memories(self, max_age_seconds: float) -> int:
        """Remove memories older than max_age_seconds. Returns count of removed memories."""
        old_memories = [m for m in self.memories if m.age_in_seconds() > max_age_seconds]
        for memory in old_memories:
            self.remove_memory(memory)
        return len(old_memories)
    
    def __len__(self) -> int:
        return len(self.memories)
    
    def __str__(self) -> str:
        return f"MemoryBank({len(self.memories)} memories)" 