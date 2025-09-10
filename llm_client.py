import os
import requests
import threading
import time
import hashlib
import json
from typing import Callable
from ctransformers import AutoModelForCausalLM

class LLMClient:
    def __init__(self, base_url: str | None = None, model_path: str | None = None):
        """Initialize local LLM client (in-process, no server).
        
        Args:
            base_url: Unused (kept for compatibility)
            model_path: Optional path to a .gguf model; defaults to env LLM_MODEL_PATH or models/tinyllama-1.1b-chat.Q4_K_M.gguf
        """
        self.base_url = base_url  # deprecated
        
        # Model configuration
        self.model_path = (
            model_path
            or os.environ.get("LLM_MODEL_PATH")
            or os.path.join("models", "tinyllama-1.1b-chat.Q4_K_M.gguf")
        )
        self.num_ctx = int(os.environ.get("LLM_NUM_CTX", 1024))
        env_threads = int(os.environ.get("LLM_NUM_THREADS", max(1, os.cpu_count() or 4)))
        self.num_threads = min(env_threads, int(os.environ.get("LLM_MAX_THREADS_CAP", 8)))
        
        # Generation configuration (concise and deterministic)
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", 0.1))
        self.top_p = float(os.environ.get("LLM_TOP_P", 0.2))
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", 20))
        self.repeat_penalty = float(os.environ.get("LLM_REPEAT_PENALTY", 1.1))
        self.stop_tokens = []
        
        # Loaded model instance
        self._llm: AutoModelForCausalLM | None = None
        
        # Simple cache (currently unused but retained for parity)
        self.response_cache = {}
        self.cache_max_size = 50
    
    def _ensure_loaded(self) -> bool:
        """Ensure the model is loaded in-process. Returns True on success."""
        if self._llm is not None:
            return True
        try:
            start = time.time()
            print(f"[LLM] Loading model: {self.model_path} (ctx={self.num_ctx}, threads={self.num_threads})")
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_type='llama',
                context_length=self.num_ctx,
                threads=self.num_threads,
                local_files_only=True
            )
            print(f"[LLM] Model loaded in {time.time()-start:.2f}s")
            return True
        except Exception as e:
            print(f"[LLM] Model load error: {e}")
            self._llm = None
            return False
    
    def send_message_async(self, message: str, on_chunk: Callable, on_complete: Callable):
        """Send message asynchronously using local model (non-streaming to avoid blocking)."""
        def _async_request():
            print(f"\n[LLM] Async request starting (ctransformers)...")
            start_time = time.time()
            try:
                if not self._ensure_loaded():
                    on_complete(f"[ERROR] Failed to load model from {self.model_path}")
                    return
                # Single-shot generation, then simulate streaming as one chunk
                text = self._llm(
                    message,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_tokens,
                    repetition_penalty=self.repeat_penalty,
                    stop=self.stop_tokens,
                    stream=False,
                )
                if text:
                    on_chunk(text)
                print(f"[LLM] Completed in {time.time()-start_time:.2f} seconds")
                on_complete(text)
            except Exception as e:
                error_msg = f"[ERROR] Local inference error: {e}"
                print(f"[LLM] {error_msg}")
                on_complete(error_msg)
        
        thread = threading.Thread(target=_async_request, daemon=True)
        thread.start()
    
    def _simulate_streaming(self, text: str, on_chunk: Callable, on_complete: Callable):
        # Retained for compatibility; not used in normal path
        print(f"[LLM] Simulating streaming for: '{text}'")
        def _stream_simulation():
            on_chunk(text)
            on_complete(text)
        threading.Thread(target=_stream_simulation, daemon=True).start()
    
    def send_message(self, message):
        """Synchronous local generation (no streaming)."""
        print(f"\n[LLM] Local completion: '{message[:60]}...' ")
        try:
            if not self._ensure_loaded():
                return f"[ERROR] Failed to load model from {self.model_path}"
            text = self._llm(
                message,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_tokens,
                repetition_penalty=self.repeat_penalty,
                stop=self.stop_tokens,
                stream=False,
            )
            print(f"[LLM] Response received: '{text}'")
            return text
        except Exception as e:
            error_msg = f"[ERROR] Local inference error: {e}"
            print(f"[LLM] {error_msg}")
            return error_msg
    
    def preload_model(self):
        """Preload the model in-process to eliminate first-request delay."""
        print(f"[LLM] Preloading local model...")
        self._ensure_loaded()
    
    def preload_model_async(self, on_complete_callback=None):
        def _preload_worker():
            print(f"[LLM] Starting async preload (local model)...")
            success = self._ensure_loaded()
            if on_complete_callback:
                on_complete_callback(success)
        thread = threading.Thread(target=_preload_worker, daemon=True)
        thread.start()
        return thread
    
    def set_model(self, model_path: str):
        """Switch to a different local model file path."""
        print(f"[LLM] Switching model to: {model_path}")
        self.model_path = model_path
        self._llm = None
        self.preload_model()
    
    def _cache_response(self, cache_key, response):
        if len(self.response_cache) >= self.cache_max_size:
            first_key = next(iter(self.response_cache))
            del self.response_cache[first_key]
        self.response_cache[cache_key] = response
        print(f"[LLM] Cached response (cache size: {len(self.response_cache)})") 