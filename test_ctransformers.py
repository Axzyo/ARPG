from ctransformers import AutoModelForCausalLM
import time

model_path = "models/tinyllama-1.1b-chat.Q4_K_M.gguf"
print("loading...")
start = time.time()
m = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama", local_files_only=True, threads=6)
print(f"loaded in {time.time()-start:.2f}s")

prompts = [
    "Hello",
    "You are Bob. Player: hello\nBob:",
    "Respond briefly: hello",
]

for i, prompt in enumerate(prompts, 1):
    print(f"\nCase {i}: {prompt!r}")
    s = time.time()
    out = m(prompt, max_new_tokens=64, temperature=0.8, top_p=0.95)
    print("sync:", repr(out))
    print(f"took {time.time()-s:.2f}s")
    print("stream:", end=" ")
    s = time.time()
    chunks = []
    for tok in m(prompt, max_new_tokens=64, temperature=0.8, top_p=0.95, stream=True):
        print(tok, end="", flush=True)
        chunks.append(tok)
    print()
    print(f"took {time.time()-s:.2f}s; chunks={len(chunks)}") 