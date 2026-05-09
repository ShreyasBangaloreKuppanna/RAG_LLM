# llm_engine.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Registry of available LLMs
LLM_REGISTRY = {
    "qwen-3b": {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "tokenizer": None,
        "model": None,
    },
    "phi-3-mini": {
        "id": "microsoft/phi-3-mini-4k-instruct",
        "tokenizer": None,
        "model": None,
    }
}


def load_llm(model_key: str):
    """Load the specified LLM into memory (CPU)."""
    if model_key not in LLM_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}")

    entry = LLM_REGISTRY[model_key]
    if entry["model"] is None:
        print(f"Loading model {entry['id']} on CPU...")
        entry["tokenizer"] = AutoTokenizer.from_pretrained(entry["id"])
        entry["model"] = AutoModelForCausalLM.from_pretrained(
            entry["id"],
            torch_dtype=torch.float32,
            device_map="cpu"
        )
    return entry


def run_llm(prompt: str, model_key: str = "qwen-3b") -> str:
    """Run inference on the selected LLM."""
    entry = load_llm(model_key)
    tokenizer = entry["tokenizer"]
    model = entry["model"]

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def list_llms():
    """Return available model keys for UI dropdowns."""
    return list(LLM_REGISTRY.keys())


# llm_engine.py

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from threading import Lock
#
# # -------- GLOBAL LOCK (thread-safe loading) --------
# _model_lock = Lock()
#
# # -------- MODEL REGISTRY --------
# LLM_REGISTRY = {
#     "qwen-3b": {
#         "id": "Qwen/Qwen2.5-3B-Instruct",
#         "tokenizer": None,
#         "model": None,
#     },
#     "phi-3-mini": {
#         "id": "microsoft/phi-3-mini-4k-instruct",
#         "tokenizer": None,
#         "model": None,
#     }
# }
#
# # -------- LOAD MODEL (CPU ONLY) --------
# def load_llm(model_key: str):
#     if model_key not in LLM_REGISTRY:
#         raise ValueError(f"Unknown model key: {model_key}")
#
#     entry = LLM_REGISTRY[model_key]
#
#     with _model_lock:
#         if entry["model"] is None:
#             print(f"[LLM] Loading model: {model_key}")
#
#             tokenizer = AutoTokenizer.from_pretrained(entry["id"])
#             model = AutoModelForCausalLM.from_pretrained(
#                 entry["id"],
#                 torch_dtype=torch.float32,   # ✅ CPU friendly
#                 device_map="cpu"
#             )
#
#             model.eval()
#
#             entry["tokenizer"] = tokenizer
#             entry["model"] = model
#
#     return entry
#
# # -------- MAIN INFERENCE --------
# def run_llm(prompt: str, model_key: str = "qwen-3b", log_fn=None) -> str:
#     """
#     Runs inference with optional logging hook
#     """
#
#     entry = load_llm(model_key)
#     tokenizer = entry["tokenizer"]
#     model = entry["model"]
#
#     # ✅ Structured SQL Prompt Enforcement
#     system_prefix = """You are a PostgreSQL expert.
#
# Rules:
# - Only output SQL queries
# - No explanations
# - Only SELECT or WITH statements
# - Ensure syntax is valid PostgreSQL
# - Always include LIMIT 100 unless explicitly specified
# """
#
#     full_prompt = system_prefix + "\n\nUser Request:\n" + prompt
#
#     if log_fn:
#         log_fn("[LLM] Sending prompt...")
#         log_fn(full_prompt[:500])  # avoid flooding logs
#
#     inputs = tokenizer(full_prompt, return_tensors="pt")
#
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=512,
#             temperature=0.2,     # ✅ stable SQL
#             do_sample=False      # ✅ deterministic
#         )
#
#     result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#     # ✅ Extract only generated portion
#     response = result[len(full_prompt):].strip()
#
#     if log_fn:
#         log_fn("[LLM] Response received:")
#         log_fn(response[:500])
#
#     return response
#
#
# # -------- LIST MODELS --------
# def list_llms():
#     return list(LLM_REGISTRY.keys())