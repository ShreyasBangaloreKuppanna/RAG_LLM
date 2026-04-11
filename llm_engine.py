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
