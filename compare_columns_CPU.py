from __future__ import annotations

import datetime
import json
import re
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# Configuration (CPU-only)
# =========================
# You can swap to:
#   "Qwen/Qwen2.5-1.5B-Instruct"  (faster on CPU)
#   "microsoft/Phi-4-mini-instruct" (compact)
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

CPU_THREADS = 4
MAX_NEW_TOKENS = 64  # small for strict JSON tasks

try:
    torch.set_num_threads(CPU_THREADS)
except Exception:
    pass

_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    torch_dtype=torch.float32,  # safer on CPU
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval()


# =========================
# Prompt templates (JSON)
# =========================
SYSTEM_PROMPT = (
    "You are a precise data engineer. Return ONLY a single valid JSON object. "
    "No code fences, no prose, no trailing text.\n"
    'Schema: {"source_col": string|null, "target_col": string|null}'
)

USER_SUFFIX = (
    "\n\nReturn ONLY valid JSON (no markdown, no code fences, no explanations). "
    'Output must match exactly: {"source_col": <string or null>, "target_col": <string or null>}.'
)


# =========================
# LLM call (deterministic)
# =========================
def _call_llm_json(user_prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> Tuple[str, str]:
    """
    Returns (raw_text, prompt_string). Raises on generation errors.
    Decodes only the newly generated tokens to avoid JSON bleed from the prompt.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt + USER_SUFFIX},
    ]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,         # deterministic JSON
            do_sample=False,
            repetition_penalty=1.05,
        )

    # Decode only generated tokens after the prompt
    gen_ids = output[0, inputs["input_ids"].shape[-1]:]
    text = _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, prompt


# =========================
# JSON parsing / sanitizing
# =========================
def _extract_json_block(text: str) -> Optional[str]:
    # 1) Direct
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # 2) Strip code fences quickly
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        json.loads(t)
        return t
    except Exception:
        pass

    # 3) First balanced {...}
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return None


def _extract_last_json_block(text: str) -> Optional[str]:
    """Fallback: extract the last balanced JSON block in the text."""
    t = text.strip()
    end = t.rfind("}")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        if t[i] == "}":
            depth += 1
        elif t[i] == "{":
            depth -= 1
            if depth == 0:
                block = t[i:end + 1]
                try:
                    json.loads(block)
                    return block
                except Exception:
                    return None
    return None


def _sanitize_and_parse_json(maybe_json: str) -> Optional[dict]:
    s = maybe_json.strip()
    # Python → JSON
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)

    # single-quoted keys/values → double quotes (safe/simple cases)
    s = re.sub(r"\{(\s*)'([^']+?)'(\s*):", r'{\1"\2"\3:', s)
    s = re.sub(r",(\s*)'([^']+?)'(\s*):", r',\1"\2"\3:', s)
    s = re.sub(r':(\s*)\'([^\'\n]*?)\'(\s*)([,}])', r':"\2"\3\4', s)

    try:
        return json.loads(s)
    except Exception:
        return None


def parse_llm_json(raw_text: str) -> Optional[dict]:
    block = _extract_json_block(raw_text)
    if block is None:
        block = _extract_last_json_block(raw_text)
        if block is None:
            return None
    try:
        return json.loads(block)
    except Exception:
        return _sanitize_and_parse_json(block)


# =========================
# Data helpers (validation)
# =========================
def _sample_column_values(df: pd.DataFrame, k: int = 10) -> Dict[str, List[str]]:
    """
    Compact per-column samples for prompting.
    Uses first-k unique stringified non-null values to stay deterministic & small.
    """
    samples: Dict[str, List[str]] = {}
    for c in df.columns:
        vals = df[c].dropna().astype(str).unique().tolist()
        samples[c] = vals[:k]
    return samples


def _normalize_series(s: pd.Series, numeric_round: int = 3) -> pd.Series:
    s = s.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() >= 0.7:
        s = s_num.round(numeric_round).astype(str).replace("nan", np.nan)
    return s.dropna()


def _value_set(s: pd.Series, cap: int = 100_000) -> set:
    v = _normalize_series(s)
    u = set(v.unique().tolist())
    if len(u) > cap:
        # simple cap truncation
        u = set(list(u)[:cap])
    return u


def _jaccard_and_containment(a: set, b: set) -> Tuple[float, float]:
    if not a and not b:
        return 0.0, 0.0
    inter = len(a & b)
    union = len(a | b) if (a or b) else 1
    jac = inter / union if union else 0.0
    cont = 0.0
    if a:
        cont = max(cont, inter / len(a))
    if b:
        cont = max(cont, inter / len(b))
    return jac, cont


def _uniqueness_ratio(col: pd.Series) -> float:
    col = col.dropna().astype(str)
    n = len(col)
    return float(col.nunique(dropna=True) / n) if n else 0.0


def _resolve_col(name: Optional[str], cols: pd.Index) -> Optional[str]:
    """Case-insensitive, trimmed resolution of a column name within given columns."""
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    for c in cols:
        if c.strip().lower() == key:
            return c
    return None


def _score_pair(df_src: pd.DataFrame, df_tgt: pd.DataFrame, src_col: str, tgt_col: str) -> Tuple[float, dict]:
    src_u = _uniqueness_ratio(df_src[src_col])
    tgt_u = _uniqueness_ratio(df_tgt[tgt_col])
    a = _value_set(df_src[src_col])
    b = _value_set(df_tgt[tgt_col])
    jacc, cont = _jaccard_and_containment(a, b)
    confidence = float(0.35 * src_u + 0.35 * tgt_u + 0.20 * jacc + 0.10 * cont)
    metrics = {"src_uniqueness": src_u, "tgt_uniqueness": tgt_u, "jaccard": jacc, "containment": cont}
    return confidence, metrics


def _deterministic_fallback(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    min_confidence: float,
) -> Dict[str, Optional[str]]:
    """
    Heuristic fallback if the LLM abstains or parsing fails.
    1) Try identical column names first, pick best by confidence.
    2) If still nothing, try all pairs (capped by simple heuristics).
    """
    best = {"source_col": None, "target_col": None, "confidence": 0.0, "metrics": {}}

    # 1) Same-name columns
    common = [c for c in df_src.columns if c in df_tgt.columns]
    for c in common:
        conf, metrics = _score_pair(df_src, df_tgt, c, c)
        if conf > best["confidence"]:
            best = {"source_col": c, "target_col": c, "confidence": conf, "metrics": metrics}

    if best["confidence"] >= min_confidence:
        return best

    # 2) All pairs (small datasets or small column count)
    # To avoid O(n^2) blowups on very wide tables, cap by a simple rule:
    if len(df_src.columns) * len(df_tgt.columns) <= 400:  # e.g., <= 20x20 columns
        for s in df_src.columns:
            for t in df_tgt.columns:
                conf, metrics = _score_pair(df_src, df_tgt, s, t)
                if conf > best["confidence"]:
                    best = {"source_col": s, "target_col": t, "confidence": conf, "metrics": metrics}

    if best["confidence"] >= min_confidence:
        return best

    # No confident guess
    best["source_col"] = None
    best["target_col"] = None
    return best


# =========================
# Public function
# =========================
def find_id_column_mapping(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    sample_k: int = 10,
    min_confidence: float = 0.60,     # accept mapping if >= this score
    reprompt_on_parse_fail: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Ask the LLM to propose matching ID/Key columns, then validate on data and return a confidence.
    Returns:
        {
          "source_col": str|None,
          "target_col": str|None,
          "confidence": float,
          "metrics": { "src_uniqueness": float, "tgt_uniqueness": float, "jaccard": float, "containment": float },
          "raw_text": str  # raw model output (debug)
        }
    """

    # 1) Build compact samples (for prompt)
    src_samples = _sample_column_values(df_src, k=sample_k)
    tgt_samples = _sample_column_values(df_tgt, k=sample_k)

    user = (
        "Find the primary ID/Key columns that match between these two tables.\n"
        f"Source Columns & Samples: {src_samples}\n"
        f"Target Columns & Samples: {tgt_samples}\n"
        'If no clear match, return {"source_col": null, "target_col": null}.'
    )

    # 2) Call LLM (1st attempt)
    raw_text, _ = _call_llm_json(user)
    data = parse_llm_json(raw_text)

    # 3) Optional reprompt if parsing failed
    if data is None and reprompt_on_parse_fail:
        user2 = user + "\n\nYour previous response was not valid JSON. Return ONLY a single JSON object as specified."
        raw_text, _ = _call_llm_json(user2)
        data = parse_llm_json(raw_text)

    # Safe default if still not a dict
    if not isinstance(data, dict):
        # Deterministic fallback attempt
        fb = _deterministic_fallback(df_src, df_tgt, min_confidence=min_confidence)
        if fb["source_col"] is None or fb["target_col"] is None:
            return {
                "source_col": None,
                "target_col": None,
                "confidence": 0.0,
                "metrics": {"src_uniqueness": 0.0, "tgt_uniqueness": 0.0, "jaccard": 0.0, "containment": 0.0},
                "raw_text": raw_text,
            }
        else:
            return {
                "source_col": fb["source_col"],
                "target_col": fb["target_col"],
                "confidence": fb["confidence"],
                "metrics": fb["metrics"],
                "raw_text": raw_text,
            }

    # 4) Schema guard (strict keys + case-insensitive resolution)
    # Keep only required keys if model added extras
    data = {k: data.get(k) for k in ("source_col", "target_col")}
    src_col = _resolve_col(data.get("source_col"), df_src.columns)
    tgt_col = _resolve_col(data.get("target_col"), df_tgt.columns)

    # If LLM abstains or name not in columns → fallback
    if (src_col is None) or (tgt_col is None):
        fb = _deterministic_fallback(df_src, df_tgt, min_confidence=min_confidence)
        if fb["source_col"] is None or fb["target_col"] is None:
            return {
                "source_col": None,
                "target_col": None,
                "confidence": 0.0,
                "metrics": {"src_uniqueness": 0.0, "tgt_uniqueness": 0.0, "jaccard": 0.0, "containment": 0.0},
                "raw_text": raw_text,
            }
        else:
            return {
                "source_col": fb["source_col"],
                "target_col": fb["target_col"],
                "confidence": fb["confidence"],
                "metrics": fb["metrics"],
                "raw_text": raw_text,
            }

    # 5) Validate candidates on data and compute confidence
    confidence, metrics = _score_pair(df_src, df_tgt, src_col, tgt_col)

    # 6) Thresholding
    if confidence < min_confidence:
        # Low confidence: attempt deterministic fallback; else abstain
        fb = _deterministic_fallback(df_src, df_tgt, min_confidence=min_confidence)
        if fb["source_col"] is None or fb["target_col"] is None:
            return {
                "source_col": None,
                "target_col": None,
                "confidence": confidence,
                "metrics": metrics,
                "raw_text": raw_text,
            }
        else:
            return {
                "source_col": fb["source_col"],
                "target_col": fb["target_col"],
                "confidence": fb["confidence"],
                "metrics": fb["metrics"],
                "raw_text": raw_text,
            }

    # 7) Success
    return {
        "source_col": src_col,
        "target_col": tgt_col,
        "confidence": confidence,
        "metrics": metrics,
        "raw_text": raw_text,
    }


# ----- Quick self-test (optional) -----
if __name__ == "__main__":
    start= datetime.datetime.now()
    # Tiny demo tables
    # df_a = pd.DataFrame({
    #     "breaker_id": ["BR-001", "BR-002", "BR-003", None],
    #     "year": [2020, 2021, 2022, 2023],
    #     "notes": ["a", "b", "c", "d"]
    # })
    # df_b = pd.DataFrame({
    #     "leistungsschalter_id": ["BR-001", "BR-002", "BR-003", "BR-004"],
    #     "baujahr": [2020, 2021, 2022, 2023],
    #     "bemerkung": ["x", "y", "z", "w"]
    # })

    # Example with your CSVs (adjust paths/encoding as needed):
    df_a = pd.read_csv('data_2024_2/Bestandsdaten/Hochspannung/110kVLeistungsschalter.csv', delimiter=';')
    df_b = pd.read_csv('data_2025/Hochspannung/110kVLeistungsschalter.csv', encoding="ISO-8859-1", delimiter=';')

    mapping = find_id_column_mapping(df_a, df_b)
    print(json.dumps(mapping, indent=2))
    print('time taken =', datetime.datetime.now() - start )
    pass