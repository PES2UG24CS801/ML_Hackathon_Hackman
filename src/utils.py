# src/utils.py
import os
import numpy as np
from typing import List

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

def load_words(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        words = [line.strip().lower() for line in f if line.strip() and line.isalpha()]
    if not words:
        raise ValueError("No valid words found in file.")
    return words

def mask_to_str(mask: List[str]) -> str:
    return ' '.join(c if c else '_' for c in mask)

def get_guessed_vec(guessed: set) -> np.ndarray:
    return np.array([1 if c in guessed else 0 for c in ALPHABET], dtype=np.float32)