"""Preprocessing utilities for text in resumes and job descriptions."""

import re
from typing import List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def normalize_text(text: str) -> str:
    """Lowercase, remove special characters and stopwords.

    This is intentionally simple and easy to read, not a perfect NLP pipeline.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    # keep only letters / numbers / spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens: List[str] = []
    for tok in text.split():
        if tok in ENGLISH_STOP_WORDS:
            continue
        tokens.append(tok)
    return " ".join(tokens)


def extract_years_of_experience(text: str) -> int:
    """Very small heuristic to pull out years of experience from a sentence.

    Examples it can catch:
    - "2+ years of experience"
    - "3 years experience"
    If nothing is found we simply return 0.
    """
    if not isinstance(text, str):
        text = str(text)

    pattern = r"(\d+)\s*\+?\s*years?"
    match = re.search(pattern, text.lower())
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return 0
    return 0
