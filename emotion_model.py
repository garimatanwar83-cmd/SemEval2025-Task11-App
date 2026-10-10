from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

EMOTION_LABELS = ["anger", "fear", "joy", "sadness", "surprise"]
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "anger": [
        "angry",
        "rage",
        "mad",
        "furious",
        "hate",
        "annoyed",
        "offended",
        "resent",
        "irritated",
        "frustrated",
        "upset",
        "bitter",
        "horrible",
        "shame",
        "ashamed",
    ],
    "fear": [
        "fear",
        "afraid",
        "scared",
        "terrified",
        "panic",
        "nervous",
        "anxious",
        "worried",
        "threat",
        "danger",
        "frightened",
        "unsafe",
        "horror",
        "tremble",
    ],
    "joy": [
        "joy",
        "happy",
        "love",
        "glad",
        "excited",
        "delighted",
        "pleased",
        "cheerful",
        "smile",
        "amused",
        "wonderful",
        "great",
        "grateful",
        "good",
        "peaceful",
    ],
    "sadness": [
        "sad",
        "cry",
        "crying",
        "hurt",
        "lonely",
        "grief",
        "depressed",
        "broken",
        "sorrow",
        "disappointed",
        "ashamed",
        "shameful",
        "pain",
        "loss",
    ],
    "surprise": [
        "surprise",
        "surprised",
        "wow",
        "amazing",
        "unexpected",
        "shocked",
        "astonished",
        "startled",
        "amazed",
        "curious",
        "sudden",
    ],
}

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "task-dataset" / "semeval-2025-task11-dataset"


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())


def _matches_word(text: str, word: str) -> bool:
    return bool(re.search(rf"(?:^|\s){re.escape(word)}(?:$|\s)", text))


def predict_emotions(text: str) -> Dict[str, float]:
    scores = {label: 0.0 for label in EMOTION_LABELS}
    if not text or not text.strip():
        return scores

    normalized = normalize_text(text)
    detected = 0

    for label, words in EMOTION_KEYWORDS.items():
        matches = 0
        for word in words:
            if _matches_word(normalized, word):
                matches += 1
        scores[label] = min(1.0, matches / 3.0)
        detected += matches

    if detected == 0:
        return scores

    total = sum(scores.values())
    if total > 0:
        scores = {label: score / total for label, score in scores.items()}

    return scores


def get_available_languages() -> List[str]:
    target_dir = DATASET_ROOT / "track_a" / "train"
    if not target_dir.exists():
        return []
    return sorted(path.stem for path in target_dir.glob("*.csv"))


def load_language_split(language: str, split: str = "train") -> pd.DataFrame:
    csv_path = DATASET_ROOT / "track_a" / split / f"{language}.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["id", "text", "anger", "fear", "joy", "sadness", "surprise"])
    return pd.read_csv(csv_path)


def dataset_summary() -> Dict[str, int | List[str]]:
    languages = get_available_languages()
    total_rows = 0
    for language in languages:
        df = load_language_split(language, split="train")
        total_rows += len(df)

    return {
        "languages": languages,
        "language_count": len(languages),
        "total_examples": total_rows,
        "labels": EMOTION_LABELS,
    }


def sample_rows(language: str, n: int = 5) -> pd.DataFrame:
    train_df = load_language_split(language, split="train")
    if not train_df.empty:
        return train_df.head(n)

    dev_df = load_language_split(language, split="dev")
    return dev_df.head(n)
