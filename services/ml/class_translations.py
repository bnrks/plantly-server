"""Utilities for converting model class labels to user-friendly Turkish strings.

The model outputs labels from `ml/classes/classes.json` (e.g. `Apple__Apple_Scab`).
We keep a stable mapping here so both chat (LLM prompts) and persistence can use
consistent, user-facing names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Exact labels present in `ml/classes/classes.json`.
# Keep these user-facing strings Turkish and reasonably short.
CLASS_TR: dict[str, str] = {
    "Apple__Apple_Scab": "Elma - Karalekesi",
    "Apple__Black_Rot": "Elma - Kara çürüklük",
    "Apple__Cedar_Apple_Rust": "Elma - Pas (sedir-elma pası)",
    "Apple__Healthy": "Elma - Sağlıklı",
    "Cherry__Healthy": "Kiraz - Sağlıklı",
    "Cherry__Powdery_Mildew": "Kiraz - Külleme",
    "Corn__Common_Rust": "Mısır - Pas (yaygın pas)",
    "Corn__Gray_Leaf_Spot": "Mısır - Gri yaprak lekesi",
    "Corn__Healthy": "Mısır - Sağlıklı",
    "Corn__Northern_Leaf_Blight": "Mısır - Yaprak yanıklığı (kuzey)",
    "Grape__Black_Rot": "Üzüm - Kara çürüklük",
    "Grape__Esca": "Üzüm - Esca",
    "Grape__Healthy": "Üzüm - Sağlıklı",
    "Grape__Leaf_Blight": "Üzüm - Yaprak yanıklığı",
    "Peach__Bacterial_Spot": "Şeftali - Bakteriyel leke",
    "Peach__Healthy": "Şeftali - Sağlıklı",
    "Pepper__Bacterial_Spot": "Biber - Bakteriyel leke",
    "Pepper__Healthy": "Biber - Sağlıklı",
    "Potato__Early_Blight": "Patates - Erken yanıklık",
    "Potato__Healthy": "Patates - Sağlıklı",
    "Potato__Late_Blight": "Patates - Geç yanıklık",
    "Strawberry__Healthy": "Çilek - Sağlıklı",
    "Strawberry__Leaf_Scorch": "Çilek - Yaprak kavrulması",
    "Tomato__Bacterial_Spot": "Domates - Bakteriyel leke",
    "Tomato__Early_Blight": "Domates - Erken yanıklık",
    "Tomato__Healthy": "Domates - Sağlıklı",
    "Tomato__Late_Blight": "Domates - Geç yanıklık",
}


@dataclass(frozen=True)
class ParsedLabel:
    plant: str
    condition: str


def parse_label(label: str) -> Optional[ParsedLabel]:
    """Parse `Plant__Condition` labels. Returns None if format is unknown."""
    if not label:
        return None
    if "__" not in label:
        return None
    plant, condition = label.split("__", 1)
    if not plant or not condition:
        return None
    return ParsedLabel(plant=plant, condition=condition)


def to_tr_label(label: str) -> str:
    """Convert a raw model label into a Turkish user-facing string."""
    if not label:
        return "Bilinmeyen teşhis"

    direct = CLASS_TR.get(label)
    if direct:
        return direct

    # Safe fallback: avoid exposing raw class label to end user.
    parsed = parse_label(label)
    if parsed is None:
        return "Bitki sorunu"

    plant_tr = {
        "Apple": "Elma",
        "Cherry": "Kiraz",
        "Corn": "Mısır",
        "Grape": "Üzüm",
        "Peach": "Şeftali",
        "Pepper": "Biber",
        "Potato": "Patates",
        "Strawberry": "Çilek",
        "Tomato": "Domates",
    }.get(parsed.plant, "Bitki")

    # Generic condition fallback
    if parsed.condition.lower() == "healthy":
        return f"{plant_tr} - Sağlıklı"

    return f"{plant_tr} - Hastalık belirtisi"
