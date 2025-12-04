"""Model architectures for ICD code prediction."""

from .caml import CAML
from .led_classifier import LEDClassifier

__all__ = [
    "CAML",
    "LEDClassifier",
]
