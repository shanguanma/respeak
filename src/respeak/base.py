"""Unified model interface for respeak inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """All respeak models share this load / generate surface."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> BaseModel:
        """Load weights / backends from model ids or local paths."""

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference. Call-time kwargs may override construction defaults."""
