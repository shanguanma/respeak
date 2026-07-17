"""Shared model testing helpers (transformers-style ModelTesterMixin)."""

from __future__ import annotations

import inspect
from typing import Any


class ModelTesterMixin:
    """Minimal common checks shared by all respeak model tests.

    Subclasses should set:
      - ``all_model_classes``: tuple of model classes under test
      - ``model_tester``: instance of a model-specific *ModelTester
    """

    all_model_classes: tuple[type, ...] = ()
    model_tester: Any = None

    def test_model_classes_inherit_base(self):
        from respeak.base import BaseModel

        for model_class in self.all_model_classes:
            with self.subTest(model_class=model_class.__name__):
                self.assertTrue(
                    issubclass(model_class, BaseModel),
                    f"{model_class.__name__} must subclass BaseModel",
                )

    def test_from_pretrained_is_classmethod(self):
        for model_class in self.all_model_classes:
            with self.subTest(model_class=model_class.__name__):
                self.assertTrue(callable(getattr(model_class, "from_pretrained", None)))
                self.assertIsInstance(
                    inspect.getattr_static(model_class, "from_pretrained"),
                    classmethod,
                )

    def test_generate_signature(self):
        for model_class in self.all_model_classes:
            with self.subTest(model_class=model_class.__name__):
                sig = inspect.signature(model_class.generate)
                params = list(sig.parameters)
                self.assertGreaterEqual(len(params), 2)  # self + at least one input
                self.assertIn("self", params)
