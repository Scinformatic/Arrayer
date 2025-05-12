"""Exceptions raised by Arrayer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class ArrayerError(Exception):
    """Base class for all Arrayer exceptions."""
    def __init__(
        self,
        message: str,
    ):
        super().__init__(message)
        self.message = message
        return


class InputError(ArrayerError):
    """Exception raised when an input is invalid."""

    def __init__(self, name: str, value: Any, problem: str) -> None:
        super().__init__(problem)
        self.name = name
        self.value = value
        return
