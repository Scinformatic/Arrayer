"""Exceptions raised by Arrayer."""

from __future__ import annotations


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

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.name = name
        return
