"""Tests for matrix operations and properties."""

import yaml
from pathlib import Path

import jax.numpy as jnp
import pytest

import arrayer
from arrayer.matrix import is_rotation, is_orthogonal, has_unit_determinant

from arrayer_testsuite import data


def load_test_cases() -> dict:
    """Load golden and negative test cases from matrix.yaml."""
    with open(data.filepath("matrix.yaml"), 'r') as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize(
    "function, key",
    [
        (is_rotation, "is_rotation"),
        (is_orthogonal, "is_orthogonal"),
        (has_unit_determinant, "has_unit_determinant"),
    ],
    ids=["is_rotation", "is_orthogonal", "has_unit_determinant"]
)
def matrix_properties__golden_file__test(function, key: str) -> None:
    """Test matrix property functions against golden file expectations."""
    data = load_test_cases()[key]["golden"]

    for case in data:
        input_matrices = jnp.array(case["input"])
        expected = case["expected"]

        result = function(input_matrices)

        if isinstance(expected, list):
            # Batched result
            expected_array = jnp.array(expected, dtype=bool)
            assert jnp.all(result == expected_array), f"Failed case: {case['case']}"
        else:
            # Single matrix result
            assert result is expected, f"Failed case: {case['case']}"


@pytest.mark.parametrize(
    "func, key",
    [
        (is_rotation, "is_rotation"),
        (is_orthogonal, "is_orthogonal"),
        (has_unit_determinant, "has_unit_determinant"),
    ],
    ids=["is_rotation", "is_orthogonal", "has_unit_determinant"]
)
def matrix_properties__negative_cases__test(func, key: str) -> None:
    """Test functions raise expected exceptions on invalid input."""
    data = load_test_cases()[key]["negative"]

    for case in data:
        bad_input = jnp.array(case["input"])
        expected_error = eval(case["error"])  # Use safe eval if dynamic imports are a concern
        expected_message = case.get("message", "")

        with pytest.raises(expected_error, match=expected_message):
            func(bad_input)
