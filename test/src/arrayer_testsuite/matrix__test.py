"""Tests for matrix operations and properties."""

import yaml

import jax.numpy as jnp
import pytest
import jaxtyping as jaxtyping  # used dynamically by eval()

from arrayer import matrix

from arrayer_testsuite import data

function_names = [
    "is_rotation",
    "is_orthogonal",
    "has_unit_determinant",
    "linearly_dependent_pairs",
]
func_key_pairs = [(getattr(matrix, name), name) for name in function_names]


def load_test_cases() -> dict:
    """Load golden and negative test cases from matrix.yaml."""
    with open(data.filepath("matrix.yaml"), 'r') as f:
        return yaml.safe_load(f)


cases = load_test_cases()


@pytest.mark.parametrize("function, key", func_key_pairs, ids=function_names)
def matrix_properties__golden_file__test(function, key: str) -> None:
    """Test matrix property functions against golden file expectations."""
    for case in cases[key]["golden"]:
        input_matrices = jnp.array(case["input"]["matrix"])
        expected = case["expected"]
        result = function(matrix=input_matrices)
        expected_array = jnp.array(expected)
        assert jnp.all(result == expected_array), f"Failed case: {case['case']}"
    return


@pytest.mark.parametrize("function, key", func_key_pairs, ids=function_names)
def matrix_properties__negative_cases__test(function, key: str) -> None:
    """Test functions raise expected exceptions on invalid input."""
    for case in cases[key]["negative"]:
        bad_input = jnp.array(case["input"]["matrix"])
        expected_error = eval(case["error"])
        expected_message = case.get("message", "")
        with pytest.raises(expected_error, match=expected_message):
            function(matrix=bad_input)
    return
