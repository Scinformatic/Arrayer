"""Tests for tensor operations and properties."""

import yaml

import jax.numpy as jnp
import pytest
import jaxtyping as jaxtyping  # used dynamically by eval()

from arrayer import tensor

from arrayer_testsuite import data


with open(data.filepath("tensor.yaml"), 'r') as f:
    test_data = yaml.safe_load(f)


@pytest.mark.parametrize(
    "case",
    test_data["argin"]["golden"],
    ids=[test_case["case"] for test_case in test_data["argin"]["golden"]]
)
def argin__test(case: dict) -> None:
    """Test `arrayer.tensor.argin` function against golden file expectations."""
    expected = jnp.array(case["expected"])
    result = tensor.argin(**case["input"])
    assert jnp.all(result == expected), f"Failed case: {case['case']}"


@pytest.mark.parametrize(
    "case",
    test_data["indices_sorted_by_value"]["golden"],
    ids=[test_case["case"] for test_case in test_data["indices_sorted_by_value"]["golden"]]
)
def indices_sorted_by_value__test(case: dict) -> None:
    """Test `arrayer.tensor.indices_sorted_by_value` function against golden file expectations."""
    result = tensor.indices_sorted_by_value(**case["input"])
    expected = case["expected"]
    if expected:
        expected = jnp.array(expected)
    else:
        input_tensor = jnp.array(case["input"]["tensor"])
        expected = jnp.stack(jnp.unravel_index(jnp.array([]), input_tensor.shape), axis=-1)
    assert jnp.array_equal(result, expected), f"Failed case: {case['case']}"
