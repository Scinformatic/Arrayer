import pytest
import yaml
import jax.numpy as jnp

from arrayer import pca

from arrayer_testsuite import data


def load_test_cases() -> dict:
    """Load golden and negative test cases from pca.yaml."""
    with open(data.filepath("pca.yaml"), 'r') as f:
        return yaml.safe_load(f)["cases"]


@pytest.mark.parametrize("case", load_test_cases())
def test_pca__golden_file(case):
    points = jnp.array(case["input"])
    variance_type = case.get("variance_type", "unbiased")

    P, variance, t, transformed = pca(points, variance_type)

    expected = case["expected"]
    expected_P = jnp.array(expected["P"])
    expected_variance = jnp.array(expected["variance"])
    expected_t = jnp.array(expected["t"])
    expected_transformed = jnp.array(expected["transformed"])

    # Shape assertions
    assert P.shape == expected_P.shape, f"Shape mismatch in P: got {P.shape}, expected {expected_P.shape}"
    assert variance.shape == expected_variance.shape, f"Shape mismatch in variance: got {variance.shape}, expected {expected_variance.shape}"
    assert t.shape == expected_t.shape, f"Shape mismatch in t: got {t.shape}, expected {expected_t.shape}"
    assert transformed.shape == expected_transformed.shape, f"Shape mismatch in transformed: got {transformed.shape}, expected {expected_transformed.shape}"

    # Value comparisons
    assert jnp.allclose(P, expected_P, atol=1e-6), f"Failed case {case['case']}: P"
    assert jnp.allclose(variance, expected_variance, atol=1e-6), f"Failed case {case['case']}: variance"
    assert jnp.allclose(t, expected_t, atol=1e-6), f"Failed case {case['case']}: t"
    assert jnp.allclose(transformed, expected_transformed, atol=1e-6), f"Failed case {case['case']}: transformed"
    return
