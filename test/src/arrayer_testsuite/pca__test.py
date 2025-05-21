from collections.abc import Sequence
import pytest
import yaml
import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

import arrayer
import arrayer.exception  # used dynamically by eval()

from arrayer_testsuite import data


def load_test_cases() -> dict:
    """Load golden and negative test cases from pca.yaml."""
    with open(data.filepath("pca.yaml"), 'r') as f:
        return yaml.safe_load(f)


cases = load_test_cases()


@pytest.mark.parametrize("case", cases["golden"])
def pca__golden_file__test(case):
    """Test PCA against golden file expectations."""
    output = arrayer.pca(points=jnp.array(case["input"]["points"]))
    expected_output = case["output"]

    for output_name, expected_output_value in expected_output.items():
        expected_output_value = jnp.array(expected_output_value)
        output_value = getattr(output, output_name)
        assert output_value.shape == expected_output_value.shape, f"Shape mismatch in {output_name}: expected {expected_output_value.shape}, got {output_value.shape}."
        assert jnp.allclose(output_value, expected_output_value, atol=1e-6), f"Value mismatch in {output_name}: expected {expected_output_value}, got {output_value}."
    return


@pytest.mark.parametrize("case", cases["negative"])
def pca__negative_cases__test(case):
    """Test PCA against negative cases."""
    for case in cases["negative"]:
        bad_input = jnp.array(case["input"]["points"])
        expected_error = eval(case["error"])
        expected_message = case.get("message", "")
        with pytest.raises(expected_error, match=expected_message):
            arrayer.pca(points=bad_input)
    return


def pca__batch_vs_single__test():
    """Test PCA batch vs single outputs."""
    n_samples = 100
    for batch_shape in ((5,), (6, 7), (8, 9, 10)):
        for n_features in (2, 3, 4):
            points = np.random.rand(*batch_shape, n_samples, n_features)
            output_batch = arrayer.pca(points)

            # Verify output shapes
            output_batch_shapes = get_expected_output_shapes(batch_shape, n_samples, n_features)
            for output_name, expected_shape in output_batch_shapes.items():
                value = getattr(output_batch, output_name)
                assert value.shape == expected_shape, f"Shape mismatch in {output_name}: expected {expected_shape}, got {value.shape}."

            # Verify batch vs single output
            for batch_idx in np.ndindex(*batch_shape):
                output_single = arrayer.pca(points[batch_idx])
                for output_name in output_batch_shapes.keys():
                    value_batch = getattr(output_batch, output_name)[batch_idx]
                    value_single = getattr(output_single, output_name)
                    assert jnp.allclose(value_batch, value_single, atol=1e-6), f"Value mismatch in {output_name} for batch {batch_idx}: expected {value_single}, got {value_batch}."


def pca__sklearn_comparison__test():
    n_tests = 100
    n_points = 10
    n_features_cases = (2, 3, 4)
    for _ in range(n_tests):
        for n_features in n_features_cases:
            points = np.random.rand(n_points, n_features)
            arrayer_output = arrayer.pca(points)
            sklearn_pca = SklearnPCA()
            sklearn_output = {"points": sklearn_pca.fit_transform(points)}
            sklearn_output |= {
                "components": sklearn_pca.components_,
                "singular_values": sklearn_pca.singular_values_,
                "variance_ratio": sklearn_pca.explained_variance_ratio_,
                "variance_unbiased": sklearn_pca.explained_variance_,
                "translation": sklearn_pca.mean_ * -1
            }
            expected_equal_names = ["singular_values", "variance_ratio", "variance_unbiased", "translation"]
            sklearn_is_rotation = arrayer.matrix.is_rotation(sklearn_output["components"], tol=1e-5)
            if sklearn_is_rotation:
                expected_equal_names.extend(["points", "components"])
            else:
                sklearn_components = sklearn_output["components"]
                sklearn_components[-1] *= -1
                assert jnp.allclose(arrayer_output.components, sklearn_components, atol=1e-5), f"Value mismatch in components (sklearn reflected): expected {sklearn_components}, got {arrayer_output.components} for points {points}."
            for output_name in expected_equal_names:
                arrayer_value = getattr(arrayer_output, output_name)
                sklearn_value = sklearn_output[output_name]
                assert jnp.allclose(arrayer_value, sklearn_value, atol=1e-5), f"Value mismatch in {output_name}: expected {sklearn_value}, got {arrayer_value}."


def get_expected_output_shapes(batch_shape: Sequence[int] | None, n_samples: int, n_features: int) -> dict:
    """Get expected output shapes for PCA."""
    shape_single = {
        "points": (n_samples, n_features),
        "components": (n_features, n_features),
        "singular_values": (n_features,),
        "translation": (n_features,),
        "variance_magnitude": (n_features,),
        "variance_ratio": (n_features,),
        "variance_biased": (n_features,),
        "variance_unbiased": (n_features,),
    }
    return shape_single if batch_shape is None else {k: (*batch_shape, *v) for k, v in shape_single.items()}
