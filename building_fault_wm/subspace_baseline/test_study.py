from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from nfoursid.nfoursid import NFourSID
from threadpoolctl import threadpool_limits

from . import study


class SubspaceStudyTest(unittest.TestCase):
    def test_block_hankel_keeps_sequence_order(self) -> None:
        values = np.arange(8, dtype=float).reshape(4, 2)
        expected = np.array(
            [[0, 2, 4], [1, 3, 5], [2, 4, 6], [3, 5, 7]], dtype=float
        )
        np.testing.assert_array_equal(study._block_hankel(values, 2), expected)

    def test_model_round_trip(self) -> None:
        order = 2
        model = study.SubspaceModel(
            block_rows=8,
            state_order=order,
            innovation_clip_sigma=3.0,
            a=np.eye(order) * 0.8,
            b=np.zeros((order, 6)),
            c=np.ones((4, order)),
            d=np.zeros((4, 6)),
            covariance=np.eye(order + 4),
            singular_values=np.arange(8, dtype=float),
        )
        restored = study.restore_model(model.payload())
        np.testing.assert_array_equal(restored.a, model.a)
        self.assertAlmostEqual(restored.spectral_radius, 0.8)

    def test_open_loop_uses_causal_input(self) -> None:
        model = study.SubspaceModel(
            block_rows=8,
            state_order=1,
            innovation_clip_sigma=0.0,
            a=np.array([[1.0]]),
            b=np.ones((1, 6)),
            c=np.ones((4, 1)),
            d=np.zeros((4, 6)),
            covariance=np.eye(5),
            singular_values=np.ones(4),
        )
        inputs = np.ones((8, 6))
        prediction = study.open_loop_prediction(
            model, np.zeros(1), inputs, anchor=0, horizon=2
        )
        np.testing.assert_array_equal(prediction, np.full(4, 6.0))

    def test_single_experiment_matches_reference_implementation(self) -> None:
        generator = np.random.default_rng(4)
        steps = 120
        inputs = generator.normal(size=(steps, 6))
        outputs = np.zeros((steps, 4))
        transition = np.diag([0.8, 0.7, 0.6, 0.5])
        input_map = generator.normal(scale=0.1, size=(4, 6))
        for step in range(steps - 1):
            outputs[step + 1] = (
                transition @ outputs[step]
                + input_map @ inputs[step]
                + generator.normal(scale=0.01, size=4)
            )
        columns = [*(f"u{i}" for i in range(6)), *(f"y{i}" for i in range(4))]
        frame = pd.DataFrame(np.concatenate([inputs, outputs], axis=1), columns=columns)
        reference = NFourSID(
            frame,
            [f"y{i}" for i in range(4)],
            [f"u{i}" for i in range(6)],
            num_block_rows=8,
        )
        reference.subspace_identification()
        state_space, covariance = reference.system_identification(rank=4)
        candidate = study.identify_models([(inputs, outputs)], 8)[4]
        np.testing.assert_allclose(candidate.a, state_space.a, atol=1e-10, rtol=0)
        np.testing.assert_allclose(candidate.b, state_space.b, atol=1e-10, rtol=0)
        np.testing.assert_allclose(candidate.c, state_space.c, atol=1e-10, rtol=0)
        np.testing.assert_allclose(candidate.d, state_space.d, atol=1e-10, rtol=0)
        reordered = np.block(
            [
                [covariance[4:, 4:], covariance[4:, :4]],
                [covariance[:4, 4:], covariance[:4, :4]],
            ]
        )
        np.testing.assert_allclose(
            candidate.covariance, reordered, atol=1e-10, rtol=0
        )

    def test_identification_is_stable_across_outer_blas_limits(self) -> None:
        generator = np.random.default_rng(19)
        sequences = []
        for _ in range(3):
            inputs = generator.normal(size=(80, 6))
            outputs = generator.normal(size=(80, 4))
            sequences.append((inputs, outputs))
        with threadpool_limits(limits=1, user_api="blas"):
            single = study.identify_models(sequences, 8)[4]
        with threadpool_limits(limits=4, user_api="blas"):
            multiple = study.identify_models(sequences, 8)[4]
        for left, right in (
            (single.a, multiple.a),
            (single.b, multiple.b),
            (single.c, multiple.c),
            (single.d, multiple.d),
            (single.covariance, multiple.covariance),
        ):
            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
