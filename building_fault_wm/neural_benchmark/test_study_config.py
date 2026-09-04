from dataclasses import replace

import pytest

from .study_config import ARMS, StudyConfig


def test_arm_factorization_and_binary_reliability_contract():
    config = StudyConfig()
    assert config.model_config().sensor_observation_indices == (0, 1)
    assert config.model_config().health_classes == 2
    assert config.model_config().action_dim == 1
    assert config.gru_batch_size == config.batch_size * (
        config.sequence_length - config.direct_horizon
    )
    expected = {
        "legacy": ("bypass", 0.0, 0.0),
        "ungated_h8": ("bypass", 0.0, 1.0),
        "aux_h8": ("bypass", 0.25, 1.0),
        "gated_h8": ("learned", 0.25, 1.0),
        "huber_h8": ("huber", 0.0, 1.0),
    }
    for arm in ARMS:
        arm_config = config.arm_config(arm)
        loss = config.loss_config(arm)
        assert (
            arm_config.gate_mode,
            loss.health_weight,
            loss.direct_horizon_weight,
        ) == expected[arm]
        assert loss.overshooting_weight == 0.1
        assert loss.direct_horizon == 8


def test_config_rejects_post_hoc_update_and_seed_changes():
    config = StudyConfig()
    with pytest.raises(ValueError, match="checkpoint"):
        replace(config, updates=500)
    with pytest.raises(ValueError, match="first three"):
        replace(config, development_seeds=config.confirmatory_seeds[1:4])
    with pytest.raises(ValueError, match="distinct"):
        replace(config, schedule_seed=config.development_seeds[0])
    with pytest.raises(ValueError, match="GRU exposure"):
        replace(config, gru_batch_size=config.gru_batch_size - 1)
