"""Frozen dimensions and optimizer settings for the v3 comparator."""

from __future__ import annotations

from dataclasses import asdict, dataclass


REFERENCE_RSSM_ACTIVE_PARAMETERS = 19_784
EXPECTED_ACTIVE_PARAMETERS = 19_789
PARENT_SCHEDULE_SCHEMA = "boptest-reliability-rssm-training-schedule-v1"
SHARED_RUNTIME_SOURCE_RELATIVE_PATHS = (
    "health_rssm/model.py",
    "health_rssm/training.py",
    "multicase_fault_benchmark/__init__.py",
    "multicase_fault_benchmark/protocol.py",
    "multicase_fault_benchmark/worker_collect.py",
    "multicase_fault_benchmark/fault_data.py",
    "multicase_fault_benchmark/reliability_loss.py",
    "multicase_fault_benchmark/reliability_model.py",
    "multicase_fault_benchmark/runtime_provenance.py",
    "multicase_fault_benchmark/study_config.py",
    "multicase_fault_benchmark/study_train.py",
)


@dataclass(frozen=True)
class DeterministicTransportConfig:
    """Protocol constants for the source-exposure-matched comparator."""

    observation_dim: int = 4
    action_dim: int = 1
    context_dim: int = 5
    hidden_dim: int = 64
    decoder_hidden_dim: int = 53

    sequence_length: int = 48
    batch_size: int = 4
    direct_horizon: int = 8
    updates: int = 400
    checkpoint_updates: tuple[int, ...] = (100, 200, 300, 400)

    learning_rate: float = 3e-4
    gradient_clip: float = 100.0
    smooth_l1_beta: float = 1.0
    observation_weight: float = 1.0
    direct_h8_weight: float = 1.0

    paired_model_seeds: tuple[int, ...] = (
        202608011,
        202608012,
        202608013,
        202608014,
        202608015,
    )

    def __post_init__(self) -> None:
        dimensions = (
            self.observation_dim,
            self.action_dim,
            self.context_dim,
            self.hidden_dim,
            self.decoder_hidden_dim,
        )
        if dimensions != (4, 1, 5, 64, 53):
            raise ValueError("the frozen deterministic model dimensions changed")
        if (
            self.sequence_length,
            self.batch_size,
            self.direct_horizon,
            self.updates,
            self.checkpoint_updates,
        ) != (48, 4, 8, 400, (100, 200, 300, 400)):
            raise ValueError("the frozen source-exposure protocol changed")
        if (
            self.learning_rate,
            self.gradient_clip,
            self.smooth_l1_beta,
            self.observation_weight,
            self.direct_h8_weight,
        ) != (3e-4, 100.0, 1.0, 1.0, 1.0):
            raise ValueError("the frozen optimizer or loss contract changed")
        if len(set(self.paired_model_seeds)) != 5:
            raise ValueError("the paired model seeds must contain five identities")

    @property
    def filter_input_dim(self) -> int:
        return (
            3 * self.observation_dim
            + self.action_dim
            + self.context_dim
        )

    @property
    def h8_sources_per_sequence(self) -> int:
        return self.sequence_length - self.direct_horizon

    @property
    def h8_endpoints_per_update(self) -> int:
        return self.batch_size * self.h8_sources_per_sequence

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FROZEN_CONFIG = DeterministicTransportConfig()
