"""Fixed contract for the strengthened Ridge-ARX robustness analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CASES = (
    "bestest_hydronic_heat_pump",
    "multizone_office_simple_air",
    "twozone_apartment_hydronic",
)
POLICIES = ("old_2h", "new_4h")
SILENT_FAMILIES = ("bias", "drift", "stuck")
MODEL_SEEDS = (
    202608011,
    202608012,
    202608013,
    202608014,
    202608015,
)


@dataclass(frozen=True)
class RobustnessConfig:
    """Immutable selection, exposure, rollout, and analysis contract."""

    observation_dim: int = 4
    action_dim: int = 1
    context_dim: int = 5
    histories: tuple[int, ...] = (4, 8, 16, 24, 40)
    alphas: tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        1_000.0,
        10_000.0,
        100_000.0,
    )
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    sequence_length: int = 48
    source_history: int = 8
    scheduled_sources_per_sequence: int = 40
    batch_size: int = 4
    updates: int = 400
    bootstrap_draws: int = 10_000
    bootstrap_seed: int = 202608029

    def __post_init__(self) -> None:
        if self.histories != (4, 8, 16, 24, 40):
            raise ValueError("strengthened ARX history grid changed")
        if self.alphas != tuple(10.0**power for power in range(-4, 6)):
            raise ValueError("strengthened ARX alpha grid changed")
        if self.horizons != (1, 2, 4, 8):
            raise ValueError("strengthened ARX horizon grid changed")
        if (
            self.sequence_length,
            self.source_history,
            self.scheduled_sources_per_sequence,
            self.batch_size,
            self.updates,
        ) != (48, 8, 40, 4, 400):
            raise ValueError("strengthened ARX exposure contract changed")
        if self.bootstrap_draws != 10_000 or self.bootstrap_seed != 202608029:
            raise ValueError("strengthened ARX bootstrap contract changed")

    @property
    def max_history(self) -> int:
        return max(self.histories)

    @property
    def scheduled_rows_per_model(self) -> int:
        return (
            self.updates
            * self.batch_size
            * self.scheduled_sources_per_sequence
        )

    def feature_dim(self, history: int) -> int:
        if history not in self.histories:
            raise ValueError("history is outside the strengthened grid")
        return history * (
            3 * self.observation_dim + self.action_dim
        ) + self.action_dim + 2 * self.context_dim

    def active_coefficients(self, history: int) -> int:
        return self.observation_dim * (self.feature_dim(history) + 1)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CONFIG = RobustnessConfig()

if CONFIG.scheduled_rows_per_model != 64_000:
    raise AssertionError("strengthened ARX must reuse 64,000 rows per model")
if CONFIG.feature_dim(40) != 531:
    raise AssertionError("H40 strengthened ARX must have 531 input features")

