"""Frozen constants for the secondary schedule-matched ARX addendum."""

from __future__ import annotations

from dataclasses import asdict, dataclass


PARENT_PACKAGE_DIGEST = (
    "b758859c6cb99d34930452c36e3fd59b5abd0e7f56b19710fa2b1998b23760b8"
)
CASES = (
    "bestest_hydronic_heat_pump",
    "multizone_office_simple_air",
    "twozone_apartment_hydronic",
)
POLICIES = ("old_2h", "new_4h")
SILENT_FAMILIES = ("bias", "drift", "stuck")


@dataclass(frozen=True)
class ARXAddendumConfig:
    """Prospectively fixed information, exposure, and selection contract."""

    observation_dim: int = 4
    action_dim: int = 1
    context_dim: int = 5
    history: int = 8
    sequence_length: int = 48
    batch_size: int = 4
    updates: int = 400
    scheduled_sources_per_sequence: int = 40
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    alphas: tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
    )
    model_seeds: tuple[int, ...] = (
        202608011,
        202608012,
        202608013,
        202608014,
        202608015,
    )

    def __post_init__(self) -> None:
        if (
            self.observation_dim,
            self.action_dim,
            self.context_dim,
            self.history,
        ) != (4, 1, 5, 8):
            raise ValueError("the frozen ARX information contract changed")
        if (
            self.sequence_length,
            self.batch_size,
            self.updates,
            self.scheduled_sources_per_sequence,
        ) != (48, 4, 400, 40):
            raise ValueError("the frozen ARX source-exposure contract changed")
        if self.horizons != (1, 2, 4, 8):
            raise ValueError("the frozen ARX evaluation horizons changed")
        if self.alphas != (
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0,
            10.0,
            100.0,
        ):
            raise ValueError("the frozen Ridge selection grid changed")
        if len(set(self.model_seeds)) != 5:
            raise ValueError("the addendum requires five paired schedule seeds")

    @property
    def feature_dim(self) -> int:
        history_fields = (
            3 * self.observation_dim
            + self.action_dim
        )
        current_fields = self.action_dim + 2 * self.context_dim
        return self.history * history_fields + current_fields

    @property
    def active_coefficients(self) -> int:
        return self.observation_dim * (self.feature_dim + 1)

    @property
    def scheduled_rows_per_model(self) -> int:
        return (
            self.updates
            * self.batch_size
            * self.scheduled_sources_per_sequence
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FROZEN_CONFIG = ARXAddendumConfig()

if FROZEN_CONFIG.feature_dim != 115:
    raise AssertionError("the frozen ARX feature dimension must be 115")
if FROZEN_CONFIG.active_coefficients != 464:
    raise AssertionError("the frozen ARX coefficient count must be 464")
if FROZEN_CONFIG.scheduled_rows_per_model != 64_000:
    raise AssertionError("the frozen ARX exposure must be 64,000 rows per model")

