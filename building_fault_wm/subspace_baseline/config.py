"""Fixed development-selection and held-out evaluation contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CASES = (
    "bestest_hydronic_heat_pump",
    "multizone_office_simple_air",
    "twozone_apartment_hydronic",
)
POLICIES = ("old_2h", "new_4h")
MODEL_SEEDS = (
    202608011,
    202608012,
    202608013,
    202608014,
    202608015,
)
SILENT_FAMILIES = ("bias", "drift", "stuck")


@dataclass(frozen=True)
class SubspaceConfig:
    """Immutable model-selection and analysis settings."""

    observation_dim: int = 4
    input_dim: int = 6
    block_rows: tuple[int, ...] = (8, 12, 16)
    state_orders: tuple[int, ...] = (2, 4, 6, 8, 12, 16)
    innovation_clip_sigmas: tuple[float, ...] = (0.0, 3.0, 5.0)
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    bootstrap_draws: int = 10_000
    bootstrap_seed: int = 202608029
    maximum_spectral_radius: float = 1.0
    blas_threads: int = 1

    def __post_init__(self) -> None:
        if self.block_rows != (8, 12, 16):
            raise ValueError("subspace block-row grid changed")
        if self.state_orders != (2, 4, 6, 8, 12, 16):
            raise ValueError("subspace state-order grid changed")
        if self.innovation_clip_sigmas != (0.0, 3.0, 5.0):
            raise ValueError("subspace innovation grid changed")
        if self.horizons != (1, 2, 4, 8):
            raise ValueError("subspace horizon grid changed")
        if self.bootstrap_draws != 10_000 or self.bootstrap_seed != 202608029:
            raise ValueError("subspace bootstrap contract changed")
        if self.maximum_spectral_radius != 1.0:
            raise ValueError("subspace stability rule changed")
        if self.blas_threads != 1:
            raise ValueError("subspace linear-algebra thread limit changed")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CONFIG = SubspaceConfig()
