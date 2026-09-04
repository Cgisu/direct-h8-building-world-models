"""Fixed development and evaluation contract for the RC comparator."""

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
class RcConfig:
    """Immutable model-selection and analysis settings."""

    topologies: tuple[str, ...] = ("1r1c", "2r2c")
    equipment_ridge_alphas: tuple[float, ...] = (
        0.0,
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
    )
    innovation_clip_sigmas: tuple[float, ...] = (0.0, 3.0, 5.0)
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    bootstrap_draws: int = 10_000
    bootstrap_seed: int = 202608029
    step_seconds: int = 900
    maximum_zone_coefficient_sum: float = 0.3
    maximum_mass_coefficient_sum: float = 0.3
    minimum_active_conductance_coefficient: float = 1e-5
    minimum_mass_to_zone_capacity_ratio: float = 1.0
    maximum_mass_to_zone_capacity_ratio: float = 100.0
    minimum_hvac_gain_k_per_kw_step: float = 1e-5
    maximum_hvac_gain_k_per_kw_step: float = 1.0
    maximum_effective_solar_aperture_m2: float = 10_000.0
    maximum_constant_heat_flow_kw: float = 1_000.0
    measurement_variance: float = 0.0004
    blas_threads: int = 1
    validation_processes: int = 3
    thermal_multistart_count: int = 9

    def __post_init__(self) -> None:
        if self.topologies != ("1r1c", "2r2c"):
            raise ValueError("RC topology grid changed")
        if self.equipment_ridge_alphas != (
            0.0,
            0.0001,
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
            1000.0,
            10000.0,
        ):
            raise ValueError("RC equipment regularization grid changed")
        if self.innovation_clip_sigmas != (0.0, 3.0, 5.0):
            raise ValueError("RC observer grid changed")
        if self.horizons != (1, 2, 4, 8):
            raise ValueError("RC horizon grid changed")
        if self.bootstrap_draws != 10_000 or self.bootstrap_seed != 202608029:
            raise ValueError("RC bootstrap contract changed")
        if (
            self.step_seconds != 900
            or self.blas_threads != 1
            or self.validation_processes != 3
            or self.thermal_multistart_count != 9
        ):
            raise ValueError("RC numerical contract changed")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CONFIG = RcConfig()
