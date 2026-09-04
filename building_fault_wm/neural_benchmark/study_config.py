from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .reliability_loss import ReliabilityLossConfig
from .reliability_model import GateMode, ReliabilityRSSMConfig


ArmName = Literal[
    "legacy",
    "ungated_h8",
    "aux_h8",
    "gated_h8",
    "huber_h8",
]
ARMS: tuple[ArmName, ...] = (
    "legacy",
    "ungated_h8",
    "aux_h8",
    "gated_h8",
    "huber_h8",
)


@dataclass(frozen=True)
class ArmConfig:
    gate_mode: GateMode
    health_weight: float
    direct_horizon_weight: float


ARM_CONFIGS: dict[ArmName, ArmConfig] = {
    "legacy": ArmConfig("bypass", 0.0, 0.0),
    "ungated_h8": ArmConfig("bypass", 0.0, 1.0),
    "aux_h8": ArmConfig("bypass", 0.25, 1.0),
    "gated_h8": ArmConfig("learned", 0.25, 1.0),
    "huber_h8": ArmConfig("huber", 0.0, 1.0),
}


@dataclass(frozen=True)
class StudyConfig:
    development_seeds: tuple[int, ...] = (202608011, 202608012, 202608013)
    confirmatory_seeds: tuple[int, ...] = (
        202608011,
        202608012,
        202608013,
        202608014,
        202608015,
    )
    schedule_seed: int = 202608010
    bootstrap_seed: int = 202608019

    observation_dim: int = 4
    action_dim: int = 1
    context_dim: int = 5
    sensor_dim: int = 2
    health_classes: int = 2
    constraint_dim: int = 1
    deterministic_dim: int = 32
    stochastic_dim: int = 8
    embedding_dim: int = 32
    hidden_dim: int = 64

    sequence_length: int = 48
    batch_size: int = 4
    gru_batch_size: int = 160
    updates: int = 400
    checkpoint_every: int = 100
    learning_rate: float = 3e-4
    gradient_clip: float = 100.0

    observation_weight: float = 1.0
    kl_weight: float = 0.1
    kl_balance: float = 0.8
    kl_free_nats: float = 1.0
    overshooting_horizon: int = 8
    overshooting_weight: float = 0.1
    direct_horizon: int = 8
    direct_horizon_beta: float = 1.0
    action_diagnostic_shift: int = 8

    validation_checkpoints: tuple[int, ...] = (100, 200, 300, 400)
    relative_improvement_threshold: float = 0.10
    healthy_degradation_limit: float = 0.05
    huber_noninferiority_limit: float = 0.02
    bootstrap_draws: int = 2_000

    def __post_init__(self) -> None:
        if self.development_seeds != self.confirmatory_seeds[:3]:
            raise ValueError("development seeds must be the first three paired seeds")
        if len(set(self.confirmatory_seeds)) != 5:
            raise ValueError("the confirmatory study requires five distinct seeds")
        if len({self.schedule_seed, self.bootstrap_seed, *self.confirmatory_seeds}) != 7:
            raise ValueError("model, schedule, and bootstrap seeds must be distinct")
        if (
            self.observation_dim,
            self.action_dim,
            self.context_dim,
            self.sensor_dim,
            self.health_classes,
        ) != (4, 1, 5, 2, 2):
            raise ValueError("the frozen observation/action/context/reliability contract changed")
        if self.sequence_length <= self.direct_horizon:
            raise ValueError("sequence length must exceed the direct imagination horizon")
        if self.direct_horizon != 8 or self.overshooting_horizon != 8:
            raise ValueError("the study is frozen to an eight-step horizon")
        if self.action_diagnostic_shift != 8:
            raise ValueError("the alternate-action diagnostic is frozen to one H8 block")
        if self.updates <= 0 or self.updates % self.checkpoint_every:
            raise ValueError("updates must be a positive multiple of checkpoint cadence")
        expected_checkpoints = tuple(
            range(self.checkpoint_every, self.updates + 1, self.checkpoint_every)
        )
        if self.validation_checkpoints != expected_checkpoints:
            raise ValueError("validation checkpoints must cover the frozen update grid")
        if (
            self.batch_size <= 0
            or self.gru_batch_size != self.batch_size * (self.sequence_length - self.direct_horizon)
            or self.learning_rate <= 0
            or self.gradient_clip <= 0
        ):
            raise ValueError(
                "optimizer settings must be positive and GRU exposure must match RSSM H8 sources"
            )
        if not 0 <= self.kl_balance <= 1 or self.kl_free_nats < 0:
            raise ValueError("KL settings are invalid")
        if self.bootstrap_draws != 2_000:
            raise ValueError("the frozen development and confirmation bootstrap uses 2,000 draws")
        limits = (
            self.relative_improvement_threshold,
            self.healthy_degradation_limit,
            self.huber_noninferiority_limit,
        )
        if any(not 0 < value < 1 for value in limits):
            raise ValueError("study effect thresholds must lie in (0, 1)")

    def model_config(self) -> ReliabilityRSSMConfig:
        return ReliabilityRSSMConfig(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sensor_dim=self.sensor_dim,
            constraint_dim=self.constraint_dim,
            deterministic_dim=self.deterministic_dim,
            stochastic_dim=self.stochastic_dim,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            health_classes=self.health_classes,
            context_dim=self.context_dim,
            sensor_observation_indices=(0, 1),
            healthy_class_index=0,
        )

    def arm_config(self, arm: ArmName) -> ArmConfig:
        try:
            return ARM_CONFIGS[arm]
        except KeyError as error:
            raise ValueError(f"unknown study arm: {arm}") from error

    def loss_config(self, arm: ArmName) -> ReliabilityLossConfig:
        arm_config = self.arm_config(arm)
        return ReliabilityLossConfig(
            observation_weight=self.observation_weight,
            cost_weight=0.0,
            constraint_weight=0.0,
            continuation_weight=0.0,
            health_weight=arm_config.health_weight,
            kl_weight=self.kl_weight,
            kl_balance=self.kl_balance,
            kl_free_nats=self.kl_free_nats,
            overshooting_horizon=self.overshooting_horizon,
            overshooting_weight=self.overshooting_weight,
            direct_horizon=self.direct_horizon,
            direct_horizon_weight=arm_config.direct_horizon_weight,
            direct_horizon_beta=self.direct_horizon_beta,
        )

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "arms": {arm: asdict(self.arm_config(arm)) for arm in ARMS},
            "model_config": asdict(self.model_config()),
            "loss_configs": {
                arm: asdict(self.loss_config(arm)) for arm in ARMS
            },
        }
