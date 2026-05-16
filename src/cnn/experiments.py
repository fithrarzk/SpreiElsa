from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product


CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


@dataclass(frozen=True)
class CNNExperimentConfig:
    experiment_id: str
    conv_depth: int
    filters_base: int
    kernel_size: int
    pooling: str
    image_size: int = 64
    num_classes: int = 6
    dense_units: int = 64
    learning_rate: float = 0.001

    @property
    def filters(self) -> list[int]:
        return [self.filters_base * (2**idx) for idx in range(self.conv_depth)]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["filters"] = self.filters
        return data


def all_shared_experiments(image_size: int = 64) -> list[CNNExperimentConfig]:
    configs: list[CNNExperimentConfig] = []
    for depth, filters_base, kernel_size, pooling in product([1, 2], [16, 32], [3, 5], ["max", "avg"]):
        experiment_id = f"d{depth}_f{filters_base}_k{kernel_size}_{pooling}"
        configs.append(
            CNNExperimentConfig(
                experiment_id=experiment_id,
                conv_depth=depth,
                filters_base=filters_base,
                kernel_size=kernel_size,
                pooling=pooling,
                image_size=image_size,
            )
        )
    return configs


def get_experiment(experiment_id: str, image_size: int = 64) -> CNNExperimentConfig:
    for config in all_shared_experiments(image_size=image_size):
        if config.experiment_id == experiment_id:
            return config
    available = ", ".join(config.experiment_id for config in all_shared_experiments(image_size=image_size))
    raise ValueError(f"Unknown experiment_id {experiment_id!r}. Available: {available}")

