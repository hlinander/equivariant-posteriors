from typing import List
from lib.data_factory import get_factory
from dataclasses import dataclass, field
from lib.data_factory import get_factory
import math


@dataclass(frozen=True)
class DataSubsetConfig:
    data_config: object
    subset: List[int] = field(
        default_factory=lambda: []
    )  # The subset to use, one of ``all`` or the keys in ``cifarc_subsets``
    minimum_epoch_length: int = 10000

    def serialize_human(self):
        return dict(data_config=self.data_config.serialize_human(), subset=self.subset)


# fmt:on
class DataSubset:
    def __init__(self, config: DataSubsetConfig):
        self.ds = get_factory().create(config.data_config)
        n_cycles = math.ceil(config.minimum_epoch_length / len(config.subset))
        assert isinstance(config.subset, list)
        self.subset = config.subset * n_cycles

    @staticmethod
    def data_spec(config: DataSubsetConfig):
        return get_factory().get_class(config.data_config).data_spec(config.data_config)

    def __getitem__(self, idx):
        # breakpoint()
        return self.ds[self.subset[idx]]

    def __len__(self):
        return len(self.subset)
