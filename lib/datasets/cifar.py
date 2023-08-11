import functools
import torch
import torchvision
from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass(frozen=True)
class DataCIFARConfig:
    validation: bool = False

    def serialize_human(self):
        return dict(validation=self.validation)


class DataCIFAR(torch.utils.data.Dataset):
    def __init__(self, data_config: DataCIFARConfig):
        self.CIFAR = torchvision.datasets.CIFAR10(
            "datasets",
            train=not data_config.validation,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ]
            ),
        )
        self.n_classes = 10
        self.config = data_config

    @staticmethod
    def data_spec():
        return DataSpec(
            input_shape=torch.Size([4 * 3, 16 * 16]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([10]),
        )

    @functools.cache
    def __getitem__(self, idx):
        cifar_sample = self.CIFAR[idx]
        image = torch.flatten(cifar_sample[0]).reshape(3, 32, 32)
        # image = image.unfold(0, 14, 14)
        # image = image.unfold(1, 14, 14)
        # image = image.reshape(2 * 2, 14 * 14)
        image = image.reshape(-1, 16 * 16)
        return image, cifar_sample[1], idx

    def __len__(self):
        return len(self.CIFAR)
