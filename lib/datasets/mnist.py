import functools
import torch
import torchvision
from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass(frozen=True)
class DataMNISTConfig:
    validation: bool = False

    def serialize_human(self, factories):
        return dict(validation=self.validation)


class DataMNIST(torch.utils.data.Dataset):
    def __init__(self, data_config: DataMNISTConfig):
        self.MNIST = torchvision.datasets.MNIST(
            "datasets",
            train=not data_config.validation,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        self.n_classes = 10
        self.config = data_config

    def data_spec(self):
        return DataSpec(
            input_shape=torch.Size([28 * 28]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([self.n_classes]),
        )

    @functools.cache
    def __getitem__(self, idx):
        mnist_sample = self.MNIST[idx]
        image = torch.flatten(mnist_sample[0]).reshape(28, 28)
        # image = image.unfold(0, 14, 14)
        # image = image.unfold(1, 14, 14)
        # image = image.reshape(2 * 2, 14 * 14)
        image = image.reshape(-1, 14 * 14)
        return image, mnist_sample[1], idx

    def __len__(self):
        return len(self.MNIST)
