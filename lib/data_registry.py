from lib.datasets.sine import DataSineConfig, DataSine
from lib.datasets.spiral import DataSpiralsConfig, DataSpirals
from lib.datasets.uniform import DataUniformConfig, DataUniform
from lib.datasets.mnist import DataMNISTConfig, DataMNIST
from lib.datasets.mnist_2 import DataMNIST2Config, DataMNIST2
from lib.datasets.cifar import DataCIFARConfig, DataCIFAR
from lib.datasets.cifar_c import DataCIFAR10CConfig, DataCIFAR10C
from lib.datasets.cifar_2 import DataCIFAR2Config, DataCIFAR2
from lib.datasets.subset import DataSubsetConfig, DataSubset
from lib.datasets.join import DataJoinConfig, DataJoin


def register_datasets():
    datasets = dict()
    datasets[DataSpiralsConfig] = DataSpirals
    datasets[DataSineConfig] = DataSine
    datasets[DataUniformConfig] = DataUniform
    datasets[DataMNISTConfig] = DataMNIST
    datasets[DataMNIST2Config] = DataMNIST2
    datasets[DataCIFARConfig] = DataCIFAR
    datasets[DataCIFAR10CConfig] = DataCIFAR10C
    datasets[DataCIFAR2Config] = DataCIFAR2
    datasets[DataSubsetConfig] = DataSubset
    datasets[DataJoinConfig] = DataJoin
    return datasets
