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
from lib.datasets.commonsense_qa import DataCommonsenseQaConfig, DataCommonsenseQa
from experiments.weather.data import DataHP, DataHPConfig

# from experiments.lora_ensembles.configs import NLPDatasetConfig
# from experiments.lora_ensembles.configs import NLPDataset


def register_datasets(factory):
    factory.register(DataSpiralsConfig, DataSpirals)
    factory.register(DataSineConfig, DataSine)
    factory.register(DataUniformConfig, DataUniform)
    factory.register(DataMNISTConfig, DataMNIST)
    factory.register(DataMNIST2Config, DataMNIST2)
    factory.register(DataCIFARConfig, DataCIFAR)
    factory.register(DataCIFAR10CConfig, DataCIFAR10C)
    factory.register(DataCIFAR2Config, DataCIFAR2)
    factory.register(DataSubsetConfig, DataSubset)
    factory.register(DataJoinConfig, DataJoin)
    factory.register(DataCommonsenseQaConfig, DataCommonsenseQa)
    factory.register(DataHPConfig, DataHP)
