import torch

# import tqdm

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metrics
from lib.data_factory import DataCIFARConfig


def create_config_function(model_config: object, batch_size: int):
    def create_config(ensemble_id):
        loss = torch.nn.CrossEntropyLoss()

        def ce_loss(outputs, targets):
            return loss(outputs["logits"], targets)

        train_config = TrainConfig(
            # model_config=MLPClassConfig(widths=[50, 50]),
            model_config=model_config,  # MLPClassConfig(widths=[128] * 2),
            train_data_config=DataCIFARConfig(),
            val_data_config=DataCIFARConfig(validation=True),
            loss=ce_loss,
            optimizer=OptimizerConfig(
                optimizer=torch.optim.SGD,
                kwargs=dict(weight_decay=1e-4, lr=0.05, momentum=0.9),
                # kwargs=dict(weight_decay=0.0, lr=0.001),
            ),
            batch_size=batch_size,  # 2**13,
            ensemble_id=ensemble_id,
        )
        train_eval = create_classification_metrics(None, 10)
        train_run = TrainRun(
            compute_config=ComputeConfig(distributed=False, num_workers=16),
            train_config=train_config,
            train_eval=train_eval,
            epochs=300,
            save_nth_epoch=1,
            validate_nth_epoch=5,
        )
        return train_run

    return create_config
