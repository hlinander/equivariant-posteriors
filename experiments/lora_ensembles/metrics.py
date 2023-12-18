import torch
import torchmetrics as tm
import torch.nn.functional as F
from typing import Dict


def create_metric_sample(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Set input_ids for masked tokens to -100 so they are not used in loss computation
    input_ids[attention_mask == 0] = -100

    # labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids.clone()[:, 1:]
    labels[:, -1] = -100

    # reshape to [batch_size * sequence_length, num_classes] and [batch_size * sequence_length]
    logits = output["logits"].view(-1, output["logits"].size(-1))
    labels = labels.view(-1)

    # Create a mask for filtering out ignored indices
    mask = labels != -100

    # Apply the mask to filter both predictions and targets
    logits = logits[mask]
    labels = labels[mask]

    return dict(
        output=logits.detach(),
        prediction=F.softmax(logits, dim=-1).detach(),
        target=labels.detach(),
    )


def calibration_error(output, batch):
    num_classes = output["logits"].shape[-1]
    metric_sample = create_metric_sample(output, batch)
    return tm.functional.classification.calibration_error(
        metric_sample["prediction"],
        metric_sample["target"],
        n_bins=15,
        num_classes=num_classes,
        task="multiclass",
    )


def accuracy(output, batch):
    num_classes = output["logits"].shape[-1]
    metric_sample = create_metric_sample(output, batch)
    return tm.functional.accuracy(
        metric_sample["prediction"],
        metric_sample["target"],
        task="multiclass",
        num_classes=num_classes,
    )
