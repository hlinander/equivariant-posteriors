#!/usr/bin/env python
import torch
import numpy as np
from pathlib import Path
import tqdm
import json
from typing import List
import math

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.regression_metrics import create_regression_metrics

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import symlink_checkpoint_files
from lib.files import prepare_results
from lib.serialization import serialize_human

import lib.data_factory as data_factory
import lib.model_factory as model_factory
from dataclasses import dataclass
from lib.dataspec import DataSpec

import transformers
import peft
import datasets


@dataclass
class LLama2Config:
    checkpoint: str
    lora_rank: int = 16
    lora_alpha: float = 16
    lora_dropout: float = 0.05

    def serialize_human(self):
        return serialize_human(self.__dict__)


class LLama2Model(torch.nn.Module):
    def __init__(self, config: LLama2Config, data_spec: DataSpec):
        super().__init__()
        self.config = config
        # assert data_spec.input_shape[-1] == 2
        # assert data_spec.output_shape[-1] == 2
        self.full_model = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=config.checkpoint,
                num_labels=2,
                device_map="auto",
                offload_folder="offload",
                trust_remote_code=True,
            )
        )
        self.full_model.config.pad_token_id = self.full_model.config.eos_token_id
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.SEQ_CLS,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",
                "v_proj",
            ],
        )

        self.model = peft.get_peft_model(self.full_model, peft_config)

    def forward(self, x):
        return dict(logits=x, predictions=x)


@dataclass
class NLPDatasetConfig:
    dataset: str
    model_checkpoint: str
    max_len: int = 512

    def serialize_human(self):
        return serialize_human(self.__dict__)


class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, data_config: NLPDatasetConfig):
        self.config = data_config
        self.dataset = datasets.load_dataset(data_config.dataset)["train"]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            data_config.model_checkpoint, add_prefix_space=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        def preprocessing_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, max_length=data_config.max_len
            )

        self.tokenized_dataset = self.dataset.map(preprocessing_function, batched=True)
        self.tokenized_dataset = self.tokenized_dataset.rename_column("target", "label")
        self.tokenized_dataset.set_format("torch")
        self.collate_fn = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)

    @staticmethod
    def data_spec(config: NLPDatasetConfig):
        return DataSpec(
            input_shape=torch.Size([1]),
            output_shape=torch.Size([1]),
            target_shape=torch.Size([1]),
        )

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.dataset)


LLAMA_CHECKPOINT = "meta-llama/Llama-2-7b-hf"

def create_config(ensemble_id):
    loss = torch.nn.L1Loss()

    def reg_loss(outputs, targets):
        return loss(outputs["logits"], targets)

    train_config = TrainConfig(
        # model_config=MLPConfig(widths=[256, 256, 256]),
        model_config=LLama2Config(checkpoint=LLAMA_CHECKPOINT),
        train_data_config=NLPDatasetConfig(
            dataset="mehdiiraqui/twitter_disaster",
            model_checkpoint=LLAMA_CHECKPOINT,
            # model_checkpoint="roberta-large",
            max_len=512,
        ),
        val_data_config=NLPDatasetConfig(
            dataset="mehdiiraqui/twitter_disaster",
            # model_checkpoint="meta-llama/Llama-2-7b-hf",
            model_checkpoint=LLAMA_CHECKPOINT,
            max_len=512,
        ),
        # val_data_config=NLPDatasetConfig(npoints=100),
        loss=reg_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
        ),
        batch_size=256,
        ensemble_id=ensemble_id,
        _version=34,
    )
    train_eval = create_regression_metrics(torch.nn.functional.l1_loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=10),
        train_config=train_config,
        train_eval=train_eval,
        epochs=300,
        save_nth_epoch=1,
        validate_nth_epoch=5,
    )
    return train_run


if __name__ == "__main__":
    device_id = ddp_setup()

    # clifford_structure = torch.tensor(
    #     np.load(Path(__file__).parent / "./clifford_110_structure.npz"),
    #     dtype=torch.float32,
    # )
    # clifford_blades = json.loads(
    #     open(Path(__file__).parent / "clifford_110_blades.json").read()
    # )
    # basis = {tuple(key): idx for idx, key in enumerate(clifford_blades)}
    # v1 = torch.tensor([0.0] * 8)
    # v2 = torch.tensor([0.0] * 8)
    # v1[basis[(1,)]] = 1.0
    # v2[basis[(2,)]] = 1.0
    # w = torch.einsum("i,ikl,k->l", v1, clifford_structure, v2)
    data_factory.get_factory()
    data_factory.register_dataset(NLPDatasetConfig, NLPDataset)

    mf = model_factory.get_factory()
    mf.register(LLama2Config, LLama2Model)
    ensemble_config = create_ensemble_config(create_config, 1)
    ensemble = create_ensemble(ensemble_config, device_id)

    ds = data_factory.get_factory().create(NLPDatasetConfig(npoints=100))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=9,
        shuffle=False,
        drop_last=False,
    )

    result_path = prepare_results(
        Path(__file__).parent,
        f"{Path(__file__).stem}",
        ensemble_config,
    )
    symlink_checkpoint_files(ensemble, result_path)

    # import matplotlib.pyplot as plt

    # for xs, ys, ids in tqdm.tqdm(dl):
    #     xs = xs.to(device_id)

    #     output = ensemble.members[0](xs)["logits"].cpu().detach().numpy()
    #     fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    #     for idx, (start_deltas, delta, target) in enumerate(
    #         zip(xs.cpu().numpy(), output, ys.numpy())
    #     ):
    #         start = np.zeros_like(delta)
    #         # breakpoint()
    #         for i in range(start_deltas.shape[0]):
    #             start[i + 1] = start[i] + start_deltas[i]
    #         # delta = np.concatenate([[[0, 0]], delta], axis=0)
    #         ax = axs[idx // 3, idx % 3]
    #         ax.plot(start[:, 0], start[:, 1], "g--", label="initial", alpha=0.2)
    #         ax.plot(
    #             start[:, 0] + delta[:, 0],
    #             start[:, 1] + delta[:, 1],
    #             "r-",
    #             label="clifford",
    #         )
    #         ax.plot(
    #             start[:, 0] + target[:, 0],
    #             start[:, 1] + target[:, 1],
    #             "b-",
    #             label="target",
    #         )
    #         ax.legend()
    #         ax.set_title(f"{length_cable(start + target)}, {99 * 0.05}")
    #         fig.suptitle("80 iteration constraint resolve")
    #         fig.savefig("clifford_test.pdf")
    #     raise Exception("exit")
