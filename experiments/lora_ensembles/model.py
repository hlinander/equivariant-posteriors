from typing import Dict, List
from dataclasses import dataclass

import transformers
import peft

import torch

from lib.dataspec import DataSpec
import lib.serialize_human
from lib.serialization import DeserializeConfig, get_checkpoint_path, deserialize_model
from lib.train_dataclasses import TrainRun


@dataclass
class LLama2Config:
    checkpoint: str
    lora_rank: int = 16
    lora_alpha: float = 16
    lora_dropout: float = 0.05

    def serialize_human(self):
        return lib.serialize_human.serialize_human(self.__dict__)


class LLama2Model(torch.nn.Module):
    def __init__(self, config: LLama2Config, data_spec: DataSpec):
        super().__init__()
        self.config = config
        # assert data_spec.input_shape[-1] == 2
        # assert data_spec.output_shape[-1] == 2
        self.full_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=config.checkpoint,
            num_labels=2,
            device_map=0,
            # offload_folder="offload",
            trust_remote_code=True,
        )
        self.full_model.config.pad_token_id = self.full_model.config.eos_token_id
        self.peft_config = peft.LoraConfig(
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

        self.model = peft.get_peft_model(self.full_model, self.peft_config)
        # self.dummy = torch.nn.Linear(1, 1)
        # breakpoint()

    def state_dict(self):
        """Override state_dict with only adapter weights"""
        return peft.get_peft_model_state_dict(self.model)

    def load_state_dict(self, state_dict, strict=False):
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, batch):
        output = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        predictions = torch.softmax(output["logits"].detach(), dim=-1)
        output["predictions"] = predictions
        return output
        # return dict(logits=x, predictions=x)


@dataclass
class DeserializedModelStateDict:
    state_dict: Dict[str, torch.Tensor]
    epoch: int


def deserialize_model_state_dict(config: DeserializeConfig):
    train_config = config.train_run.train_config
    checkpoint_path = get_checkpoint_path(train_config)
    if (
        not (checkpoint_path / "model").is_file()
        or not (checkpoint_path / "epoch").is_file()
    ):
        return None
    else:
        print(f"{checkpoint_path}")

    try:
        model_state_dict = torch.load(
            checkpoint_path / "model", map_location=torch.device(config.device_id)
        )

        model_epoch = torch.load(checkpoint_path / "epoch")
    except Exception as e:
        print(f"Failed to deserialize_model: {e}")
        return None

    return DeserializedModelStateDict(state_dict=model_state_dict, epoch=model_epoch)


@dataclass
class LORAEnsemble:
    model: torch.nn.Module
    ensemble_state_dicts: List[Dict[str, torch.Tensor]]

    def ensemble_forward(self, batch):
        outputs = []
        for member_state_dict in self.ensemble_state_dicts:
            self.model.load_state_dict(member_state_dict)
            output = self.model(batch)
            output = {k: v.detach() for k, v in output.items()}
            outputs.append(output)


def create_lora_ensemble(member_configs: List[TrainRun], device_id):
    state_dicts = []
    for member_config in member_configs:
        deserialized_state_dict = deserialize_model_state_dict(
            DeserializeConfig(train_run=member_config, device_id=device_id)
        )
        if not deserialized_state_dict.epoch >= member_config.epochs:
            print(
                f"WARNING: Member not fully trained ({deserialized_state_dict.epoch}/{member_config.epochs} epochs)"
            )
        state_dicts.append(deserialized_state_dict.state_dict)
    deserialized_model = deserialize_model(
        DeserializeConfig(train_run=member_configs[0], device_id=device_id)
    )
    return LORAEnsemble(
        model=deserialized_model.model, ensemble_state_dicts=state_dicts
    )
