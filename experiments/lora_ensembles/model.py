from dataclasses import dataclass

import transformers
import peft

import torch

from lib.dataspec import DataSpec
import lib.serialize_human


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
