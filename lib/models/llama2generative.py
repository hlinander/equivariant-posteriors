from dataclasses import dataclass
import transformers
import peft
import torch
from lib.dataspec import DataSpec
import lib.serialize_human
from typing import List


# Define LLAMA 2 Configuration
@dataclass
class LLama2GenerativeConfig:
    checkpoint: str = "meta-llama/Llama-2-7b-hf"
    lora_rank: int = 16
    lora_alpha: float = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = ["q_proj", "v_proj"]

    def serialize_human(self):
        return lib.serialize_human.serialize_human(self.__dict__)

# Define LLAMA 2 Model with PEFT
class LLama2Generative(torch.nn.Module):
    def __init__(self, model_config: LLama2GenerativeConfig, data_config: DataSpec):
        super().__init__()
        self.config = model_config
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config.checkpoint,
            device_map=0,
        )

        self.peft_config = peft.LoraConfig(
            task_type="CAUSAL_LM",
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            bias="none",
            target_modules=model_config.target_modules
        )
        self.model = peft.get_peft_model(self.base_model, self.peft_config)

    def forward(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        return outputs

    def state_dict(self):
        """Override state_dict with only adapter weights"""
        return peft.get_peft_model_state_dict(self.model)

    def load_state_dict(self, state_dict, strict=False):

        """ Correct key mismatches in the state_dict due to naming differences in LoRA layers. 
        Specifically, this modifies the keys to include the '.default.' segment where necessary, 
        aligning the keys in the provided state_dict with the format expected by the PEFT model. """
        updated_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("lora_A.weight", "lora_A.default.weight")
            new_key = new_key.replace("lora_B.weight", "lora_B.default.weight")
            updated_state_dict[new_key] = value

        self.model.load_state_dict(updated_state_dict, strict=False)