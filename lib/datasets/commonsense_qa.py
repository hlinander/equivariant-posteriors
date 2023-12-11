from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
from dataclasses import dataclass
import transformers
import torch
import torch.nn.functional as F

from typing import Dict
from lib.train_dataclasses import TrainEpochState
from lib.dataspec import DataSpec
from lib.metric import MetricSample
import lib.serialize_human


@dataclass
class DataCommonsenseQaConfig:
    dataset: str = "commonsense_qa"
    model_checkpoint: str = "meta-llama/Llama-2-7b-hf"
    max_len: int = 256
    validation: bool = False
    num_samples: int = None

    def serialize_human(self):
        return lib.serialize_human.serialize_human(self.__dict__)


class DataCommonsenseQa(Dataset):
    def __init__(self, data_config: DataCommonsenseQaConfig):
        # data config
        self.data_config = data_config

        # Load the dataset
        raw_dataset = load_dataset(data_config.dataset)

        # Select train or validation split
        dataset_split = "validation" if data_config.validation else "train"
        self.dataset = raw_dataset[dataset_split]

        # If num_samples is specified and is a positive integer, slice the dataset
        if (
            data_config.num_samples
            and isinstance(data_config.num_samples, int)
            and data_config.num_samples > 0
        ):
            self.dataset = self.dataset.select(range(data_config.num_samples))
        else:
            self.dataset = self.dataset

        # Apply the formatting to the dataset
        formatted_dataset = self.dataset.map(self._format_question_answer)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            data_config.model_checkpoint, add_prefix_space=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize the dataset
        col_to_delete = [
            "id",
            "question",
            "question_concept",
            "choices",
            "formatted_question_answer",
            "answerKey",
        ]
        self.tokenized_dataset = formatted_dataset.map(
            self._preprocess, batched=True, remove_columns=col_to_delete
        )

        self.tokenized_dataset.set_format("torch")
        self.collate_fn = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _format_question_answer(self, item):
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]

        # Formatting choices
        formatted_choices = "\n".join(
            [f"({label.lower()}) {choices[i]}" for i, label in enumerate(labels)]
        )

        # Constructing a formatted question-answer string
        formatted_question_answer = f"Q: {question}\nAnswer Choices:\n{formatted_choices}\nA: ({answer_key.lower()})."

        # Return a dictionary with the formatted question-answer pair
        return {"formatted_question_answer": formatted_question_answer}

    def _preprocess(self, batch):
        # Extract the text to be tokenized.
        texts = batch["formatted_question_answer"]

        # Tokenize the text
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.data_config.max_len,
            padding="max_length",
        )

    @staticmethod
    def data_spec(config: DataCommonsenseQaConfig):
        return DataSpec(
            input_shape=torch.Size([1]),
            output_shape=torch.Size([1]),
            target_shape=torch.Size([1]),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.tokenized_dataset[idx]
        data["sample_id"] = [idx]
        return data
