#!/usr/bin/env python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metrics
from lib.generic_ablation import generic_ablation

import lib.data_factory as data_factory
import lib.model_factory as model_factory
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.data_registry import DataCommonsenseQaConfig, DataCommonsenseQa
from lib.models.llama2generative import LLama2GenerativeConfig




# Define Loss Function for Generative Task
def generative_loss(outputs, batch):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    # Set input_ids for masked tokens to -100 so they are not used in loss computation
    input_ids[attention_mask == 0] = -100

    # labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids.clone()[:, 1:]
    labels[:, -1] = -100  # Ignore the loss for the last token

    # loss
    logits = outputs["logits"]
    # Reshape logits to [batch_size * sequence_length, num_classes]
    # Reshape labels to [batch_size * sequence_length]
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

# LLAMA Checkpoint
LLAMA_CHECKPOINT = "meta-llama/Llama-2-7b-hf"

# Configuration for Training
def create_config(ensemble_id, lora_rank=16):
    train_config = TrainConfig(
        model_config=LLama2GenerativeConfig(checkpoint=LLAMA_CHECKPOINT, lora_rank=lora_rank),
        train_data_config=DataCommonsenseQaConfig(
            dataset="commonsense_qa",
            model_checkpoint=LLAMA_CHECKPOINT,
            max_len=256,
            validation=True,
            num_samples = 1,
        ),
        val_data_config=DataCommonsenseQaConfig(
            dataset="commonsense_qa",
            model_checkpoint=LLAMA_CHECKPOINT,
            max_len=256,
            validation=True,
            num_samples = 1,
        ),
        loss=generative_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=0.00, lr=1e-4),
        ),
        batch_size=1,
        ensemble_id=ensemble_id,
        gradient_clipping=0.3,
        _version=37,
    )
    train_eval = create_classification_metrics(None, 32000)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=100,
        save_nth_epoch=1,
        validate_nth_epoch=1,
    )
    return train_run


def evaluate(model, eval_dataset, tokenizer, print_sample=True):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_examples = 0
    tokenizer = eval_dataset.tokenizer

    device = next(model.parameters()).device
    eval_loader = DataLoader(eval_dataset, batch_size=32)  # Adjust batch size as needed

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # Assuming labels are part of your batch

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += labels.numel()

            if print_sample:
                # Print one sample token-by-token (from the first batch only)
                print_sample_comparison(input_ids[0], logits[0], labels[0], tokenizer)

    avg_loss = total_loss / len(eval_loader)
    accuracy = total_correct / total_examples
    print(f"Evaluation Results: Average Loss = {avg_loss}, Accuracy = {accuracy}")

def print_sample_comparison(input_ids, logits, labels, tokenizer):
    print("Sample Token-by-Token Comparison:")
    for j in range(input_ids.size(0)):
        predicted_token_id = logits[j].argmax(dim=-1).item()
        true_token_id = labels[j].item()
        if true_token_id != -100:  # Ignore padding tokens
            predicted_token = tokenizer.decode([predicted_token_id])
            true_token = tokenizer.decode([true_token_id])
            print(f"Token {j}: True: '{true_token}', Predicted: '{predicted_token}'")

def evaluate(model, eval_dataset, print_sample=True):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_examples = 0
    tokenizer = eval_dataset.tokenizer

    device = next(model.parameters()).device
    eval_loader = DataLoader(eval_dataset, batch_size=4)  # Adjust batch size as needed

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            labels[:, :-1] = labels.clone()[:, 1:]

            # Mask for filtering labels and logits
            valid_labels_mask = labels != -100

            reshaped_batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            outputs = model(batch=reshaped_batch)
            logits = outputs.logits

            # Apply mask to logits and labels
            filtered_logits = logits[valid_labels_mask]
            filtered_labels = labels[valid_labels_mask]

            loss = generative_loss(outputs, reshaped_batch)

            total_loss += loss.item()
            total_correct += (filtered_logits.argmax(dim=-1) == filtered_labels).sum().item()
            total_examples += filtered_labels.numel()


            if print_sample:
                # Print one sample token-by-token (from the first batch only)
                print_sample_comparison(input_ids[0], logits[0], labels[0], tokenizer)

    avg_loss = total_loss / len(eval_loader)
    accuracy = total_correct / total_examples
    print(f"Evaluation Results: Average Loss = {avg_loss}, Accuracy = {accuracy}")

def print_sample_comparison(input_ids, logits, labels, tokenizer):
    print("Sample Token-by-Token Comparison:")
    for j in range(input_ids.size(0)):
        predicted_token_id = logits[j].argmax(dim=-1).item()
        true_token_id = labels[j].item()
        if true_token_id != -100:  # Ignore padding tokens
            predicted_token = tokenizer.decode([predicted_token_id])
            true_token = tokenizer.decode([true_token_id])
            print(f"Token {j}: True: '{true_token}', Predicted: '{predicted_token}'")


def main():
    print("Start")
    if torch.cuda.is_available():
        device_id = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    else:
        device_id = "cpu"
    print("device finished")
    ensemble_config = create_ensemble_config(create_config, 1)
    print("ensemble_config finished")
    ensemble = create_ensemble(ensemble_config, device_id)
    print("ensemble finished")
    print(ensemble.n_members)

    eval_dataset_config = DataCommonsenseQaConfig(
            dataset="commonsense_qa",
            model_checkpoint=LLAMA_CHECKPOINT,
            max_len=256,
            validation=True,
            num_samples = 1
        )
    eval_dataset = DataCommonsenseQa(eval_dataset_config)
    evaluate(ensemble.members[0], eval_dataset, print_sample=True)

if __name__ == "__main__":
    main()

