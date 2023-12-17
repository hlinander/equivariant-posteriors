
import torch.nn.functional as F


# Define Loss Function for Conventional Next Token Generative Task
def generative_next_token_loss(outputs, batch, lora_l2=True):
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
    loss_batch = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # l2 lora loss
    loss_l2 = outputs["lora_l2_loss"]

    return loss_batch + loss_l2

# Define Loss Function for Single Token Generative Task
def generative_single_token_loss(outputs, batch, lora_l2=True):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    # Find the index of the first zero in the attention mask of the first example in the batch.
    # This (index - 2) is assumed to be the position of the "answer token" in the sequence.
    index_of_zero = (attention_mask[0] == 0).nonzero(as_tuple=True)[0]
    if len(index_of_zero) > 0:
        answer_index = index_of_zero[0].item() - 2
    else:
        # Default to the last token before padding if no zero is found.
        answer_index = -2

    # Extract logits from the model outputs
    logits = outputs["logits"]

    # Compute the cross-entropy loss for the specific "answer token".
    # This compares the model's prediction for the answer token with its true label.
    loss_batch =  F.cross_entropy(logits[:, answer_index-1, :], input_ids[:, answer_index])

    # l2 lora loss
    loss_l2 = outputs["lora_l2_loss"]

    return loss_batch + loss_l2