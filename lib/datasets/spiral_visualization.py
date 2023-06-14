import torch
from lib.train_dataclasses import TrainEpochState


def visualize_spiral(plt, state: TrainEpochState, device_id):
    state.model.eval()
    with torch.no_grad():
        for input, target, _ in state.val_dataloader:
            input = input.to(device_id, non_blocking=True)
            output = state.model(input)[1].cpu()

    class1 = input[output[:, 0] <= 0.5].cpu()
    class2 = input[output[:, 0] > 0.5].cpu()
    for data in [class1, class2]:
        x = data[:, 0].reshape(-1).numpy().tolist()
        y = data[:, 1].reshape(-1).numpy().tolist()
        plt.scatter(x, y)
