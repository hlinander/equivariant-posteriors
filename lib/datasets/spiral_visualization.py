import torch
from lib.train_dataclasses import TrainEpochState


def visualize_spiral(plt, state: TrainEpochState):
    state.model.eval()
    with torch.no_grad():
        for input, target, _ in state.val_dataloader:
            input = input.to("cuda:0", non_blocking=True)
            output = state.model(input).cpu()

    class1 = input[output[:, 0] <= 0.5].cpu()
    class2 = input[output[:, 0] > 0.5].cpu()
    for data in [class1, class2]:
        x = data[:, 0].reshape(-1).numpy().tolist()
        y = data[:, 1].reshape(-1).numpy().tolist()
        plt.scatter(x, y)
