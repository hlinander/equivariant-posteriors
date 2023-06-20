import torch
import torchvision as tv
from lib.train_dataclasses import TrainEpochState


def visualize_mnist(plt, state: TrainEpochState, device_id):
    state.model.eval()
    input, target, ids = next(iter(state.val_dataloader))
    with torch.no_grad():
        input = input.to(device_id, non_blocking=True)
        output = state.model(input)
        digits = torch.argmax(output["predictions"].cpu(), dim=-1)

    images = input.cpu().reshape(-1, 1, 28, 28)
    images = 255.0 * images[:4]
    image_grid = tv.utils.make_grid(images.long(), 2)
    image_grid = image_grid.numpy().transpose((1, 2, 0))[:, :, 0].tolist()

    digits = digits[:4].cpu().split(2)
    digits_str = "\n".join([str(x.tolist()) for x in digits])
    plt.matrix_plot(image_grid)
    plt.text(digits_str, 0, 2, color="red")
