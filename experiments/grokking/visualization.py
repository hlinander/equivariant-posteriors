import torch
from lib.train_dataclasses import TrainEpochState


def _sample_predictions(model, dataloader, device_id, n_samples):
    batch = next(iter(dataloader))
    batch = {k: v.to(device_id) for k, v in batch.items()}
    with torch.no_grad():
        output = model(batch)
    preds = output["predictions"].detach().cpu()
    targets = batch["target"].detach().cpu()
    pred_classes = preds.argmax(dim=-1)
    n = min(n_samples, len(targets))
    correct = (pred_classes[:n] == targets[:n]).sum().item()
    rows = []
    for i in range(n):
        gt = targets[i].item()
        pc = pred_classes[i].item()
        conf = preds[i, pc].item()
        marker = "ok" if gt == pc else "!!"
        rows.append(f"{gt:>4} {pc:>4} {conf:>5.2f}  {marker}")
    return correct, n, rows


def visualize_parity(plt, state: TrainEpochState, device_id, n_samples=6):
    state.model.eval()

    lines = []
    for label, dl in [("Train", state.train_dataloader), ("Val", state.val_dataloader)]:
        if dl is None:
            continue
        correct, n, rows = _sample_predictions(state.model, dl, device_id, n_samples)
        lines.append(f"{label} ({correct}/{n} correct)")
        lines.append(f"{'GT':>4} {'Pred':>4} {'Conf':>5}")
        lines.extend(rows)
        lines.append("")

    text = "\n".join(lines)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.text(text, 0, 1, color="black")
    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.xticks([])
    plt.yticks([])
