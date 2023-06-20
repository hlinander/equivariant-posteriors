import torch


def lambda1(model: torch.nn.Module, batch: torch.Tensor):
    jacobians = torch.autograd.functional.jacobian(model, batch).squeeze()
    # jacobian will be diagonal over the samples in the batch
    # pick out just the diagonal part of the tensor
    jacobians = jacobians.diagonal(dim1=0, dim2=2)

    jacobians = jacobians.permute(2, 0, 1)

    u, s, vh = torch.linalg.svd(jacobians)

    # Return singular values (ordered by norm)
    return s[:, 0]
