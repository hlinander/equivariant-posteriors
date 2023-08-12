import torch


def lambda1(model: torch.nn.Module, batch: torch.Tensor):
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    jacobians = torch.autograd.functional.jacobian(model, batch).squeeze()
    # end.record()
    # torch.cuda.synchronize()
    # t_jacobian = start.elapsed_time(end)
    # jacobian will be diagonal over the samples in the batch
    # pick out just the diagonal part of the tensor
    jacobians = jacobians.diagonal(dim1=0, dim2=2)

    jacobians = jacobians.permute(2, 0, 1)

    # start.record()
    u, s, vh = torch.linalg.svd(jacobians)
    # end.record()
    # torch.cuda.synchronize()
    # t_svd = start.elapsed_time(end)
    # print(f"jac: {t_jacobian}, svd: {t_svd}")

    # Return singular values (ordered by norm)
    return torch.log(s[:, 0])
