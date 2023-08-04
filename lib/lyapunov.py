import torch
from lib.metric import MetricSample


class LyapunovMetric:
    def __call__(
        self,
        metric_sample: MetricSample,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        def _model(x):
            return model(x)["logits"]

        lambdas = lambda1(
            _model,
            metric_sample.batch.detach().reshape(metric_sample.batch.shape[0], -1),
        )
        return lambdas

    def name(self):
        return "Lyapunov"


def lambda1(model: torch.nn.Module, batch: torch.Tensor):
    # jacobians = torch.func.jacrev(model)(batch).squeeze()
    jacobians = torch.autograd.functional.jacobian(
        model, batch, vectorize=True
    ).squeeze()
    # jacobian will be diagonal over the samples in the batch
    # pick out just the diagonal part of the tensor
    jacobians = jacobians.diagonal(dim1=0, dim2=2)

    jacobians = jacobians.permute(2, 0, 1)

    u, s, vh = torch.linalg.svd(jacobians)

    # Return singular values (ordered by norm)
    return s[:, 0]
