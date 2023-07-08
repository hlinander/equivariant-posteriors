import torch

# class LyapunovMetric:
#     def __call__(
#         self,
#         metric_sample: MetricSample,
#         model: torch.nn.Module,
#     ) -> torch.Tensor:
#         output = metric_sample.output.detach()
#         prediction = metric_sample.prediction.detach()
#         target = metric_sample.target.detach()
#         lambda1(model, metric_sample.)
#         return values

#     def name(self):
#         return self.metric_name


def lambda1(model: torch.nn.Module, batch: torch.Tensor):
    jacobians = torch.autograd.functional.jacobian(model, batch).squeeze()
    # jacobian will be diagonal over the samples in the batch
    # pick out just the diagonal part of the tensor
    jacobians = jacobians.diagonal(dim1=0, dim2=2)

    jacobians = jacobians.permute(2, 0, 1)

    u, s, vh = torch.linalg.svd(jacobians)

    # Return singular values (ordered by norm)
    return s[:, 0]
