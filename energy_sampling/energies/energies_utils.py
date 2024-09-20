import math
import torch


def regression_unnorm_log_prob(theta, x, y, use_prior=False, prior_scale=10.0):
    weighted_sum = -theta @ x.T

    # TODO(jberner): check
    # weighted_sum = theta @ x.T
    # offset = torch.nn.functional.relu(weighted_sum)
    # denominator = offset + torch.log(
    #     torch.exp(weighted_sum - offset) + torch.exp(-offset)
    # )
    # log_prediction = -denominator
    # swapped_y = -(y - 1.0)

    # assert log_prediction.shape == weighted_sum.shape
    # assert swapped_y.shape == (x.shape[-0], 1)
    # log_prediction = log_prediction + swapped_y.T * weighted_sum
    # unnorm_log_prob = torch.sum(log_prediction, dim=1, keepdim=True)

    unnorm_log_prob = torch.nn.functional.logsigmoid(weighted_sum)
    not_y = ~y
    unnorm_log_prob[:, not_y] -= weighted_sum[:, not_y]
    unnorm_log_prob = unnorm_log_prob.sum(dim=-1, keepdim=True)
    assert unnorm_log_prob.shape == (theta.shape[0], 1)

    if use_prior:
        prior_var = prior_scale**2
        norm_const = -0.5 * x.shape[1] * (2.0 * math.pi * prior_var).log()
        sq_sum = torch.sum(theta**2, dim=-1, keepdim=True)
        prior_log_prob = norm_const - 0.5 * sq_sum / prior_var
        assert prior_log_prob.shape == unnorm_log_prob.shape
        unnorm_log_prob = unnorm_log_prob + prior_log_prob

    return torch.flatten(unnorm_log_prob)