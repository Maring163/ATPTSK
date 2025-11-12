import torch, math

def atp_softmin(x, epsilon=0.001, dim=-1, kappa=690, xi=730):
    """
    Adaptive Three-Parameters Softmin.
    the index parameter q, eta and omega are adaptively determined according to current membership values x
    :param x: membership values, tensor type
    :param epsilon: the hyperparamter newly added in this approach, default=0.001
    :param dim: {int}, get the minimum on which dimension
    :return: ATP-softmin's output, tensor type
    """
    x = x.double()
    num_rules, num_features = x.size(1), x.size(2)
    random_feature = torch.randint(0, num_features, size=(1, num_rules))
    x[:, range(num_rules), random_feature] *= math.exp(- 1 / 10 ** 5) # avoiding x_max equal x_min
    x_max, x_min = x.data.max(dim=dim).values, x.data.min(dim=dim).values
    eta = 1 / (x_min * x_max ** ((kappa - math.log(num_features)) / xi)) ** (xi / (xi + kappa - math.log(num_features)))
    q = -xi / (eta * x_max).log()
    omega = ((x - x_min.unsqueeze(dim)) <= epsilon).float().sum(dim)
    firing_str = (1 / eta) * ((((eta.unsqueeze(dim) * x) ** q.unsqueeze(dim)).sum(dim) / omega) ** (1 / q))

    return firing_str
