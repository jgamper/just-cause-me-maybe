import torch
from torch import nn
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule

class FullyConnected(nn.Sequential):
    def __init__(self, sizes, final_activation=None):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)

class DistributionNet(nn.Module):
    @staticmethod
    def get_class(dtype):
        for cls in DistributionNet.__subclasses__():
            if cls.__name__.lower() == dtype + "net":
                return cls
        raise ValueError("dtype not supported: {}".format(dtype))

class BernoulliNet(DistributionNet):
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes+[1])

    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=-10, max=10)
        return logits

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits)

class NormalNet(DistributionNet):
    """
        net = NormalNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Normal(loc, scale)
#
# class Model(PyroModule):
#     def __init__(self):

num_data=1000
feature_dim=5
latent_dim=20
hidden_dim=200
num_layers=3
num_epochs=50
batch_size=100
learning_rate=1e-3
learning_rate-decay=0.1
weight_decay=1e-4
seed=1234567890
