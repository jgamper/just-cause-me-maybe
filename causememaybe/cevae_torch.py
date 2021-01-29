import torch
from torch import nn
import torch.nn.functional as F
from causememaybe.mlp import make_mlp
from torch.distributions import Bernoulli, Normal, kl

def kl_loss_with_normal(dist):
    """
    Computes kl divergence loss
    :param input_dist:
    :param target_dist:
    :return:
    """
    target_dict = Normal(loc=torch.zeros_like(dist.loc), scale=torch.ones_like(dist.scale))
    return kl.kl_divergence(dist, target_dict)

# Sub-modules
class PropensityModel(nn.Module):
    def __init__(self, feature_dim: int, treatment_dim: int):
        super(PropensityModel, self).__init__()
        self.mlp = make_mlp(
            1,
            [(feature_dim, treatment_dim)],
            include_bn=False,
            final_activation=nn.Sigmoid,
        )

    def forward(self, input):
        propensity = self.mlp(input)
        dist = Bernoulli(propensity)
        return dist


class OutcomeModel(nn.Module):
    def __init__(self, feature_dim: int, regression: bool = False):
        super(OutcomeModel, self).__init__()
        self.regression = regression
        # Two branches for two outcomes
        self.fork = nn.ModuleList(
            [
                make_mlp(
                    1,
                    [(feature_dim, 1)],
                    include_bn=False,
                    final_activation=None,
                )
                for i in range(2)
            ]
        )

    def forward(self, inupt, treatment):
        mu_0 = self.fork[0](inupt)
        mu_1 = self.fork[1](inupt)
        outcome_sampler = Normal if self.regression else Bernoulli
        mu = (1 - treatment) * mu_0 + treatment * mu_1
        dist = outcome_sampler(mu)
        return dist


class EncodingModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(EncodingModel, self).__init__()
        self.mu = make_mlp(1, [(input_dim, output_dim)], include_bn=False)
        self.sigma = make_mlp(
            1, [(input_dim, output_dim)], include_bn=False
        )

    def forward(self, input):
        mu, sigma = self.mu(input), F.softplus(self.sigma(input))
        dist = Normal(mu, sigma)
        return dist

class ZEncodingModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ZEncodingModel, self).__init__()
        # Two branches for two outcomes
        self.fork = nn.ModuleList(
            [
                make_mlp(
                    1,
                    [(input_dim, output_dim)],
                    include_bn=False,
                    final_activation=None,
                )
                for i in range(4)
            ]
        )

    def forward(self, xy, treatment):
        t = treatment
        mu_0, sigma_0 = self.fork[0](xy), F.softplus(self.fork[1](xy))
        mu_1, sigma_1 = self.fork[2](xy), F.softplus(self.fork[3](xy))
        dist = Normal((1 - t) * mu_0 + t * mu_1, (1 - t) * sigma_0 + t * sigma_1)
        return dist

class PModel(nn.Module):
    def __init__(self, features_dim: int, latent_confounder_dim: int, treatment_dim: int, regression: bool):
        super(PModel, self).__init__()
        self.p_t_z = PropensityModel(latent_confounder_dim, treatment_dim)
        self.p_y_zt = OutcomeModel(latent_confounder_dim, regression)
        self.p_x_z = EncodingModel(latent_confounder_dim, features_dim)

    def forward(self, latent, treatment):
        p_x_z = self.p_x_z(latent)
        p_t_z = self.p_t_z(latent)
        p_y_zt = self.p_y_zt(latent, treatment)
        return p_x_z, p_t_z, p_y_zt

class QGuide(nn.Module):
    def __init__(self, features_dim: int, latent_confounder_dim: int, treatment_dim: int, regression: bool):
        super(QGuide, self).__init__()
        self.q_t_x = PropensityModel(features_dim, treatment_dim)
        self.q_y_xt = OutcomeModel(features_dim, regression)
        self.q_z_tyx = ZEncodingModel(features_dim+1, latent_confounder_dim)

    def forward(self, feature, treatment, outcome):
        q_z_tyx = self.q_z_tyx(torch.cat([feature, treatment], 1), outcome)
        q_t_x = self.q_t_x(feature)
        q_y_xt = self.q_y_xt(feature, treatment)
        return q_z_tyx, q_t_x, q_y_xt
