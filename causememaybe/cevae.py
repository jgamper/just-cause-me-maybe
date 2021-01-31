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
    target_dict = Normal(
        loc=torch.zeros_like(dist.loc), scale=torch.ones_like(dist.scale)
    )
    return kl.kl_divergence(dist, target_dict)

class Model(nn.Module):
    def __init__(self, input_dim: int, distributions_dim: int, num_branches: int, binary: bool=False, model_sigma: bool=False):
        super(Model, self).__init__()
        self.model_sigma = model_sigma
        self.binary = binary
        self.num_branches = num_branches
        self.fork = nn.ModuleList(
            [
                make_mlp(
                    1,
                    [(input_dim, distributions_dim)],
                    include_bn=False,
                    final_activation=nn.Sigmoid if binary else None,
                )
                for i in range(num_branches)
            ]
        )
        if self.model_sigma:
            self.sigma_fork = nn.ModuleList(
                [
                    make_mlp(
                        1,
                        [(input_dim, distributions_dim)],
                        include_bn=False,
                        final_activation=nn.Softplus,
                    )
                    for i in range(num_branches)
                ]
            )

    def get_mu(self, input, treatment=None):
        mu = self.fork[0](input)
        if self.num_branches > 1:
            mu_1 = self.fork[1](input)
            mu = (1 - treatment) * mu + treatment * mu_1
        return mu

    def get_sigma(self, input, treatment=None):
        sigma = self.sigma_fork[0](input)
        if self.num_branches > 1:
            sigma_1 = self.sigma_fork[1](input)
            sigma = (1 - treatment) * sigma + treatment * sigma_1
        return sigma

    def forward(self, input, treatment=None):
        mu = self.get_mu(input, treatment)
        if self.binary:
            return Bernoulli(mu)
        else:
            if self.model_sigma:
                sigma = self.get_sigma(input, treatment)
                return Normal(mu, sigma)
            else:
                return Normal(mu, 1)

class PModel(nn.Module):
    def __init__(
        self,
        features_dim: int,
        latent_confounder_dim: int,
        treatment_dim: int,
        outcome_dim: int
    ):
        super(PModel, self).__init__()
        self.p_t_z = Model(latent_confounder_dim, treatment_dim, num_branches=1, binary=True, model_sigma=False)
        self.p_y_zt = Model(latent_confounder_dim, outcome_dim, num_branches=2, binary=False, model_sigma=False)
        self.p_x_z = Model(latent_confounder_dim, features_dim, num_branches=1, binary=False, model_sigma=True)

    def forward(self, latent, treatment):
        p_x_z = self.p_x_z(latent)
        p_t_z = self.p_t_z(latent)
        p_y_zt = self.p_y_zt(latent, treatment)
        return p_x_z, p_t_z, p_y_zt


class QGuide(nn.Module):
    def __init__(
        self,
        features_dim: int,
        latent_confounder_dim: int,
        treatment_dim: int,
        outcome_dim: int
    ):
        super(QGuide, self).__init__()
        self.q_t_x = Model(features_dim, treatment_dim, num_branches=1, binary=True, model_sigma=False)
        self.q_y_xt = Model(features_dim, outcome_dim, num_branches=2, binary=False, model_sigma=False)
        self.q_z_tyx = Model(features_dim + 1, latent_confounder_dim, num_branches=2, binary=False, model_sigma=True)

    def forward(self, feature, treatment, outcome):
        q_z_tyx = self.q_z_tyx(torch.cat([feature, treatment], 1), outcome)
        q_t_x = self.q_t_x(feature)
        q_y_xt = self.q_y_xt(feature, treatment)
        return q_z_tyx, q_t_x, q_y_xt


class CEVAE(nn.Module):
    def __init__(
        self,
        features_dim: int,
        latent_confounder_dim: int,
        treatment_dim: int,
        outcome_dim: int,
    ):
        super(CEVAE, self).__init__()
        self.pmodel = PModel(
            features_dim, latent_confounder_dim, treatment_dim, outcome_dim
        )
        self.qguide = QGuide(
            features_dim, latent_confounder_dim, treatment_dim, outcome_dim
        )

    def forward(self, feature, treatment, outcome):
        q_z_tyx, q_t_x, q_y_xt = self.qguide(feature, treatment, outcome)
        p_x_z, p_t_z, p_y_zt = self.pmodel(q_z_tyx.rsample(), treatment)
        return {
            "q_z_tyx": q_z_tyx,
            "q_t_x": q_t_x,
            "q_y_xt": q_y_xt,
            "p_x_z": p_x_z,
            "p_t_z": p_t_z,
            "p_y_zt": p_y_zt,
        }

    def negative_sampling_loss(self, feature, treatment, outcome):
        treatment_flipped = 1 - treatment
        q_z_t_flipped_yx, _, _ = self.qguide(feature, treatment_flipped, outcome)
        return kl_loss_with_normal(q_z_t_flipped_yx)

    def loss(self, distributions, feature, treatment, outcome, negative_sampling=False):
        # Reconstruction loss
        l1 = distributions["p_x_z"].log_prob(feature)  # p(x|z)
        l2 = distributions["p_t_z"].log_prob(treatment)  # p(t|z)
        l3 = distributions["p_y_zt"].log_prob(outcome)  # p(y|t,z)

        # REGULARIZATION LOSS
        # approximate KL
        l4 = kl_loss_with_normal(distributions["q_z_tyx"])

        # AUXILIARY LOSS
        # q(t|x)
        l5 = distributions["q_t_x"].log_prob(treatment)
        # q(y|x,t)
        l6 = distributions["q_y_xt"].log_prob(outcome)

        # Negative sampling
        if negative_sampling:
            l7 = self.negative_sampling_loss(feature, treatment, outcome)
            loss = torch.mean(l1 + l2 + l3 + l4 + l5 + l6 + l7)
        else:
            loss = torch.mean(l1 + l2 + l3 + l4 + l5 + l6)
        return -1*loss

if __name__ == "__main__":
    import numpy as np
    from scipy.special import expit
    from torch.optim import Adam
    from tqdm import tqdm

    N = 1000
    sigma_0 = 3
    sigma_1 = 5
    Z = np.random.binomial(1, 0.5, size=(N))
    X = np.random.normal(loc=Z, scale=(Z * sigma_1 ** 2 + (1 - Z) * sigma_0 ** 2))
    T = np.random.binomial(1, 0.75 * Z + 0.25 * (1 - Z))
    Y = np.random.binomial(1, expit(3 * (Z + 2 * (2 * T - 1))))


    def sample_batch(z, x, t, y):
        ind = np.random.randint(0, len(z), size=16)
        return (i[ind, None].astype(np.float32) for i in [z, x, t, y])


    cevae = CEVAE(features_dim=1, latent_confounder_dim=1, treatment_dim=1, outcome_dim=1)

    optimizer = Adam(cevae.parameters())


    def update():
        optimizer.zero_grad()
        z, x, t, y = sample_batch(Z, X, T, Y)
        x, t, y = torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(y)
        distributions = cevae(x, t, y)

        loss = cevae.loss(distributions, x, t, y, negative_sampling=False)

        loss.backward()
        optimizer.step()
        return loss, distributions


    losses = []
    for i in tqdm(range(100)):
        l, dists = update()
        losses.append(l)