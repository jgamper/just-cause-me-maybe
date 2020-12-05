"""
How do you train your three-headed DragonNet to do causal inference? An answer to this question, young
padawan, you may find in the following article https://arxiv.org/abs/1906.02120 by such notable Jedis as
C. Shi, D. Blei, V. Veitch from a galaxy far far away in a far far future (2019).
Their implementation has been sent to us from far far future: https://github.com/claudiashi57/dragonnet
"""
from torch import nn
from torch import optim
import copy
from pytorch_lightning import LightningModule
from causememaybe.models.mlp import make_mlp
from causememaybe.models.dragon.modules import EpsilonLayer, TargettedLoss
from typing import Union

class DragonNet(LightningModule):
    def __init__(self,
                 input_dim: int,
                 ratio: Union[float, int],
                 feature_dim: int = 200,
                 hidden_dim: int = 200,
                 lr: float = 3e-4,
                 weight_decay: float = 0):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        self.encoder = make_mlp(
            num_layers=3,
            num_hidden=[
                (input_dim, hidden_dim),
                (hidden_dim, hidden_dim),
                (hidden_dim, feature_dim),
            ],
            include_bn=False,
            final_activation=nn.ELU,
        )

        # According to the paper for the decoders they had 100 hidden units
        self.decoders = self.make_decoders(feature_dim, int(hidden_dim / 2))

        self.epsilon_module = EpsilonLayer()

        self.loss = TargettedLoss(ratio)

    def forward(self, x):
        features = self.encoder(x)
        out = {}
        for decoder_name, decoder in self.decoders.items():
            out[decoder_name] = (decoder(features))
        return out

    def training_step(self, batch, batch_idx):

        x, treatment, y_true, y_cf_true = batch
        outputs = self.forward(x)
        epsilons = self.epsilon_module(outputs["propensity_score"])
        loss = self.loss(
            outputs["propensity_score"],
            outputs["outcome_of_0"],
            outputs["outcome_of_1"],
            epsilons,
            y_true,
            treatment,
        )
        return loss

    @staticmethod
    def make_decoders(feature_dim, hidden_dim):
        """
        Builds a three-headed decoder
        :param feature_dim: dimension of input features
        :param hidden_dim:
        :return:
        """
        # Propensity score decoder
        propensity = make_mlp(
            num_layers=1,
            num_hidden=[(feature_dim, 1)],
            include_bn=False,
            final_activation=nn.Sigmoid,
        )

        # Conditional outcome decoders
        # consists of three layers
        outcome = make_mlp(
            num_layers=3,
            num_hidden=[
                (feature_dim, hidden_dim),
                (hidden_dim, hidden_dim),
                (hidden_dim, 1)
            ],
            include_bn=False,
        )

        return nn.ModuleDict(
            {
                "propensity_score": propensity,
                "outcome_of_0": outcome,
                "outcome_of_1": copy.deepcopy(outcome),
            }
        )

    def configure_optimizers(self):
        """
        This is required as part of pytorch-lightning
        :return:
        """
        return optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )