import torch
from torch import nn


class TargettedLoss(nn.Module):
    def __init__(self, ratio):
        """
        TODO add description
        """
        super(TargettedLoss, self).__init__()
        self.ratio = ratio
        self.dragon_loss = DragonLoss()

    def forward(
        self, propensity_score, outcome_of_0, outcome_of_1, epsilons, y_true, treatment
    ):
        """
        TODO add description
        :param propensity_score:
        :param outcome_of_0:
        :param outcome_of_1:
        :param y_true:
        :param treatment:
        :return:
        """
        unregularized_loss = self.dragon_loss(
            propensity_score, outcome_of_0, outcome_of_1, y_true, treatment
        )

        predicted_outcomes = treatment * outcome_of_1 + (1 - treatment) * outcome_of_0
        h = treatment / propensity_score - (1 - treatment) / (1 - propensity_score)
        predicted_outcomes_pertrubed = predicted_outcomes + epsilons * h
        targeted_regularization = torch.mean(
            torch.square(y_true - predicted_outcomes_pertrubed)
        )

        return unregularized_loss + self.ratio * targeted_regularization


class DragonLoss(nn.Module):
    def __init__(self):
        """
        TODO add description
        """
        super(DragonLoss, self).__init__()
        self.binary_cross_entropy = nn.BCELoss()

    def forward(self, propensity_score, outcome_of_0, outcome_of_1, y_true, treatment):
        """
        TODO add description
        :param propensity_score:
        :param outcome_of_0:
        :param outcome_of_1:
        :param y_true:
        :param treatment:
        :return:
        """
        bce_loss = self.binary_cross_entropy(propensity_score, treatment)
        reg_loss = mean_squarred_error(outcome_of_0, outcome_of_1, y_true, treatment)
        return bce_loss + reg_loss


def mean_squarred_error(outcome_of_0, outcome_of_1, y_true, treatment):
    """
    TODO add description
    :param outcome_of_0:
    :param outcome_of_1:
    :param y_true:
    :param treatment:
    :return:
    """
    loss_of_0 = torch.mean((1 - treatment) * torch.square(outcome_of_0 - y_true))
    loss_of_1 = torch.mean(treatment * torch.square(outcome_of_1 - y_true))
    return loss_of_0 + loss_of_1


class EpsilonLayer(nn.Module):
    def __init__(self):
        """
        TODO add description
        """
        super(EpsilonLayer, self).__init__()
        self.epsilon = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.epsilon * torch.ones_like(x)
