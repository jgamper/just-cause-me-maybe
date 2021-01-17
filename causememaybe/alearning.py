import torch
from torch import nn


class _ALearningLoss(nn.Module):
    def __init__(self):
        super(_ALearningLoss, self).__init__()

    def compute_v(self, treatments, propensities, predictions):
        return ((treatments + 1) / 2 - propensities) * predictions

    def m_loss(self, outcomes, treatments, propensities, predictions):
        """
        :param outcomes: True observed outcomes
        :param treatments: Treatments that subjects received [-1,1]
        :param propensities: Estimated propensity scores
        :param predictions: Predicted treatment effect
        :return:
        """
        raise NotImplementedError

    def forward(self, outcomes, treatments, propensities, predictions):
        assert outcomes.size() == treatments.size()
        assert propensities.size() == treatments.size()
        assert predictions.size() == treatments.size()
        v, M = self.m_loss(outcomes, treatments, propensities, predictions)
        return M.mean()


class BinaryLoss(_ALearningLoss):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def m_loss(self, outcomes, treatments, propensities, predictions):
        v = self.compute_v(treatments, propensities, predictions)
        M = -1 * (outcomes * v - torch.log(1 + torch.exp(v)))
        return v, M


class ExponentialLoss(_ALearningLoss):
    def __init__(self):
        super(ExponentialLoss, self).__init__()

    def m_loss(self, outcomes, treatments, propensities, predictions):
        v = self.compute_v(treatments, propensities, predictions)
        M = outcomes * torch.exp(-1 * v)
        return v, M


class ContinuousLoss(_ALearningLoss):
    def __init__(self):
        super(ContinuousLoss, self).__init__()

    def m_loss(self, outcomes, treatments, propensities, predictions):
        v = self.compute_v(treatments, propensities, predictions)
        M = (outcomes - v) ** 2
        return v, M
