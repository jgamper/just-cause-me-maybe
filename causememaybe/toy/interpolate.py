import torch
import numpy as np


def interpolate_constand_pdf(s: torch.Tensor, sub: int = 5):
    """
    Interpolates survival rates from discrete model outputs
    :param s: Survival rates
    :param cuts:
    :param sub:
    :return:
    """
    n, m = s.shape
    device = s.device
    diff = (s[:, 1:] - s[:, :-1]).contiguous().view(-1, 1).repeat(1, sub).view(n, -1)
    rho = (
        torch.linspace(0, 1, sub + 1, device=device)[:-1].contiguous().repeat(n, m - 1)
    )
    s_prev = s[:, :-1].contiguous().view(-1, 1).repeat(1, sub).view(n, -1)
    surv = torch.zeros(n, int((m - 1) * sub + 1))
    surv[:, :-1] = diff * rho + s_prev
    surv[:, -1] = s[:, -1]

    return surv


def interpolate_cuts(cuts: np.ndarray, sub: int = 5):
    """
    Interpolates cuts used for discrete survival modeling
    :param cuts:
    :param sub:
    :return:
    """
    cuts = np.append(
        np.concatenate(
            [
                np.linspace(start, end, num=sub + 1)[:-1]
                for start, end in zip(cuts[:-1], cuts[1:])
            ]
        ),
        cuts[-1],
    )
    return cuts
