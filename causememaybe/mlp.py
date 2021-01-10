from typing import List, Tuple, Optional
from torch import nn


def make_mlp(
    num_layers: int,
    num_hidden: List[Tuple[int, int]],
    include_bn: bool = True,
    final_activation: Optional[nn.Module] = None,
) -> nn.Module:
    """
    TODO add drop-out regularisation
    :param num_layers:
    :param num_hidden:
    :param include_bn:
    :param final_activation:
    :return:
    """
    # Creates layers in an order Linear, Tanh, Linear, Tanh,.. and so on.. using list comprehension
    if include_bn == True:
        layers = [
            [
                nn.Linear(num_hidden[i][0], num_hidden[i][1]),
                nn.BatchNorm1d(num_hidden[i][1]),
                nn.ELU(),
            ]
            for i in range(num_layers - 1)
        ]
    else:
        layers = [
            [nn.Linear(num_hidden[i][0], num_hidden[i][1]), nn.ELU()]
            for i in range(num_layers - 1)
        ]

    layers = [layer for sublist in layers for layer in sublist]

    # Append last layer which will be just Linear in this case
    layers.append(
        nn.Linear(num_hidden[num_layers - 1][0], num_hidden[num_layers - 1][1])
    )
    if final_activation != None:
        layers.append(final_activation())

    # Convert into model
    model = nn.Sequential(*layers)

    return model