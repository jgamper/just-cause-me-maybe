import torch
from torch import nn
import pandas as pd
from typing import Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss
from causememaybe.mlp import make_mlp
import numpy as np


def split_dataset(df_train: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Returns a dataset split into train, test
    :param name:
    :return:
    """
    df_test = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_test.index)
    df_val = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_val.index)
    return df_train, df_val, df_test


def get_labels(
    duration, events, labtrans: Optional[LabTransDiscreteTime]
) -> Tuple[torch.Tensor]:
    """
    Preprocess targets (we end up with two targets: duration and event).
    Time is discretized into `num_duration` equidistant points.
    :param duration: time-to-event data
    :param events: indicator function if censored or not
    :param labtrans: discrete label transformation object
    :return:
    """
    if labtrans != None:
        y = labtrans.fit_transform(duration, events)
        y_duration = torch.from_numpy(y[0])
        y_event = torch.from_numpy(y[1])
        return y_duration, y_event
    else:
        return torch.from_numpy(duration), torch.from_numpy(events)


def get_dataset(
    df: pd.DataFrame, batch_size: int, labtrans: Optional[LabTransDiscreteTime]
) -> Tuple[DataLoader, torch.Tensor, pd.DataFrame]:
    """
    Converts dataframe into train, valid and test dataloaders
    :param df:
    :param batch_size:
    :param labtrans:
    :return:
    """
    df_train, df_val, df_test = split_dataset(df)

    # Preprocess featuers
    x_train, x_val, x_test = [
        torch.from_numpy(df[["x0", "x1"]].to_numpy().astype("float32"))
        for df in [df_train, df_val, df_test]
    ]

    # Get treatment indicator
    a_train, a_val, a_test = [
        torch.from_numpy(df.treatment_assignment.to_numpy())
        for df in [df_train, df_val, df_test]
    ]

    # Get labels
    y_train_duration, y_train_event = get_labels(
        df_train.observed.values, df_train.delta.values, labtrans
    )
    y_val_duration, y_val_event = get_labels(
        df_val.observed.values, df_val.delta.values, labtrans
    )
    y_test_duration, y_test_event = get_labels(
        df_test.observed.values, df_test.delta.values, labtrans
    )

    # Make dataloader for training set
    train_dataset = TensorDataset(x_train, y_train_duration, y_train_event, a_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = TensorDataset(x_val, y_val_duration, y_val_event, a_val)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    test_dataset = TensorDataset(x_test, y_test_duration, y_test_event, a_test)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, df_train, df_val, df_test


def output2surv(output: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Transform a network output tensor to discrete survival estimates.
    Ref: LogisticHazard
    """
    hazards = output.sigmoid()
    return hazard2surv(hazards, epsilon)


def hazard2surv(hazard: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Transform discrete hazards to discrete survival estimates.
    Ref: LogisticHazard
    """
    return (1 - hazard).add(epsilon).log().cumsum(1).exp()


class SurvivalDragon(nn.Module):
    def __init__(self, out_size: int):
        super(SurvivalDragon, self).__init__()
        self.encoder = make_mlp(1, [(2, 2)], include_bn=False, final_activation=nn.ReLU)
        self.loss_func = NLLLogistiHazardLoss()
        self.survival_decoders = self.make_survival_decoders(out_size)

    def make_survival_decoders(self, out_size):
        decoders = []
        for i in range(2):
            mlp = make_mlp(1, [(2, out_size)], include_bn=False)
            decoders.append(mlp)
        return nn.ModuleList(decoders)

    def forward(self, x):
        features = self.encoder(x)
        y0 = self.survival_decoders[0](features)
        y1 = self.survival_decoders[1](features)
        if hasattr(self, "propensity_model"):
            propensity_score = self.propensity_model(features)
        else:
            propensity_score = None

        return features, [y0, y1], propensity_score
