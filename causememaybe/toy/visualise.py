import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import networkx as nx
import numpy as np
from causememaybe.toy.generate_toy import DataGeneratingProcess


def plot_dataset(dgp: DataGeneratingProcess):
    """
    Illiustrates our dataset
    :param dgp:
    :return:
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("terrain")
    ind0, ind1 = (
        np.where(dgp.latent_binary_confounder == 0),
        np.where(dgp.latent_binary_confounder == 1),
    )
    ax.scatter(
        dgp.X[ind0, 0],
        dgp.X[ind0, 1],
        marker="d",
        label="$G = \\mathdefault{0}$",
        alpha=0.5,
        c="g",
        edgecolors="black",
        s=60,
    )
    scatter = ax.scatter(
        dgp.X[ind1, 0],
        dgp.X[ind1, 1],
        marker="o",
        label="$G = \\mathdefault{1}$",
        alpha=0.5,
        c="y",
        edgecolors="black",
        s=60,
    )
    legend = ax.legend()
    ax.set_xlabel("Observed feature 1")
    ax.set_ylabel("Observed feature 2")
    ax.set_title("Our Dataset")
    return fig, ax


def plot_treatment(dgp: DataGeneratingProcess):
    """
    Plots the treatment assignment and propensity score density
    :param dgp:
    :return:
    """
    sample_size = 1000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    if dgp.overlap != "random":
        h = 0.2
        x_min, x_max = dgp.X[:, 0].min() - 0.5, dgp.X[:, 0].max() + 0.5
        y_min, y_max = dgp.X[:, 1].min() - 0.5, dgp.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = dgp.clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        cm = plt.cm.RdBu
        Z = Z.reshape(xx.shape)
        pcm = ax1.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
        cbar = fig.colorbar(pcm, ax=ax1)
        cbar.set_label("Propensity Model $\pi(x)$")

    for i in range(sample_size):
        t = dgp.treatment[i]
        c = "c" if t == 1 else "m"
        m = "o" if dgp.latent_binary_confounder[i] == 1 else "d"
        ax1.scatter(
            dgp.X[i, 0], dgp.X[i, 1], marker=m, alpha=0.8, c=c, edgecolors="black", s=80
        )

    combs = [
        ("c", "o", "$G = \\mathdefault{1}, A = \\mathdefault{1}$"),
        ("m", "o", "$G = \\mathdefault{1}, A = \\mathdefault{0}$"),
        ("c", "d", "$G = \\mathdefault{0}, A = \\mathdefault{1}$"),
        ("m", "d", "$G = \\mathdefault{0}, A = \\mathdefault{0}$"),
    ]

    handles = [
        Line2D(
            [0],
            [0],
            color=c,
            marker=m,
            lw=0,
            markersize=15,
            label=l,
            markeredgecolor="black",
        )
        for c, m, l in combs
    ]

    ax1.legend(handles=handles, loc="lower left")

    ax1.set_xlabel("Observed feature 1")
    ax1.set_ylabel("Observed feature 2")
    ax1.set_title("Our Dataset, Subset of {} samples".format(sample_size))

    treated = dgp.treatment_prob[np.where(dgp.treatment == 1)]
    control = dgp.treatment_prob[np.where(dgp.treatment == 0)]
    bins = np.linspace(0, 1, 30)
    ax2.hist(treated, bins, alpha=0.5, label="$A = \\mathdefault{1}$", hatch="//")
    ax2.hist(control, bins, alpha=0.5, label="$A = \\mathdefault{0}$")
    ax2.set_xlabel("Propensity Score $\pi(x)$")
    ax2.set_ylabel("Density")
    ax2.set_title("Overlap")
    ax2.legend(loc="upper right")
    return fig, (ax1, ax2)
