import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd

beta_0_func = lambda X: np.exp(0.7 - 0.3 * X[:, 0] + 0.8 * X[:, 1])
beta_1_func = lambda X: np.exp(0.9 - 0.5 * X[:, 0] + 0.5 * X[:, 1])


def sample_rho(size, g, alphas, betas, coef: float = 0.2):
    ones = np.ones(size)
    rho_1 = np.random.beta(alphas, betas, size=size)
    rho_2 = 0.65 * g + 0.35 * (1 - g)
    rho = (1 - coef) * rho_1 + coef * rho_2
    rho = np.vstack([rho, ones]).transpose()
    rho = np.min(rho, axis=1)
    return rho


def generate_factual(df):
    ys = []
    for id, row in df.iterrows():
        t = row["treatment_assignment"].astype(np.int32)
        y = [row["y_control"], row["y_test"]]
        ys.append(y[t])
    df["observed"] = np.array(ys)
    return df


class DiscreteToyProcess:
    def __init__(
        self,
        n_samples: int = 5000,
        noise: float = 0.4,
        random_state: int = 0,
        overlap: str = "random",
        coef: float = 0.2,
    ):
        """

        :param n_samples:
        :param noise:
        :param random_state:
        :param overlap: one of ["random", "strong", "moderate", "weak"]
        :param coef:
        """
        assert overlap in ["random", "strong", "moderate", "weak"]
        self.n_samples = n_samples
        self.noise = (noise,)
        self.random_state = random_state
        self.overlap = overlap
        self.coef = coef
        if self.overlap != "random":
            self._propensity_scale = {"strong": 0.5, "moderate": 2, "weak": 4}[
                self.overlap
            ]
        self._initialise_dataset()

    def _initialise_dataset(self):
        X, latent_factor = make_moons(
            n_samples=self.n_samples, noise=self.noise, random_state=self.random_state
        )
        X = MaxAbsScaler().fit_transform(X)
        self.X = X
        self.latent_binary_confounder = latent_factor
        self.y_test, self.rho_test, self.mu_test = self.sample_outcomes("test")
        self.y_control, self.rho_control, self.mu_control = self.sample_outcomes(
            "control"
        )
        self.treatment, self.treatment_prob = self._treatment_process()
        self.df = pd.DataFrame(
            {
                "x0": self.X[:, 0],
                "x1": self.X[:, 1],
                "latent_binary_confounder": self.latent_binary_confounder,
                "treatment_assignment": self.treatment,
                "true_treatment_prob": self.treatment_prob,
                "y_test": self.y_test,
                "y_control": self.y_control,
                "mu_test": self.mu_test,
                "mu_control": self.mu_control,
                "rho_test": self.rho_test,
                "rho_control": self.rho_control,
            }
        )
        self.df = generate_factual(self.df)

    def _treatment_process(self):
        if self.overlap == "random":
            treatment = np.random.randint(0, 2, size=self.X.shape[0])
            return treatment, np.ones_like(treatment) * 0.5
        else:
            self.clf = LogisticRegression()
            self.clf.fit(self.X, self.latent_binary_confounder)
            # Set hard coefficients
            self.clf.coef_ = np.array([[1.18, -1.76]]) * self._propensity_scale
            self.clf.intercept_ = 0.00765551
            treatment_prob = self.clf.predict_proba(self.X)[:, 1]
            treatment = np.random.binomial(n=1, p=treatment_prob)
            return treatment, treatment_prob

    def sample_outcomes(self, treatment: str):
        size = self.X.shape[0]
        if treatment == "test":
            betas = beta_1_func(self.X)
            alpha = 3
            alphas = np.ones(size) * alpha
            rho = sample_rho(
                size, self.latent_binary_confounder, alphas, betas, coef=self.coef
            )
        elif treatment == "control":
            betas = beta_1_func(self.X)
            alpha = 1
            alphas = np.ones(size) * alpha
            rho = sample_rho(
                size, self.latent_binary_confounder, alphas, betas, coef=self.coef
            )

        y = np.random.binomial(1, rho, size)
        mu = (
            1 / (1 + betas / alphas) * (1 - self.coef)
            + self.coef * self.latent_binary_confounder
        )
        return y, rho, mu
