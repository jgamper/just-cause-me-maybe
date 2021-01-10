import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd

m_0_func = lambda X, G: -0.2 + 0.1/(1+np.exp(-1*X[:,0])) + 0.8 * np.sin(X[:,1]) + 0.8 * G
m_1_func = lambda X, G: -0.1 + 0.1*X[:,0]**2 - 0.2 * np.sin(X[:,1]) - 0.85 * (1 - G)
alpha_0_func = lambda X, G: np.exp(0.7-1.8*X[:,0] + 0.8 * X[:,1])
alpha_1_func = lambda X, G: np.exp(0.9-0.5*X[:,0] + 0.5*X[:,1])

def sample_gompertz(
    x, shape_parameter_alpha=-1, scale_parameter_lambda=1, beta=np.array([1, -0.2])
):
    """
    Generates survival data according to:
    @article{bender2005generating,
        title={Generating survival times to simulate Cox proportional hazards models},
        author={Bender, Ralf and Augustin, Thomas and Blettner, Maria},
        journal={Statistics in medicine},
        volume={24},
        number={11},
        pages={1713--1723},
        year={2005},
        publisher={Wiley Online Library}
        }
    PDF: https://www.econstor.eu/bitstream/10419/31002/1/483869465.PDF
    :param x:
    :param shape_parameter_alpha: alpha \in \{-\infty, +\infty}
    :param scale_parameter_lambda: lambda > 0
    :param beta:
    :return:
    """
    u = np.log(np.random.uniform(0, 1, x.shape[0]))
    square_brackets = 1 - (shape_parameter_alpha * u) / (
        scale_parameter_lambda * np.exp(np.dot(x, beta))
    )
    return (1 / shape_parameter_alpha) * np.log(square_brackets)

def sample_weidbull(
    x, scale_parameter_lambda=1200, shape_parameter_alpha=2, median=False
):
    """
    Generates survival data according to:
    @article{bender2005generating,
        title={Generating survival times to simulate Cox proportional hazards models},
        author={Bender, Ralf and Augustin, Thomas and Blettner, Maria},
        journal={Statistics in medicine},
        volume={24},
        number={11},
        pages={1713--1723},
        year={2005},
        publisher={Wiley Online Library}
        }
    PDF: https://www.econstor.eu/bitstream/10419/31002/1/483869465.PDF
    :param x:
    :param shape_parameter_alpha: alpha \in \{-\infty, +\infty}
    :param scale_parameter_lambda: lambda > 0
    :param median: used for median time computation
    :return:
    """
    if median == False:
        numerator = -1 * np.log(np.random.uniform(0, 1, x.shape[0]))
    if median == True:
        numerator = -1 * np.log(0.5)
    denominator = scale_parameter_lambda * np.exp(x)
    return (numerator / denominator)**(1/shape_parameter_alpha)


def generate_factual(df):
    ys = []
    indicators = []
    for id, row in df.iterrows():
        t = row["treatment_assignment"]
        y1 = row["y_test"]
        y0 = row["y_control"]
        c = row["c"]
        if t:
            y = np.min([y1, c])
            indi = np.argmin([y1, c])
        else:
            y = np.min([y0, c])
            indi = np.argmin([y0, c])
        ys.append(y)
        indicators.append(indi)
    df["observed"] = np.array(ys)
    df["delta"] = np.array(indicators)
    return df


class DataGeneratingProcess:
    def __init__(
        self,
        n_samples: int = 5000,
        noise: float = 0.4,
        random_state: int = 0,
        censoring_rate: float = 2.5,
        overlap: str = "random",
        proportional_hazards: bool = True
    ):
        """

        :param n_samples:
        :param noise:
        :param random_state:
        :param censoring_rate: one of [2.5, 0.5] 2 corresponds to smaller censoring
        :param overlap: one of ["random", "strong", "moderate", "weak"]
        :param proportional_hazards: if False then weibull scale paramter is a function of covariates
        """
        assert overlap in ["random", "strong", "moderate", "weak"]
        if censoring_rate not in [2.5, 0.5]:
            print("Check the impact your censoring rate might have!")
        self.n_samples = n_samples
        self.noise = (noise,)
        self.random_state = random_state
        self.censoring_rate = censoring_rate
        self.overlap = overlap
        if self.overlap != "random":
            self._propensity_scale = {"strong": 0.5,
                                      "moderate": 2,
                                      "weak": 4}[self.overlap]
        self.proportional_hazards = proportional_hazards
        self._initialise_dataset()

    def _initialise_dataset(self):
        X, latent_factor = make_moons(
            n_samples=self.n_samples, noise=self.noise, random_state=self.random_state
        )
        X = MaxAbsScaler().fit_transform(X)
        self.X = X
        self.latent_binary_confounder = latent_factor
        self.y_test = self.sample_outcomes("test")
        self.y_control = self.sample_outcomes("control")
        self.c = self._censoring_process()
        self.treatment, self.treatment_prob = self._treatment_process()
        self.mu_control = self.sample_outcomes("control", median=True)
        self.mu_test = self.sample_outcomes("test", median=True)
        self.df = pd.DataFrame(
            {
                "x0": self.X[:, 0],
                "x1": self.X[:, 1],
                "latent_binary_confounder": self.latent_binary_confounder,
                "treatment_assignment": self.treatment,
                "true_treatment_prob": self.treatment_prob,
                "y_test": self.y_test,
                "y_control": self.y_control,
                "c": self.c,
                "mu_test": self.mu_test,
                "mu_control": self.mu_control
            }
        )
        self.df = generate_factual(self.df)

    def _censoring_process(self):
        c = np.random.exponential(scale=self.censoring_rate, size=self.X.shape[0])
        return c

    def _treatment_process(self):
        if self.overlap == "random":
            treatment = np.random.randint(0, 2, size=self.X.shape[0])
            return treatment, np.ones_like(treatment) * 0.5
        else:
            self.clf = LogisticRegression()
            self.clf.fit(self.X, self.latent_binary_confounder)
            # Set hard coefficients
            self.clf.coef_ = np.array([[ 1.18, -1.76]]) * self._propensity_scale
            self.clf.intercept_ = 0.00765551
            treatment_prob = self.clf.predict_proba(self.X)[:, 1]
            treatment = np.random.binomial(n=1, p=treatment_prob)
            return treatment, treatment_prob

    def sample_outcomes(self, treatment: str, median=False):
        if treatment == "test":
            x = m_1_func(self.X, self.latent_binary_confounder)
            alpha = 2 if self.proportional_hazards else alpha_1_func(self.X, self.latent_binary_confounder)
            y = sample_weidbull(x,
                                shape_parameter_alpha=alpha,
                                scale_parameter_lambda=2,
                                median=median)
        elif treatment == "control":
            x = m_0_func(self.X, self.latent_binary_confounder)
            alpha = 2 if self.proportional_hazards else alpha_0_func(self.X, self.latent_binary_confounder)
            y = sample_weidbull(x,
                                shape_parameter_alpha=alpha,
                                scale_parameter_lambda=2,
                                median=median)
        else:
            raise AssertionError("Wrong treatment specified: {}".format(treatment))
        return y
