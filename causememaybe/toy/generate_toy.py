import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Dict

def sample_gompertz(x,
                    shape_parameter_alpha=-1,
                    scale_parameter_lambda=1,
                    beta=np.array([1, -0.2])):
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
    square_brackets = 1 - (shape_parameter_alpha*u)/(scale_parameter_lambda*np.exp(np.dot(x, beta)))
    return (1/shape_parameter_alpha) * np.log(square_brackets)

def generate_factual(df):
    ys = []
    indicators = []
    for id, row in df.iterrows():
        t = row['treatment_assignment']
        y1 = row['y_test']
        y0 = row['y_control']
        c = row['c']
        if t:
            y = np.min([y1, c])
            indi = np.argmin([y1, c])
        else:
            y = np.min([y0, c])
            indi = np.argmin([y0, c])
        ys.append(y)
        indicators.append(indi)
    df['observed'] = np.array(ys)
    df['delta'] = np.array(indicators)
    return df

class DataGeneratingProcess():

    def __init__(self,
                 n_samples: int = 5000,
                 noise: float = 0.4,
                 random_state: int = 0,
                 scale: float = 2.1,
                 param_test: Dict = {"shape_parameter_alpha":0.3,
                                     "scale_parameter_lambda":0.5,
                                     "beta":np.array([1, -0.2])},
                 param_control: Dict = {"shape_parameter_alpha":0.6,
                                        "scale_parameter_lambda":0.5,
                                        "beta":np.array([0.9, -0.1])},
                 uniform_censoring: bool = True,
                 random_treatment_assignment: bool = True
                 ):
        self.n_samples = n_samples
        self.noise = noise,
        self.random_state = random_state
        self.param_test = param_test
        self.param_control = param_control
        self.scale = scale
        self.uniform_censoring = uniform_censoring
        self.random_treatment_assignment = random_treatment_assignment
        self._initialise_dataset()

    def _initialise_dataset(self):
        X, latent_factor = make_moons(n_samples=self.n_samples,
                                      noise=self.noise,
                                      random_state=self.random_state)
        X = StandardScaler().fit_transform(X)
        self.X = X
        self.latent_binary_confounder = latent_factor
        self.y_test = self.sample_outcomes("test")
        self.y_control = self.sample_outcomes("control")
        self. c = self._censoring_process()
        self.treatment, self.treatment_prob = self._treatment_process()
        self.mu_test, self.mu_control = self._compute_mus()
        self.df = pd.DataFrame({
                        "x0": self.X[:,0],
                        "x1": self.X[:,1],
                        "latent_binary_confounder": self.latent_binary_confounder,
                        "treatment_assignment": self.treatment,
                        "true_treatment_prob": self.treatment_prob,
                        "mu_test": self.mu_test,
                        "mu_control": self.mu_control,
                        "y_test": self.y_test,
                        "y_control": self.y_control,
                        "c": self.c
                    })
        self.df = generate_factual(self.df)

    def _censoring_process(self):
        if not self.uniform_censoring:
            c = np.random.lognormal(mean=0.7, sigma=0.4, size=self.X.shape[0])
        else:
            c = np.random.uniform(0.5, 16, size=self.X.shape[0])
        return c

    def _treatment_process(self):
        if self.random_treatment_assignment:
            treatment = np.random.randint(0, 2, size=self.X.shape[0])
            return treatment, np.ones_like(treatment)*0.5
        else:
            self.clf = SVC(kernel="linear", C=0.025, probability=True)
            self.clf.fit(self.X, self.latent_binary_confounder)
            treatment_prob = self.clf.predict_proba(self.X)[:, 1] * np.random.rand()
            treatment = np.random.binomial(n=1, p=treatment_prob)
            return treatment, treatment_prob

    def _compute_mus(self):
        """
        Computes mean outcome given the treatment
        :return: 
        """
        mu_test = np.mean(
            np.stack([self.sample_outcomes("test") for _ in range(10000)]),
            axis=0
        )
        mu_control = np.mean(
            np.stack([self.sample_outcomes("control") for _ in range(10000)]),
            axis=0
        )
        return mu_test, mu_control

    def sample_outcomes(self, treatment: str):
        if treatment == "test":
            y = sample_gompertz(self.X, **self.param_test) + \
                 self.latent_binary_confounder * self.scale
        elif treatment == "control":
            y = sample_gompertz(self.X, **self.param_control)
        else:
            raise AssertionError("Wrong treatment specified: {}".format(treatment))
        return y






