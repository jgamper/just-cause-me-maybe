import numpy as np
from torch.utils.data import Dataset

class IhdpDataset(Dataset):
    """
    The files are obtained from https://www.fredjo.com/
    Train and test file contain 1000 repetitions of the experiment;
    there are 25 covariates within "x". Test set contains 75 units,
    and train set contains 672 units. The train set is split into 70-30
    splits.
    """
    def __init__(self, file_path, simulation_id):
        """
        :param file_path: path to the dataset npz file
        :param simulation_id: id of the simulation from 0 to 999
        """
        super(IhdpDataset, self).__init__()
        assert (simulation_id > 0) and (simulation_id < 1000)
        self.simulation_id = simulation_id
        self._load(file_path, simulation_id)

    def _load(self, file_path, simulation_id):
        dataset = np.load(file_path, allow_pickle=True)
        self.x = dataset['x'][:, :, simulation_id]
        self.t = dataset['t'][:, simulation_id]
        self.y_f = dataset['yf'][:, simulation_id]
        self.y_cf = dataset['ycf'][:, simulation_id]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        t, y_f, y_cf, x = self.t[item], self.y_f[item], self.y_cf[item], self.x[item]
        return x, t, y_f, y_cf

def iterate_sim