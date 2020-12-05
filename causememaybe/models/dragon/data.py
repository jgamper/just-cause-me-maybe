import numpy as np
from torch.utils.data import Dataset

def load_data(file_path):
    """
    Code taken directly from the source
    https://github.com/claudiashi57/dragonnet/blob/master/src/experiment/idhp_data.py
    :param file_path:
    :return:
    """
    data = np.loadtxt(file_path, delimiter=",")
    t, y, y_cf = data[:, 0], data[:, 1][:, None], data[:, 2][:, None]
    mu_0, mu_1, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 5:]
    return t.reshape(-1, 1), y, y_cf, mu_0, mu_1, x

class IhdpDataset(Dataset):

    def __init__(self, file_path):
        super(IhdpDataset, self).__init__()
        self._load(file_path)

    def _load(self, file_path):
        t, y, y_cf, mu_0, mu_1, x = load_data(file_path)
        self.t = t.astype(np.float32)
        self.y = y.astype(np.float32)
        self.y_cf = y_cf.astype(np.float32)
        self.x = x.astype(np.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, item):
        t, y, y_cf, x = self.t[item], self.y[item], self.y_cf[item], self.x[item]
        return x, t, y
