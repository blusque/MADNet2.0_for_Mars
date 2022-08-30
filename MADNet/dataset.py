import torch
from torch.utils.data import Dataset
import h5py


class DEMDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        self.file = h5py.File(file, 'r')
        self.dtm_dst = self.file.get('dtm_grp/dst1')
        self.ori_dst = self.file.get('ori_grp/dst1')
        self.len = self.ori_dst.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.dtm_dst[index, ...]).float(), torch.from_numpy(self.ori_dst[index, ...]).float()

    def __len__(self):
        return self.len
