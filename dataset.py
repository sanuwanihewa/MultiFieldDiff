import os
import numpy as np
import torch
from torch.utils.data import Dataset


def LoadDataSet(load_dir, padding=True, Norm=True, is_mask=False):
    data = np.load(load_dir)

    if data.ndim == 3:
        data = np.expand_dims(np.transpose(data, (0, 2, 1)), axis=1)
    else:
        data = np.transpose(data, (1, 0, 3, 2))

    # dtype handling
    if is_mask:
        data = data.astype(np.int32)     # keep labels
    else:
        data = data.astype(np.float32)

    if padding:
        pad_x = int((256 - data.shape[2]) / 2)
        pad_y = int((256 - data.shape[3]) / 2)
        data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))

    if (not is_mask) and Norm:
        data = (data - 0.5) / 0.5

    return data
class MixedFieldDataset(Dataset):
    """
    Returns:
      x_cond: input (source)
      y_tgt : target
      task_id: 0 for ULF->HF, 1 for HF->UHF
    """
    def __init__(self, train=True, input_path='default_path', padding=True, Norm=True):
        super().__init__()
        self.train=train

        if train:
            ulf_np = input_path + '/train/lf_t1.npy'
            hf_np_for_ulf = input_path +'/train/hf_t1.npy'
            hf_np_for_uhf = input_path +'/train/3T_T1.npy'
            uhf_np = input_path +'/train/7T_T1.npy'
            ulf_hf_np_segs = input_path +'/train/T1_segs.npy'
            hf_uhf_np_segs = input_path +'/train/7T_Segs_T1.npy'
        else:
            ulf_np = input_path +'/test/lf_t1.npy'
            hf_np_for_ulf = input_path +'/test/hf_t1.npy'
            hf_np_for_uhf = input_path +'/test/3T_T1.npy'
            uhf_np = input_path +'/test/7T_T1.npy'



        self.ulf = LoadDataSet(ulf_np, padding=padding, Norm=Norm)
        self.hf_from_ulf = LoadDataSet(hf_np_for_ulf, padding=padding, Norm=Norm)

        self.hf_for_uhf = LoadDataSet(hf_np_for_uhf, padding=padding, Norm=Norm)
        self.uhf = LoadDataSet(uhf_np, padding=padding, Norm=Norm)

        if train:
            self.ulf_hf_segs = LoadDataSet(ulf_hf_np_segs, padding=padding, Norm=Norm, is_mask=True)
            self.hf_uhf_segs = LoadDataSet(hf_uhf_np_segs, padding=padding, Norm=Norm, is_mask=True)
        else:
            self.ulf_hf_segs = np.empty(0)
            self.hf_uhf_segs = np.empty(0)

        assert len(self.ulf) == len(self.hf_from_ulf), "ULF/HF (task0) counts differ"
        assert len(self.hf_for_uhf) == len(self.uhf), "HF/UHF (task1) counts differ"

        self.n0 = len(self.ulf)
        self.n1 = len(self.hf_for_uhf)
        self.total = self.n0 + self.n1

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if idx < self.n0:
            # task 0: ULF -> HF
            x = self.ulf[idx]
            y = self.hf_from_ulf[idx]
            if self.train:
                z = self.ulf_hf_segs[idx]
            else:
                z=np.empty(0)

            task_id = 0
        else:
            # task 1: HF -> UHF
            j = idx - self.n0
            x = self.hf_for_uhf[j]
            y = self.uhf[j]
            if self.train:
                z = self.hf_uhf_segs[j]
            else:
                z=np.empty(0)
            task_id = 1

        # return tensors (C,H,W) float32
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z), torch.tensor(task_id, dtype=torch.long)
