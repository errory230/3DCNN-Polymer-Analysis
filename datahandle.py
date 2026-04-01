import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

def grouped_train_test_split(data, y ,meta, ngxids, reps_per_sample=50, n_samples=226, seed=230):
    random.seed(seed)
    all_indices = list(range(n_samples))
    random.shuffle(all_indices)

    train_size = 180
    train_samples = all_indices[:train_size]
    test_samples = all_indices[train_size:]

    train_data, test_data = [], []
    train_y, test_y = [], []
    train_meta, test_meta = [], []
    train_id, test_id = [], []
    for idx in train_samples:
        start = idx * reps_per_sample
        end = (idx + 1) * reps_per_sample
        train_data.extend(data[start:end])
        train_y.extend(y[start:end])
        train_meta.extend(meta[start:end])
        train_id.extend(ngxids[start:end])

    for idx in test_samples:
        start = idx * reps_per_sample
        end = (idx + 1) * reps_per_sample
        test_data.extend(data[start:end])
        test_y.extend(y[start:end])
        test_meta.extend(meta[start:end])
        test_id.extend(ngxids[start:end])

    return train_data, train_y, train_meta, test_data, test_y, test_meta, train_id, test_id


class VoxelPairDataset(Dataset):
    def __init__(self, X_data, y_data, meta_data, ngxids):
        self.X = X_data
        self.meta = meta_data
        self.y = y_data
        self.ngxid = ngxids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        voxel1_np = self.X[idx][0]
        voxel2_np = self.X[idx][1]
        meta_info = self.meta[idx]
        target = self.y[idx]
        ngxid = self.ngxid[idx]

        voxel1_tensor = torch.from_numpy(voxel1_np).float()
        voxel2_tensor = torch.from_numpy(voxel2_np).float()
        meta_info_tensor = torch.from_numpy(meta_info).float()
        target_tensor = torch.tensor([target], dtype=torch.float32)

        return (voxel1_tensor, voxel2_tensor, meta_info_tensor), target_tensor, ngxid

def custom_collate_fn_with_permute(batch):
    voxel1s_raw = [item[0][0] for item in batch] 
    voxel2s_raw = [item[0][1] for item in batch] 
    meta_infos = [item[0][2] for item in batch]
    targets = [item[1] for item in batch]
    ngxids = [item[2] for item in batch]

    voxel1s_permuted = [v.permute(3, 2, 1, 0) for v in voxel1s_raw]
    voxel2s_permuted = [v.permute(3, 2, 1, 0) for v in voxel2s_raw]

    max_x = 50
    max_y = 25
    max_z = 25

    padded_voxel1s = []
    padded_voxel2s = []

    for v1_permuted, v2_permuted in zip(voxel1s_permuted, voxel2s_permuted):
        pad_x1 = max_x - v1_permuted.shape[3]
        pad_y1 = max_y - v1_permuted.shape[2]
        pad_z1 = max_z - v1_permuted.shape[1]
        padded_v1 = F.pad(v1_permuted, (0, pad_x1, 0, pad_y1, 0, pad_z1), 'constant', 0)
        padded_voxel1s.append(padded_v1)

        pad_x2 = max_x - v2_permuted.shape[3]
        pad_y2 = max_y - v2_permuted.shape[2]
        pad_z2 = max_z - v2_permuted.shape[1]
        padded_v2 = F.pad(v2_permuted, (0, pad_x2, 0, pad_y2, 0, pad_z2), 'constant', 0)
        padded_voxel2s.append(padded_v2)

    stacked_meta_infos = torch.stack(meta_infos)

    return torch.stack(padded_voxel1s), torch.stack(padded_voxel2s), stacked_meta_infos, torch.stack(targets), ngxids
