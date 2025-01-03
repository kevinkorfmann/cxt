from torch.utils.data import Dataset
import torch
from pathlib import Path
import numpy as np
from typing import Literal

def discretize(sequence, population_time):
    indices = np.searchsorted(population_time, sequence, side="right") - 1
    indices = np.clip(indices, 0, len(population_time) - 1)
    return indices.tolist()


class LazyDataset(Dataset):
    def __init__(self, data_dir: str, split: Literal['train', 'test'] = 'train', test_batches: int = 100):
        super().__init__()
        self.data_dir = Path(data_dir)
        all_batches = sorted(self.data_dir.glob('X_*.npy'), key=lambda x: int(x.stem.split('_')[1]))
        
        # Split at batch level
        if split == 'test':
            self.files = all_batches[-test_batches:]
        else:
            self.files = all_batches[:-test_batches]
            
        # Cache file paths instead of opening mmaps
        self.X_files = self.files
        self.y_files = [str(f).replace('X_', 'y_') for f in self.files]
            
        # Get shape from first file
        self.samples_per_batch = np.load(self.X_files[0]).shape[0]
        
    def __len__(self):
        return len(self.files) * self.samples_per_batch
        
    def __getitem__(self, idx, discretize_target=True):
        batch_idx = idx // self.samples_per_batch
        sample_idx = idx % self.samples_per_batch
        
        # Load and process single sample
        src = np.load(self.X_files[batch_idx], mmap_mode='r')[sample_idx].copy()
        tgt = np.load(self.y_files[batch_idx], mmap_mode='r')[sample_idx].copy()

        if discretize_target:
            tgt = np.array(discretize(tgt, np.linspace(3, 14, 256)))
        
        src = torch.from_numpy(src).float()
        src = torch.log1p(src)
        
        tgt = torch.from_numpy(tgt).long() + 2
        tgt = torch.cat([torch.tensor([1]), tgt])

        return src, tgt