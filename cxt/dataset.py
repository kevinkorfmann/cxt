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



'''
class MultiDirLazyDataset(Dataset):
    def __init__(self, root_dir: str, split: Literal['train', 'test'] = 'train', test_ratio: float = 0.2):
        """
        A dataset that supports multiple subdirectories with a percentage-based split.

        Args:
            root_dir (str): Root directory containing subdirectories.
            split (Literal['train', 'test']): Whether to use training or testing data.
            test_ratio (float): Percentage of files in each subdirectory to use for testing.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.test_ratio = test_ratio
        self.X_files = []
        self.y_files = []

        # Iterate over subdirectories
        for subdir in sorted(self.root_dir.iterdir()):
            if subdir.is_dir():
                subdir_batches = sorted(subdir.glob('X_*.npy'), key=lambda x: int(x.stem.split('_')[1]))

                # Calculate the number of test files
                num_test_files = int(len(subdir_batches) * test_ratio)

                # Split files based on the calculated number
                if split == 'test':
                    selected_batches = subdir_batches[-num_test_files:]
                else:
                    selected_batches = subdir_batches[:-num_test_files]

                # Cache file paths
                self.X_files.extend(selected_batches)
                self.y_files.extend([str(f).replace('X_', 'y_') for f in selected_batches])

        # Get shape from the first file
        if self.X_files:
            self.samples_per_batch = np.load(self.X_files[0]).shape[0]
        else:
            raise ValueError("No valid data found in the provided directory structure.")

    def __len__(self):
        return len(self.X_files) * self.samples_per_batch

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
'''
    

class MultiDirLazyDataset(Dataset):
    def __init__(self, root_dir: str, split: Literal['train', 'test'] = 'train', test_ratio: float = 0.2):
        """
        A dataset that supports a hierarchy of subdirectories with a percentage-based split.

        Args:
            root_dir (str): Root directory containing subdirectories.
            split (Literal['train', 'test']): Whether to use training or testing data.
            test_ratio (float): Percentage of files in each subdirectory to use for testing.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.test_ratio = test_ratio
        self.X_files = []
        self.y_files = []

        # Recursively find all X_*.npy files
        all_files = sorted(self.root_dir.rglob('X_*.npy'), key=lambda x: (x.parent, int(x.stem.split('_')[1])))
        
        # Group files by parent directory
        from collections import defaultdict
        grouped_files = defaultdict(list)
        for file in all_files:
            grouped_files[file.parent].append(file)
        
        for subdir, subdir_batches in grouped_files.items():
            num_test_files = int(len(subdir_batches) * test_ratio)

            if split == 'test':
                selected_batches = subdir_batches[-num_test_files:]
            else:
                selected_batches = subdir_batches[:-num_test_files]

            self.X_files.extend(selected_batches)
            self.y_files.extend([str(f).replace('X_', 'y_') for f in selected_batches])

        if self.X_files:
            self.samples_per_batch = np.load(self.X_files[0]).shape[0]
        else:
            raise ValueError("No valid data found in the provided directory structure.")

    def __len__(self):
        return len(self.X_files) * self.samples_per_batch

    def __getitem__(self, idx, discretize_target=True):
        batch_idx = idx // self.samples_per_batch
        sample_idx = idx % self.samples_per_batch

        src = np.load(self.X_files[batch_idx], mmap_mode='r')[sample_idx].copy()
        tgt = np.load(self.y_files[batch_idx], mmap_mode='r')[sample_idx].copy()

        if discretize_target:
            tgt = np.array(discretize(tgt, np.linspace(3, 14, 256)))

        src = torch.from_numpy(src).float()
        src = torch.log1p(src)

        tgt = torch.from_numpy(tgt).long() + 2
        tgt = torch.cat([torch.tensor([1]), tgt])

        return src, tgt


'''
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm  # Progress bar

class MultiDirLazyDataset(Dataset):
    def __init__(self, root_dir: str, split: Literal['train', 'test'] = 'train', test_ratio: float = 0.2):
        """
        A dataset that supports a hierarchy of subdirectories with a percentage-based split.

        Args:
            root_dir (str): Root directory containing subdirectories.
            split (Literal['train', 'test']): Whether to use training or testing data.
            test_ratio (float): Percentage of files in each subdirectory to use for testing.
        """
        super().__init__()
        self.root_dir = Path(root_dir).resolve()  # Ensure absolute path
        self.split = split
        self.test_ratio = test_ratio
        self.X_files = []
        self.y_files = []

        # Recursively find all X_*.npy files
        all_files = sorted(self.root_dir.rglob('X_*.npy'), key=lambda x: (x.parent, int(x.stem.split('_')[1])))

        # Group files by parent directory
        grouped_files = defaultdict(list)
        for file in all_files:
            grouped_files[file.parent].append(file)

        # Check if all files are loadable and corresponding y_*.npy exist
        print("🔍 Checking dataset integrity...")
        for subdir, subdir_batches in tqdm(grouped_files.items(), desc="Checking Directories", unit="dir"):
            for x_file in subdir_batches:
                y_file = x_file.parent / x_file.name.replace("X_", "y_")

                # Ensure y_ file exists
                if not y_file.exists():
                    raise FileNotFoundError(f"Missing corresponding file: {y_file}")

                # Try loading the files
                try:
                    x_data = np.load(x_file, mmap_mode="r")
                    y_data = np.load(y_file, mmap_mode="r")

                    # Ensure same number of samples in X and y
                    if x_data.shape[0] != y_data.shape[0]:
                        raise ValueError(f"Shape mismatch in {x_file} and {y_file}: "
                                         f"X shape {x_data.shape}, Y shape {y_data.shape}")

                except Exception as e:
                    raise RuntimeError(f"Failed to load: {x_file} or {y_file}. Error: {e}")

        print("✅ Dataset integrity check passed! All files are correctly formatted.")

        # Select train/test splits
        for subdir, subdir_batches in grouped_files.items():
            num_test_files = max(1, int(len(subdir_batches) * test_ratio)) if test_ratio > 0 else 0

            if self.split == 'test':
                selected_batches = subdir_batches[-num_test_files:]
            else:
                selected_batches = subdir_batches[:-num_test_files]

            self.X_files.extend(selected_batches)
            self.y_files.extend([f.parent / f.name.replace('X_', 'y_') for f in selected_batches])

        if not self.X_files:
            raise ValueError(f"No valid {self.split} data found in {self.root_dir}. Check test_ratio or directory structure.")

        self.samples_per_batch = np.load(self.X_files[0]).shape[0]

    def __len__(self):
        return len(self.X_files) * self.samples_per_batch

    def __getitem__(self, idx, discretize_target=True):
        batch_idx = idx // self.samples_per_batch
        sample_idx = idx % self.samples_per_batch

        src = np.load(self.X_files[batch_idx], mmap_mode='r')[sample_idx].copy()
        tgt = np.load(self.y_files[batch_idx], mmap_mode='r')[sample_idx].copy()

        if discretize_target:
            tgt = np.array(discretize(tgt, np.linspace(3, 14, 256)))

        src = torch.from_numpy(src).float()
        src = torch.log1p(src)

        tgt = torch.from_numpy(tgt).long() + 2
        tgt = torch.cat([torch.tensor([1]), tgt])

        return src, tgt
'''