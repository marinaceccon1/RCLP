from torch.utils.data import Dataset
import random

class MergedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.total_size = len(dataset1) + len(dataset2)
        self.indices = list(range(self.total_size))
        random.shuffle(self.indices)  # Shuffle the indices

    def __getitem__(self, idx):
        idx = self.indices[idx]  # Use shuffled index
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            idx -= len(self.dataset1)
            return self.dataset2[idx]

    def __len__(self):
        return self.total_size
