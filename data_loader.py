import os, h5py, torch
import numpy as np
from torch.utils.data import Dataset

class Shapes3DDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        assert os.path.exists(self.data_path), f"images path {self.data_path} does not exist"
        self.dataset = h5py.File(self.data_path, 'r')
        self.images = self.dataset['images']
        ## self.labels = self.file['labels']
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        ## label = self.labels[idx]

        # Normalize the image to range [-1, 1]
        image = image.astype(np.float32) / 255.0 * 2 - 1

        # Transpose image to fit PyTorch's [C, H, W] format
        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float32)
        ## label = torch.tensor(label, dtype=torch.float32)

        return image, None
