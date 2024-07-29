import os, h5py, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Shapes3DDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        assert os.path.exists(self.path), f"images path {self.path} does not exist"
        self.data = h5py.File(self.path, 'r')
        self.images = self.data['images']
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

    def __del__(self):
        self.data.close()
 
# Custom collate function to handle batch loading
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    # labels = torch.stack([item[1] for item in batch])
    return images, None


def load_data(data_path,sample_data):
    dataset = Shapes3DDataset(path=data_path)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True, sampler=sample_data,num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    return data_loader