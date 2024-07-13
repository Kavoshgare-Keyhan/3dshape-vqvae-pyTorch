import os, h5py, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Shapes3DDataset(Dataset):
    def __init__(self, data, transform=None):
        assert os.path.exists(self.data_path), f"images path {self.data_path} does not exist"
        self.images = data['images']
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
# Custom collate function to handle batch loading
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    # labels = torch.stack([item[1] for item in batch])
    return images, None

def folding(path, k, num_fold):
    assert os.path.exists(path), f"images path {path} does not exist"
    data = h5py.File(path, 'r')
    fold_size = len(data)//k
    ind_start = k * fold_size
    if k==num_fold: ind_end = len(data)
    else: ind_end = (k+1) * fold_size
    val_set = data['images'][ind_start:ind_end]
    train_set = np.concatenate((data['images'][:ind_start], data['images'][ind_end:]), axis=0)
    return train_set, val_set


def load_data(dataset):
    dataset = Shapes3DDataset(data=dataset)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    return dataloader