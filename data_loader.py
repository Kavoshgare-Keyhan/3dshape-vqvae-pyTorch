import os, h5py, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ShuffleData:
    def __init__(self, data_path):
        self.path = data_path
        assert os.path.exists(self.path), f"images path {self.path} does not exist"
        self.data = h5py.File(self.path, 'r')
        self.images = np.array(self.data['images'])
        self.labels = np.array(self.data['labels'])

    def shuffle(self):
        assert len(self.images) == len(self.labels)
        p = np.random.permutation(len(self.images))
        self.images, self.labels = self.images[p], self.labels[p]
    
    def save_shuffle(self, path, output_file):
        assert os.path.exists(path), 'Path %s does not exist' % path
        with h5py.File(os.path.join(path, output_file), 'w') as f:
            f.create_dataset('images', data=self.images)
            f.create_dataset('labels', data=self.labels)


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

class Shapes3DTestDataset(Dataset):
    def __init__(self, data, transform=None):
        self.images = data
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


def load_data(data_path,sample_data,shuffle_data=True):
    dataset = Shapes3DDataset(path=data_path)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=shuffle_data, sampler=sample_data,num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    return data_loader

def load_test(test_data):
    dataset = Shapes3DTestDataset(test_data)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    return dataset, data_loader

if __name__ =='main':
    # Load data
    ## data_path = '/home/mohsen/Desktop/Academia/RUB Research Projects/INI/data/3dshapes/3dshapes.h5'
    ## sample_data = torch.utils.data.SubsetRandomSampler(np.arange(1000))  # Use a subset of data for demonstration

    ## data_loader = load_data(data_path, sample_data)

    # Shuffle data
    shuffle_data = ShuffleData(data_path='/home/mohsen/Desktop/Academia/RUB Research Projects/INI/data/3dshapes/3dshapes.h5')
    shuffle_data.shuffle()
    shuffle_data.save_shuffle(path='/home/mohsen/Desktop/Academia/RUB Research Projects/INI/data/3dshapes/', output_file='3dshapes_shuffled.h5')

    print('Data loading and shuffling completed.')
