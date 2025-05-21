import zarr
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class scDataSet(Dataset):

    def __init__(self, path):
        self.sc_store = zarr.open(path, mode='r')
        self.attrs = self.sc_store.attrs.asdict()
        self.position_normalizations = self.attrs['normalization_params']
        self.labels = self.attrs['labels']
        self.mean_0 = np.mean(np.asarray([a['mean']['0'] for _, a in self.position_normalizations.items()]))
        self.mean_1 = np.mean(np.asarray([a['mean']['1'] for _, a in self.position_normalizations.items()]))
        self.mean_2 = np.mean(np.asarray([a['mean']['2'] for _, a in self.position_normalizations.items()]))
        self.std_0 = np.mean(np.asarray([a['std']['0'] for _, a in self.position_normalizations.items()]))
        self.std_1 = np.mean(np.asarray([a['std']['1'] for _, a in self.position_normalizations.items()]))
        self.std_2 = np.mean(np.asarray([a['std']['2'] for _, a in self.position_normalizations.items()]))

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x).float()),
            transforms.Normalize(mean=[self.mean_0, self.mean_1, self.mean_2], std=[self.std_0, self.std_1, self.std_2])
                ])
        return
    
    def __len__(self):
        return self.attrs['num_cells']
    
    def __getitem__(self, index):
        img = np.squeeze(np.asarray(self.sc_store[self.attrs['labels'][str(index)]]))
        label = self.attrs['labels'][str(index)]
        
        return self.transform(img), label