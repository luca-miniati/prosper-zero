import pandas as pd
import torch

class ListingsDataset:
    def __init__(self, dataset_path, dataset_type, num_rows=None):
        self.data = pd.read_csv(dataset_path)
        if num_rows:
            self.data = self.data.iloc[:num_rows]
        self.type = dataset_type
        self.input_dimension = self.data.shape[-1] - 1

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if self.type == "test":
            return torch.from_numpy(self.data.iloc[index].values).float()
        elif self.type == "train":
            return (torch.from_numpy(self.data.iloc[index].values).float()[:-1] , torch.from_numpy(self.data.iloc[index].values).float()[-1][None] )
