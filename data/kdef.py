import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from .datagen import DataGenerator, DatasetOperator
from tqdm import tqdm
class KDEFDataset(DatasetOperator):

    def __init__(self, z, tau1, tau2):
        super().__init__(tau1, tau2)
        self.z = z



class KDEFDataGen(DataGenerator):

    def __init__(self, type, samples, data_seed, path_to_data, p, is_sequential):
        super().__init__(type, samples, data_seed)
        tfms = [
            transforms.Resize((32,32)),
             transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           
        ]
        transform = transforms.Compose(tfms)

        tset = torchvision.datasets.ImageFolder(path_to_data, transform=transform)
        loader = torch.utils.data.DataLoader(tset, batch_size=128, shuffle=True, num_workers=8)

        all_class_c0 = []
        all_class_c1 = []
        c0 = tset.classes.index("positive")
        c1 = tset.classes.index("negative")

        for data, target in tqdm(loader):
            all_class_c0.append(data[target == c0])
            all_class_c1.append(data[target == c1])
        if type=="type2":
            # Flatten and store the image tensor
            self.X = 1.0 * torch.flatten(torch.cat(all_class_c0), 1).float() 
            self.Y = 1.0 * torch.flatten(torch.cat(all_class_c1), 1).float() 
        if type=="type12":
            # Flatten and store the image tensors
            self.X = 1.0 * torch.flatten(torch.cat(all_class_c0), 1).float() 
            self.Y = 1.0 * torch.flatten(torch.cat(all_class_c0).clone(), 1).float() 
        if type=="type11":
            self.X = 1.0 * torch.flatten(torch.cat(all_class_c1), 1).float() 
            self.Y = 1.0 * torch.flatten(torch.cat(all_class_c1).clone(), 1).float() 
 
        # Create subsets based on the 'samples' parameter
        total_samples = min(self.X.shape[0], self.Y.shape[0])
            
        # p_size = int(p * total_samples)
        # self.X = self.X[:total_samples, ...]
        # self.Y = self.Y[:total_samples, ...]
        # # Swap half of the images between the two classes
        # datax_ = self.X[:p_size, :].clone()
        # datay_ = self.Y[:p_size, :].clone()
        # self.X[:p_size, :] = datay_.clone()
        # self.Y[:p_size, :] = datax_.clone()
            
        idx = torch.randperm(self.X.shape[0])
        self.X = self.X[idx, ...]
        idx = torch.randperm(self.Y.shape[0])
        self.Y = self.Y[idx, ...]
        self.z = torch.stack([self.X[:total_samples, ...], self.Y[:total_samples, ...]], dim=2)

        if is_sequential:
            num_chunks = int(total_samples / samples)
            self.index_sets_seq = np.array_split(range(total_samples), num_chunks)
        else:
            self.index_sets_seq = np.array_split(range(total_samples), [samples, samples+int(0.2*samples), samples+2*int(0.2*samples)])
    
    def generate(self, seed, tau1, tau2) -> Dataset:
        """
        Generate a subset of the data based on the provided seed.

        Args:
        - seed (int): Seed to determine which subset of the data to use.
        - tau1 (float): Tau parameter 1.
        - tau2 (float): Tau parameter 2.

        Returns:
        - Dataset: A subset of the KDEF dataset.
        """
        ind = self.index_sets_seq[seed]

        return KDEFDataset(self.z[ind, ...], tau1, tau2)
            




