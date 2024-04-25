import numpy as np
from torch.utils.data import Dataset
import torch
from .datagen import DataGenerator, DatasetOperator
import torchvision
from torchvision import transforms
import pickle

class DCGANMNISTDataset(DatasetOperator):

    def __init__(self, z, tau1, tau2):
        super().__init__(tau1, tau2)
        self.z = z


class DCGANMNISTDataGen(DataGenerator):


    def __init__(self, type, samples, data_seed, file_path1, file_path2, p, is_sequential):
        super().__init__(type, samples, data_seed)
        self.p_size = p
        
        # Define transformations for the MNIST dataset
        transforms_MNIST = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]
        )

        # Load the MNIST dataset
        mnist = torchvision.datasets.MNIST(file_path1, train=True, transform=transforms_MNIST, download=True)
        loader = torch.utils.data.DataLoader(mnist, batch_size=len(mnist), shuffle=True)
        data_true = next(iter(loader))
            
        data_gen = pickle.load(open(file_path2, 'rb'))
        data_gen = torch.from_numpy(data_gen[0])

        idx = torch.randperm(data_gen.shape[0])
        data_gen = data_gen[idx, ...]
        self.data_gen = torch.flatten(data_gen, 1).float()

        n_samples = len(data_true)
        self.X = 1.0 * torch.flatten(data_true[:(n_samples // 2), ...], 1).float()
        self.Y = 1.0 * torch.flatten(data_true[(n_samples // 2):, ...], 1).float()
        total_samples = min(min(self.X.shape[0], self.Y.shape[0]),data_gen.shape[0])


        if is_sequential:
            num_chunks = int(total_samples / samples)
            self.index_sets_seq = np.array_split(range(total_samples), num_chunks)
        else:
            self.index_sets_seq = np.array_split(range(total_samples), [samples, samples + int(0.2 * samples),
                                                                        samples + 2 * int(0.2 * samples)])

    def generate(self, seed, tau1, tau2) -> Dataset:
        """
        Generate a subset of the data based on the provided seed.

        Args:
        - seed (int): Seed to determine which subset of the data to use.
        - tau1 (float): Tau parameter 1.
        - tau2 (float): Tau parameter 2.

        Returns:
        - Dataset: A subset of the DCGAN- MNIST dataset.
        """
        ind = self.index_sets_seq[seed]
        x = self.X[ind, ...]
        y = self.Y[ind, ...]
        data_gen = self.data_gen[ind,...]
        p_size = int(self.p_size*len(x))
        y[:p_size, :] = data_gen[:p_size, :].clone()
        z = torch.stack([x, y], dim=2)
        return DCGANMNISTDataset(z, tau1, tau2)
