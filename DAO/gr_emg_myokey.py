#!/usr/bin/env python3

import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

# load and segment data (data are in numpy format)

# split train and valid dataset if necessary (convert numpy to tensor format when return)


class DataLoader:
    def __init__(self, path=None) -> None:
        self.participants_count = 12
        self.scenarios_count = 3
        self.gestures_count = 5
        super().__init__()
        self.raw_data = {}
        for i in range(self.gestures_count):
            self.raw_data[str(i + 1)] = []
        self.path = path

    # None for all participants/scenarios (default behavior)
    def load_files(self, participants=None, scenarios=None, gestures=None):

        if participants is None:
            participants = list(range(1, self.participants_count + 1))
        if scenarios is None:
            scenarios = list(range(1, self.scenarios_count + 1))
        if gestures is None:
            gestures = list(range(1, self.gestures_count + 1))

        for participant in participants:
            for scenario in scenarios:
                if self.path is None:
                    _dir = f"../Data/p{participant}_s{scenario}/"
                else:
                    _dir = self.path + f"/Data/p{participant}_s{scenario}/"
                print(_dir)
                for file in os.listdir(_dir):
                    label = int(file[1])
                    if label in gestures:
                        print(file)
                        record = np.loadtxt(_dir + file)
                        self.raw_data[str(label)].append(record[:, :8])

    def get_xy(self, window_length=100, overlap=0.5):
        x = []
        y = []

        for label in self.raw_data:
            if self.raw_data[label] is None:
                continue
            for record in self.raw_data[label]:
                start = 0
                while start + window_length <= len(record):
                    datapoint = record[start:start + window_length]
                    x.append(datapoint)
                    y.append(label)
                    start += window_length - int(overlap * window_length)

        return torch.Tensor(x), torch.Tensor(y).long()

class GrEmgMyokey(Dataset):
    """

    [[Source]]()

    **Description**

    This class provides an interface to the GrEmgMyokey dataset.

    The Omniglot dataset was introduced by Lake et al., 2015.
    Omniglot consists of 1623 character classes from 50 different alphabets, each containing 20 samples.
    While the original dataset is separated in background and evaluation sets,
    this class concatenates both sets and leaves to the user the choice of classes splitting
    as was done in Ravi and Larochelle, 2017.
    The background and evaluation splits are available in the `torchvision` package.

    **References**

    1. Kwon et al. 2020. "MyoKey: ... " PerCom.
    2. Shatilov et al. 2020. “MyoBoard: ...” IMWUT.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**
    ~~~python
    emgMyokey = l2l.sensing.datasets.GrEmgMyokey(root='./data',
                                                transform=transforms.Compose([
                                                    transforms.Resize(28, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=True)
    emgMyokey = l2l.data.MetaDataset(emgMyokey)
    ~~~

    """

    def __init__(self,
                 root,
                 mode='train',
                 participants=None,
                 scenarios=None,
                 gestures=None,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(GrEmgMyokey, self).__init__()
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            raise ('ValueError', 'Needs to check whether the path exsits')
        self.transform = transform
        self.target_transform = target_transform

        # Set up both the background and eval dataset
        dl = DataLoader(root)
        if participants is not None:
            dl.load_files(participants, scenarios, gestures)
        else:
            if mode == 'train':
                dl.load_files(participants=list(range(1, 11)))
            elif mode == 'validation':
                dl.load_files(participants=[11])
            elif mode == 'test':
                dl.load_files(participants=[12])
            else:
                raise ('ValueError', 'Needs to check mode argument')

        self.X, self.y = dl.get_xy(window_length=100, overlap=0.5)
        print('data size of ' + mode + ' is as follows.')
        print('X.shape: ', self.X.shape)
        print('y.shape: ', self.y.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        if self.transform:
            X = self.transform(X)

        if self.target_transform:
            y = self.target_transform(y)

        return X, y
