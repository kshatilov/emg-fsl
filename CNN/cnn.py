import os, sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#'/home/user/example/parent/child'
current_path = os.path.abspath('.')
#'/home/user/example/parent'
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from DAO.DataLoader import DataLoader
import numpy as np

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (10, 1))
        self.conv2 = nn.Conv2d(64, 128, (5, 8))
        self.fc1 = nn.Linear(11136, 500)
        self.fc2 = nn.Linear(500, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNN2(nn.Module):

    def __init__(self, input_data_shape, num_labels):
        super(CNN2, self).__init__()
        # conv argument: in_channel, out_channel, kernel_size,
        print('input_data_shape: ', input_data_shape)
        print('num_labels: ', num_labels)

        div = 4
        self.conv1 = nn.Conv2d(input_data_shape[1], 64, (input_data_shape[2] // div, 1))
        self.conv2 = nn.Conv2d(64, 128, (div, input_data_shape[3]))
        # self.fc1 = nn.Linear(128 * div * input_data_shape[3], 500)
        self.fc1 = nn.Linear(4608, 500)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, num_labels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        self.input, self.target = X, y
        self.transform, self.target_transform = transform, target_transform

    def __getitem__(self, index):
        X, y = self.input[index], self.target[index]
        if self.transform:
            X = self.transform(X)

        if self.target_transform:
            y = self.target_transform(y)

        return X, y

    def __len__(self):
        return len(self.input)

def get_data_loader(dataset, batch_size, shuffle=True, cuda=False, collate_fn=None, drop_last=False, augment=False):

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                collate_fn=(collate_fn or torch.utils.data.dataloader.default_collate),
                drop_last=drop_last, **({'num_workers': 0, 'pin_memory':True} if cuda else {}))

def load_data():
    dl = DataLoader()
    dl.load_files(participants=[3], scenarios=[1])
    x, y = dl.get_xy(window_length=50, overlap=0.8)
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=True)

    return train_X, test_X, train_y, test_y

def load_data_non_intertwined(train_ratio=0.9, overlap=0.8):
    dl = DataLoader()
    window_size, num_features = 50, 8
    dl.load_files(participants=[3], scenarios=[1])
    train_X, train_y, test_X, test_y = dl.get_xy_train_test(window_length=window_size, overlap=overlap, train_ratio=train_ratio)

    # normalize data
    train_X = StandardScaler().fit_transform(np.array(train_X).reshape(-1, window_size * num_features) )
    test_X = StandardScaler().fit_transform(np.array(test_X).reshape(-1, window_size * num_features))

    train_X = train_X.reshape(-1, window_size, num_features)
    test_X = test_X.reshape(-1, window_size, num_features)

    return train_X, test_X, train_y, test_y


def reshape(x):
    return x.reshape((x.shape[0],) + (1,) + x.shape[1:])


def convert_2_torch(train_X, test_X, train_y, test_y):
    train_X = reshape(np.asarray(train_X, dtype=np.double))
    test_X = reshape(np.asarray(test_X, dtype=np.double))
    train_y = reshape(np.asarray(train_y))
    test_y = reshape(np.asarray(test_y))

    print('train_X.shape: ', train_X.shape)
    print('test_X.shape: ', test_X.shape)
    print('train_y.shape: ', train_y.shape)
    print('test_y.shape: ', test_y.shape)

    return \
        torch.from_numpy(train_X), torch.from_numpy(test_X), \
        torch.from_numpy(train_y).long(), torch.from_numpy(test_y).long()


def train():

    # train_X, test_X, train_y, test_y = load_data()
    train_X, test_X, train_y, test_y = load_data_non_intertwined(train_ratio=0.8)
    train_X, test_X, train_y, test_y = convert_2_torch(train_X, test_X, train_y, test_y)
    train_y -= 1
    test_y -= 1

    train_dataset = MyDataset(train_X, train_y)
    # test_dataset = MyDataset(test_X, test_y)

    cnn = CNN2(input_data_shape=list(test_X.size()), num_labels=len(list(torch.unique(test_y))))
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    epochs = 20
    loss = 0

    dataloader = get_data_loader(train_dataset, batch_size)

    for epoch in range(epochs):
        for batch, (_input, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = cnn(_input.float())
            loss = criterion(output, target.squeeze(1).long())
            loss.backward()
            optimizer.step()

        # validate
        y_pred = cnn(test_X.float())
        y_pred = y_pred.detach().numpy()
        y_pred = [np.argmax(y, axis=None, out=None) for y in y_pred]
        val_accuracy = metrics.accuracy_score(test_y, y_pred)
        print(f"Epoch: {epoch+1}: loss {loss} validation accuracy {val_accuracy}")


    # for epoch in range(epochs):
    #     for batch in range(len(train_X) // batch_size):
    #         start = batch * batch_size
    #         _input = train_X[start:start + batch_size]
    #         target = train_y[start:start + batch_size]
    #
    #         optimizer.zero_grad()
    #         output = cnn(_input.float())
    #         loss = criterion(output, target.squeeze(1).long())
    #         loss.backward()
    #         optimizer.step()
    #
    #     # validate
    #     y_pred = cnn(test_X.float())
    #     y_pred = y_pred.detach().numpy()
    #     y_pred = [np.argmax(y, axis=None, out=None) for y in y_pred]
    #     val_accuracy = metrics.accuracy_score(test_y, y_pred)
    #     print(f"Epoch: {epoch}: loss {loss} validation accuracy {val_accuracy}")


if __name__ == '__main__':
    # set random seed
    seed = 42
    cuda = False
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    train()
