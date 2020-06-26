import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
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


def load_data():
    dl = DataLoader()
    dl.load_files(participants=[3], scenarios=[1])
    x, y = dl.get_xy(window_length=100, overlap=0.6)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    return x_train, x_test, y_train, y_test


def reshape(x):
    return x.reshape((x.shape[0],) + (1,) + x.shape[1:])


def convert_2_torch(x_train, x_test, y_train, y_test):
    x_train = reshape(np.asarray(x_train, dtype=np.double))
    x_test = reshape(np.asarray(x_test, dtype=np.double))
    y_train = reshape(np.asarray(y_train))
    y_test = reshape(np.asarray(y_test))

    print(x_train.shape)
    return \
        torch.from_numpy(x_train), torch.from_numpy(y_train).long(), \
        torch.from_numpy(x_test), torch.from_numpy(y_test).long()


def train():
    cnn = CNN()
    optimizer = optim.Adam(cnn.parameters(), lr=0.003)
    x_train, x_test, y_train, y_test = load_data()
    x_train, x_test, y_train, y_test = convert_2_torch(x_train, x_test, y_train, y_test)
    y_train -= 1
    y_test -= 1

    criterion = nn.CrossEntropyLoss()
    batch_size = 15
    epochs = 10
    loss = 0

    for epoch in range(epochs):
        for batch in range(len(x_train) // batch_size):
            start = batch * batch_size
            _input = x_train[start:start + batch_size]
            target = y_train[start:start + batch_size]

            optimizer.zero_grad()
            output = cnn(_input.float())
            loss = criterion(output, target.squeeze(1).long())
            loss.backward()
            optimizer.step()

        # validate
        y_pred = cnn(x_test.float())
        y_pred = y_pred.detach().numpy()
        y_pred = [np.argmax(y, axis=None, out=None) for y in y_pred]
        val_accuracy = metrics.accuracy_score(y_test, y_pred)
        print(f"Epoch: {epoch}: loss {loss} validation accuracy {val_accuracy}")


if __name__ == '__main__':
    train()
