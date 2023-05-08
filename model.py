import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import FFDataset, SoftmaxDataset


class FFLayer(nn.Module):
    def __init__(self, in_dims, out_dims, threshold, lr, batch_size, epochs, device):
        super().__init__()

        self.linear = nn.Linear(in_dims, out_dims)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.threshold = threshold
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()

    def normalize(self, x):
        return x / (x.norm(p=2, dim=1, keepdim=True) + 1e-7)

    def forward(self, x):
        x = self.normalize(x)
        x = F.relu(self.linear(x))
        return x

    def train(self, dataset):
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        bar = tqdm(range(self.epochs))
        loss_history = []

        for _ in bar:
            epoch_loss = 0
            for x_pos, x_neg in dataloader:
                g_pos = self.forward(x_pos).pow(2).sum(1) - self.threshold
                g_neg = self.forward(x_neg).pow(2).sum(1) - self.threshold

                loss = self.criterion(torch.cat([g_pos, g_neg]),
                                      torch.cat([torch.ones((g_pos.shape[0]), device=self.device),
                                                 torch.zeros((g_neg.shape[0]), device=self.device)]))

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            bar.set_postfix({'Loss': epoch_loss})
            loss_history.append(epoch_loss)

        x_pos, x_neg = dataset[:][0], dataset[:][1]
        x_pos = self.forward(x_pos)
        x_neg = self.forward(x_neg)

        return x_pos.detach(), x_neg.detach(), loss_history


class FFNN(nn.Module):
    def __init__(self, dims, threshold, lr, batch_size, epochs, device):
        super().__init__()

        self.fflayers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.fflayers.append(
                FFLayer(dims[d], dims[d+1], threshold, lr, batch_size, epochs, device))

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg

        pos_outputs = []
        total_loss_history = []

        for i, layer in enumerate(self.fflayers):
            print(f'Training {i} ff-layer')
            dataset = FFDataset(h_pos, h_neg)

            h_pos, h_neg, loss_history = layer.train(dataset)

            pos_outputs.append(h_pos)
            total_loss_history.append(loss_history)

        return pos_outputs, total_loss_history

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.fflayers):
            x = layer(x)
            outputs.append(x)
        return outputs


class SoftmaxLayer(nn.Module):
    def __init__(self, in_dim, lr, batch_size, epochs):
        super().__init__()

        self.linear = nn.Linear(in_dim, 10)

        self.criterion = nn.CrossEntropyLoss()

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

        self.batch_size = batch_size
        self.epochs = epochs

    def forward(self, x):
        return self.linear(x)

    def train(self, x, y):
        dataset = SoftmaxDataset(x, y)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        bar = tqdm(range(self.epochs))
        loss_history = []

        for _ in bar:
            epoch_loss = 0
            for x, y in dataloader:
                x_ = self.forward(x)
                loss = self.criterion(x_, y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            bar.set_postfix({'Loss': epoch_loss})
            loss_history.append(epoch_loss)

        return loss_history
