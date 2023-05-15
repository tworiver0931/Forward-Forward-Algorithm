import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import itertools
from tqdm import tqdm

from utils import FFDataset, SoftmaxDataset


class FFLayer(nn.Module):
    def __init__(self, in_dims, out_dims, threshold, lr, batch_size, epochs, dropout, device):
        super().__init__()

        self.linear = nn.Linear(in_dims, out_dims)
        self.dropout = nn.Dropout(dropout)

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
        return F.relu(self.dropout(self.linear(x)))

    def train(self, h_pos, h_neg):
        dataset = FFDataset(h_pos, h_neg)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        bar = tqdm(range(self.epochs))
        loss_history = []

        for _ in bar:
            epoch_loss = 0
            for h_pos_batch, h_neg_batch in dataloader:
                g_pos = self.forward(h_pos_batch).pow(2).sum(1)
                g_neg = self.forward(h_neg_batch).pow(2).sum(1)

                loss = self.criterion(torch.cat([g_pos-self.threshold, g_neg-self.threshold]),
                                      torch.cat([torch.ones((g_pos.shape[0]), device=self.device),
                                                 torch.zeros((g_neg.shape[0]), device=self.device)]))

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            bar.set_postfix({'Loss': epoch_loss})
            loss_history.append(epoch_loss)

        h_pos = self.forward(h_pos)
        h_neg = self.forward(h_neg)

        return h_pos.detach(), h_neg.detach(), loss_history

    def predict(self, x):
        x = self.normalize(x)
        return F.relu(self.linear(x))


class FFNN(nn.Module):
    def __init__(self, dims, threshold, lr, batch_size, epochs, dropout, device):
        super().__init__()

        self.fflayers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.fflayers.append(
                FFLayer(dims[d], dims[d+1], threshold, lr, batch_size, epochs, dropout, device))

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg

        pos_outputs = []
        total_loss_history = []

        for i, layer in enumerate(self.fflayers):
            print(f'Training {i} ff-layer')
            h_pos, h_neg, loss_history = layer.train(h_pos, h_neg)

            pos_outputs.append(h_pos)
            total_loss_history.append(loss_history)

        return pos_outputs, total_loss_history

    def predict(self, x):
        outputs = []
        for i, layer in enumerate(self.fflayers):
            x = layer.predict(x)
            outputs.append(x)
        return outputs


class LocalConv(nn.Module):
    def __init__(self,
                 out_channels,
                 input_shape,
                 kernel_size,
                 strides,
                 bias=False,
                 ):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.bias = bias

        input_row, input_col = input_shape[1:]
        input_filter = input_shape[0]

        output_row = self.output_length(
            input_row, self.kernel_size[0], self.strides[0])
        output_col = self.output_length(
            input_col, self.kernel_size[1], self.strides[1])

        self.output_shape = (output_row, output_col)

        self.kernel_shape = (output_row * output_col,
                             self.kernel_size[0] *
                             self.kernel_size[1] * input_filter,
                             self.out_channels
                             )

        self.kernel = nn.Parameter(torch.randn(self.kernel_shape))

        if self.bias:
            self.bias_shape = (self.out_channels, output_row, output_col)
            self.bias = nn.Parameter(torch.randn(self.bias_shape))

    def output_length(self, input_length, kernel_size, stride):
        output_length = input_length - kernel_size + 1
        return (output_length + stride - 1) // stride

    def forward(self, inputs):
        feature_dim = self.kernel_shape[1]
        channels_out = self.kernel_shape[-1]

        xs = []
        output_axes_ticks = [range(axis_max) for axis_max in self.output_shape]
        for position in itertools.product(*output_axes_ticks):
            slices = [slice(None)]
            slices.append(slice(None))
            slices.extend(
                slice(
                    position[d] * self.strides[d],
                    position[d] * self.strides[d] + self.kernel_size[d],
                )
                for d in [0, 1]
            )
            xs.append(torch.reshape(inputs[slices], (1, -1, feature_dim)))

        x_aggregate = torch.cat(xs)

        output = torch.bmm(x_aggregate, self.kernel)
        output = torch.reshape(
            output, (self.output_shape + (-1, channels_out)))
        output = torch.permute(output, (2, 3, 0, 1))

        if self.bias is not False:
            output = output + self.bias.unsqueeze(0)
        return output


class FFRFLayer(nn.Module):
    def __init__(self,
                 out_channel,
                 input_shape,
                 kernel_size,
                 strides,
                 threshold,
                 lr,
                 batch_size,
                 epochs,
                 dropout,
                 device
                 ):

        super().__init__()

        self.local_conv = LocalConv(
            out_channel, input_shape, kernel_size, strides, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.peer_norm = nn.BatchNorm2d(out_channel)

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.threshold = threshold
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = x / (x.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-7)
        return F.relu(self.dropout(self.peer_norm(self.local_conv(x))))

    def train(self, h_pos, h_neg):
        dataset = FFDataset(h_pos, h_neg)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        bar = tqdm(range(self.epochs))
        loss_history = []

        for _ in bar:
            epoch_loss = 0
            for h_pos_batch, h_neg_batch in dataloader:
                g_pos = self.forward(h_pos_batch).pow(2).mean(dim=(1, 2, 3))
                g_neg = self.forward(h_neg_batch).pow(2).mean(dim=(1, 2, 3))

                loss = self.criterion(torch.cat([g_pos-self.threshold, g_neg-self.threshold]),
                                      torch.cat([torch.ones((g_pos.shape[0]), device=self.device),
                                                 torch.zeros((g_neg.shape[0]), device=self.device)]))

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            bar.set_postfix({'Loss': epoch_loss})
            loss_history.append(epoch_loss)

        h_pos = self.forward(h_pos)
        h_neg = self.forward(h_neg)

        return h_pos.detach(), h_neg.detach(), loss_history

    def predict(self, x):
        x = x / (x.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-7)
        return F.relu(self.peer_norm(self.local_conv(x)))


class FFRFNN(nn.Module):
    def __init__(self,
                 out_channel_list,
                 input_shape_list,
                 kernel_size_list,
                 strides_list,
                 threshold,
                 lr,
                 batch_size,
                 epochs,
                 dropout,
                 device
                 ):

        super().__init__()

        self.fflayers = nn.ModuleList()
        for d in range(len(out_channel_list)):
            self.fflayers.append(
                FFRFLayer(out_channel_list[d], input_shape_list[d], kernel_size_list[d], strides_list[d], threshold, lr, batch_size, epochs, dropout, device))

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg

        pos_outputs = []
        total_loss_history = []

        for i, layer in enumerate(self.fflayers):
            print(f'Training {i} ff-layer')
            h_pos, h_neg, loss_history = layer.train(h_pos, h_neg)

            pos_outputs.append(h_pos)
            total_loss_history.append(loss_history)

        return pos_outputs, total_loss_history

    def predict(self, x):
        outputs = []
        for i, layer in enumerate(self.fflayers):
            x = layer.predict(x)
            outputs.append(x)
        return outputs


class SoftmaxLayer(nn.Module):
    def __init__(self, in_dim, out_dim, lr, batch_size, epochs):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)

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
        history = {'loss': [], 'acc': []}

        for _ in bar:
            epoch_loss = 0
            epoch_acc = 0
            for x, y in dataloader:
                x_ = self.forward(x)
                loss = self.criterion(x_, y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                epoch_loss += loss.item()
                epoch_acc += (torch.argmax(x_, 1) == y).float().mean().item()

            epoch_loss /= len(dataloader)
            epoch_acc /= len(dataloader)

            bar.set_postfix({'Loss': epoch_loss, 'Acc': epoch_acc})
            history['loss'].append(epoch_loss)
            history['acc'].append(epoch_acc)

        return history
