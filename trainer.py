import torch
from model import FFNN, FFRFNN, SoftmaxLayer
from utils import create_neg


class FCTrainer:
    def __init__(self,
                 train_set,
                 ff_dims,
                 out_dim,
                 threshold=1,
                 lr=0.001,
                 batch_size=512,
                 epochs=100,
                 dropout=0,
                 device='cuda:0'
                 ):

        self.device = device

        self.x_pos, self.y = train_set.data, train_set.targets

        # normalize data
        self.x_pos = self.x_pos.type(torch.float32) / 255
        self.mean, self.std = self.x_pos.mean(), self.x_pos.std()
        self.x_pos = (self.x_pos - self.mean) / self.std

        self.x_neg = create_neg(self.x_pos)

        self.x_pos = self.x_pos.reshape(self.x_pos.shape[0], -1)
        self.x_neg = self.x_neg.reshape(self.x_neg.shape[0], -1)

        self.x_pos = self.x_pos.to(device)
        self.x_neg = self.x_neg.to(device)
        self.y = self.y.to(device)

        self.ffnn = FFNN(ff_dims, threshold, lr, batch_size,
                         epochs, dropout, device)
        self.softmaxlayer = SoftmaxLayer(
            sum(ff_dims[2:]), out_dim, lr, batch_size, epochs)

        self.ffnn.to(device)
        self.softmaxlayer.to(device)

        self.ffnn_his = None
        self.softmax_his = None

    def train(self):
        print('\n[Start training FF-Layers]')
        pos_outputs, self.ffnn_his = self.ffnn.train(self.x_pos, self.x_neg)

        # use positive data outputs except first ff-layer
        pos_cat = torch.cat(pos_outputs[1:], dim=1)

        print('\n[Start training Softmax-Layer]')
        self.softmax_his = self.softmaxlayer.train(pos_cat, self.y)

    def test(self, test_set):
        print('[Test]')

        x_test, y_test = test_set.data, test_set.targets
        x_test = x_test.type(torch.float32) / 255
        x_test = (x_test - self.mean) / self.std
        x_test = x_test.reshape(x_test.shape[0], -1)
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)

        with torch.no_grad():
            test_outputs = self.ffnn.predict(x_test)
            test_cat = torch.cat(test_outputs[1:], dim=1)
            preds = self.softmaxlayer(test_cat)

        acc = (torch.argmax(preds, 1) == y_test).float().mean()
        print('test error: ', 1 - acc.item())


class RFTrainer:
    def __init__(self,
                 train_set,
                 out_channel_list,
                 input_shape_list,
                 kernel_size_list,
                 strides_list,
                 out_dim,
                 threshold=10,
                 lr=0.001,
                 batch_size=512,
                 epochs=100,
                 dropout=0,
                 device='cuda:0'
                 ):

        self.device = device

        self.x_pos, self.y = train_set.data, train_set.targets

        # normalize data
        self.x_pos = self.x_pos.type(torch.float32) / 255
        self.mean, self.std = self.x_pos.mean(), self.x_pos.std()
        self.x_pos = (self.x_pos - self.mean) / self.std

        self.x_neg = create_neg(self.x_pos)

        self.x_pos = self.x_pos.unsqueeze(1)
        self.x_neg = self.x_neg.unsqueeze(1)

        self.x_pos = self.x_pos.to(device)
        self.x_neg = self.x_neg.to(device)
        self.y = self.y.to(device)

        self.ffnn = FFRFNN(out_channel_list,
                           input_shape_list,
                           kernel_size_list,
                           strides_list,
                           threshold,
                           lr,
                           batch_size,
                           epochs,
                           dropout,
                           device)

        softmax_input_dim = 0
        for shape in input_shape_list[1:]:
            dim = 1
            for n in shape:
                dim *= n
            softmax_input_dim += dim

        self.softmaxlayer = SoftmaxLayer(
            softmax_input_dim, out_dim, lr, batch_size, epochs)

        self.ffnn.to(device)
        self.softmaxlayer.to(device)

        self.ffnn_his = None
        self.softmax_his = None

    def train(self):
        print('\n[Start training FF-Layers]')
        pos_outputs, self.ffnn_his = self.ffnn.train(self.x_pos, self.x_neg)

        # use positive data outputs except first ff-layer
        pos_outputs = [output.reshape(output.shape[0], -1)
                       for output in pos_outputs]
        pos_cat = torch.cat(pos_outputs[1:], dim=1)

        print('\n[Start training Softmax-Layer]')
        self.softmax_his = self.softmaxlayer.train(pos_cat, self.y)

    def test(self, test_set):
        print('[Test]')

        x_test, y_test = test_set.data, test_set.targets
        x_test = x_test.type(torch.float32) / 255
        x_test = (x_test - self.mean) / self.std

        x_test = x_test.unsqueeze(1)
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)

        with torch.no_grad():
            test_outputs = self.ffnn.predict(x_test)
            test_outputs = [output.reshape(output.shape[0], -1)
                            for output in test_outputs]
            test_cat = torch.cat(test_outputs[1:], dim=1)
            preds = self.softmaxlayer(test_cat)

        acc = (torch.argmax(preds, 1) == y_test).float().mean()
        print('test error: ', 1 - acc.item())
