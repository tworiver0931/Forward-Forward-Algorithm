import torch
from torchvision.datasets import MNIST
from model import FFNN, SoftmaxLayer
from utils import create_neg


class Trainer:
    def __init__(self, dims, threshold=0, lr=0.001, batch_size=256, epochs=100, device='cuda:0'):
      
        train_mnist_set = MNIST('./data/', train=True,
                                download=True)
        self.device = device

        self.x_pos, self.y = train_mnist_set.data, train_mnist_set.targets
        self.x_pos = self.x_pos.type(torch.float32)
        self.x_pos = (self.x_pos - 0.1307) / 0.3081

        self.x_neg = create_neg(self.x_pos)

        self.x_pos = self.x_pos.to(device)
        self.x_neg = self.x_neg.to(device)
        self.y = self.y.to(device)

        self.ffnn = FFNN(dims, threshold, lr, batch_size, epochs, device)
        self.softmaxlayer = SoftmaxLayer(sum(dims[2:]), lr, batch_size, epochs)

        self.ffnn.to(device)
        self.softmaxlayer.to(device)

        self.ffnn_his = None
        self.softmax_his = None

    def train(self):
        print('[Start training FF-Layers]')
        pos_outputs, self.ffnn_his = self.ffnn.train(self.x_pos, self.x_neg)

        # use positive data outputs except first ff-layer
        pos_cat = torch.cat(pos_outputs[1:], dim=1)
        print(pos_cat.shape)

        print('[Start training Softmax-Layer]')
        self.softmax_his = self.softmaxlayer.train(pos_cat, self.y)

    def test(self):
        print('[Test]')

        test_mnist_set = MNIST('./data/', train=False,
                               download=True)

        x_test, y_test = test_mnist_set.data, test_mnist_set.targets
        x_test = x_test.type(torch.float32)
        x_test = (x_test - 0.1307) / 0.3081
        x_test = x_test.reshape(x_test.shape[0], -1)
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)

        with torch.no_grad():
            test_outputs = self.ffnn(x_test)
            test_cat = torch.cat(test_outputs[1:], dim=1)
            pred = self.softmaxlayer(test_cat)

        acc = (torch.argmax(pred, 1) == y_test).float().mean()
        print('test error: ', 1 - acc.item())
