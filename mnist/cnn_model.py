import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.conv4 = nn.Conv2d(64, 64, (3, 3))

        self.pool = nn.MaxPool2d(2, 2)
        self.conv_bn1 = nn.BatchNorm2d(num_features=32)
        self.conv_bn2 = nn.BatchNorm2d(num_features=64)
        self.dropout = nn.Dropout(0.10)

        self.flatten_shape = self.infer_linear_neurons(torch.rand(1, 3, 32, 32))
        self.layer1 = nn.Linear(self.flatten_shape, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        h = F.relu(self.conv1(inputs))
        h = F.relu(self.conv2(h))
        h = self.conv_bn1(h)
        h = self.pool(h)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = self.conv_bn2(h)
        h = self.pool(h)

        h = h.view(batch_size, -1)

        h = F.relu(self.layer1(h))
        h = self.dropout(h)
        h = F.relu(self.layer2(h))
        h = self.dropout(h)
        out = self.layer3(h)

        return out

    def infer_linear_neurons(self, inputs):
        batch_size = inputs.shape[0]
        h = F.relu(self.conv1(inputs))
        h = F.relu(self.conv2(h))
        h = self.conv_bn1(h)
        h = self.pool(h)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = self.conv_bn2(h)
        h = self.pool(h)

        h = h.view(batch_size, -1)
        return h.shape[-1]


