import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3,3))
        self.conv2 = nn.Conv2d(16, 32, (3,3))
        self.conv3 = nn.Conv2d(32, 64, (3,3))

        self.pool = nn.MaxPool2d(2,2)

        self.flatten_shape = self.input_linear_neurons(torch.rand(1,3,32,32))
        self.layer1 = nn.Linear(self.flatten_shape, 512)
        self.layer2 = nn.Linear(512, 32)
        self.layer3 = nn.Linear(32, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        h = F.relu(self.conv1(inputs))
        h = self.pool(h)

        h = F.relu(self.conv2(h))
        h = self.pool(h)

        h = F.relu(self.conv3(h))
        h = self.pool(h)

        h = h.view(batch_size, -1)
        h = F.relu(self.layer1(h))
        h = F.relu(self.layer2(h))
        out = F.softmax(self.layer3(h), dim=1)
        return out

    def input_linear_neurons(self, inputs):
        batch_size = inputs.shape[0]
        h = F.relu(self.conv1(inputs))
        h = self.pool(h)

        h = F.relu(self.conv2(h))
        h = self.pool(h)

        h = F.relu(self.conv3(h))
        h = self.pool(h)
        
        h = h.view(batch_size, -1)
        return h.shape[-1]