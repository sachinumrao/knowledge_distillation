import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNNModel(nn.Module):
    def __init__(self):
        super(SimpleNNModel, self).__init__()

        self.layer1 = nn.Linear(32*32, 512)
        self.layer2 = nn.Linear(512, 32)
        self.layer3 = nn.Linear(32, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        # Convert images to grayscale
        x = (inputs[:, 0, :, :] + inputs[:, 1, :, :] + inputs[:, 2, :, :])/3
        # Flatten the image
        x = x.view(batch_size, -1)

        h = F.relu(self.layer1(x))
        h = F.relu(self.layer2(h))
        out = F.softmax(self.layer3(h), dim=1)
        return out
