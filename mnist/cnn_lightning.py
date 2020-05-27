import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from cnn_model import CNNModel

class NNModel(pl.LightningModule):

    def __init__(self):

        super(NNModel, self).__init__()

        self.model = CNNModel()

        # Deifne loss function
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.model(x)
        return out

    def prepare_data(self, val_size=0.20):
        # Create transformation for raw data
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )

        # Download training and testing data
        cifar10 = torchvision.datasets.CIFAR10(root="~/Data/CIFAR10", 
                            train=True, download=True, 
                            transform=transform)
        val_len = int(len(cifar10)*val_size)
        train_len = len(cifar10) - val_len
        segments = [train_len, val_len]
        self.cifar10_train, self.cifar10_val = random_split(cifar10, segments)

        self.cifar10_test = torchvision.datasets.CIFAR10(root="~/Data/CIFAR10", 
                            train=False, download=True, 
                            transform=transform)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.cifar10_train, batch_size=64,
                            shuffle=True, num_workers=8)
        return train_dataloader    

    def val_dataloader(self):
        val_dataloader = DataLoader(self.cifar10_val, batch_size=64,
                            shuffle=False, num_workers=8)
        return val_dataloader   

    def test_dataloader(self):
        test_dataloader = DataLoader(self.cifar10_test, batch_size=64,
                            shuffle=False, num_workers=8)
        return test_dataloader

    def configure_optimizers(self, lr=0.005):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        logs = {'loss': loss}
        context = {'loss': loss, 'log': logs}
        return context

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        preds = out.argmax(dim=1, keepdim=True)
        acc = preds.eq(y.view_as(preds)).sum().item()
        context = {'val_step_loss': loss, 'val_step_accuracy': acc}
        return context

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_step_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        context = {'val_loss': avg_loss, 'log': logs}
        return context

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        preds = out.argmax(dim=1, keepdim=True)
        acc = preds.eq(y.view_as(preds)).sum().item()
        context = {'test_loss': loss, 'test_accuracy': acc}
        return context

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'avg_test_loss': avg_loss}
        context = {'avg_test_loss': avg_loss, 'log': logs}
        return context

if __name__ == '__main__':

    # Create logger
    wandb_logger = WandbLogger(project="CIFAR10_CNN")

    # Create model
    model = NNModel()
    model.prepare_data()

    wandb_logger.watch(model, log='all', log_freq=100)

    # Create trainer object
    trainer = pl.Trainer(max_epochs=30, logger=wandb_logger)

    # Train the Model
    trainer.fit(model)
