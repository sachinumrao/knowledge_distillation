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

from sklearn.metrics import accuracy_score


class NNModel(pl.LightningModule):

    def __init__(self):

        super(NNModel, self).__init__()

        self.model = CNNModel()

        # Define loss function
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
                                               train=True,
                                               download=True,
                                               transform=transform)
        val_len = int(len(cifar10)*val_size)
        train_len = len(cifar10) - val_len
        segments = [train_len, val_len]
        self.cifar10_train, self.cifar10_val = random_split(cifar10, segments)

        self.cifar10_test = torchvision.datasets.CIFAR10(root="~/Data/CIFAR10",
                                                         train=False,
                                                         download=True,
                                                         transform=transform)

    @pl.data_loader
    def train_dataloader(self):
        train_dataloader = DataLoader(self.cifar10_train,
                                      batch_size=16,
                                      shuffle=True,
                                      num_workers=8)
        return train_dataloader    

    @pl.data_loader
    def val_dataloader(self):
        val_dataloader = DataLoader(self.cifar10_val,
                                    batch_size=16,
                                    shuffle=False,
                                    num_workers=8)
        return val_dataloader   

    @pl.data_loader
    def test_dataloader(self):
        test_dataloader = DataLoader(self.cifar10_test,
                                     batch_size=16,
                                     shuffle=False,
                                     num_workers=8)
        return test_dataloader

    def configure_optimizers(self, lr=0.0005):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    @staticmethod
    def accuracy(self, y, y_hat):
        y = y.numpy().reshape((-1,))
        y_hat = torch.argamx(y_hat, dim=1, keepdim=True).numpy().reshape((-1,))
        acc = accuracy_score(y, y_hat)
        return acc

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        acc = self.accuracy(y, out)
        logs = {'loss': loss, 'acc': acc}

        train_metric = {'loss': loss, 'acc': acc, 'log': logs}
        return train_metric

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        acc = self.accuracy(y, out)
        val_metric = {'val_loss': loss, 'val_acc': acc}
        return val_metric

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss,
                'val_acc': avg_acc}
        context = {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': logs}
        return context

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        acc = self.accuracy(y, out)
        test_metric = {'test_loss': loss, 'test_acc': acc}
        return test_metric

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss,
                'test_acc': avg_acc}
        context = {'test_loss': avg_loss, 'test_acc': avg_acc, 'log': logs}
        return context


if __name__ == '__main__':

    seed = 42
    pl.seed_everything(seed)

    # Create logger
    wandb_logger = WandbLogger(project="CIFAR10_CNN")

    # Create model
    model = NNModel()
    model.prepare_data()

    wandb_logger.watch(model, log='all', log_freq=100)

    # Create trainer object
    trainer = pl.Trainer(max_epochs=5, logger=wandb_logger, profiler=True, deterministic=True)

    # Train the Model
    trainer.fit(model)

    # Test the model
    trainer.test()

# Todo: add accuracy metrics in *_steps and *_epoch_end
# Todo: add validation and test epoch end functions
