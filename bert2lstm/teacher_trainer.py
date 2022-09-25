import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# global variables
TRAIN_FILE = "~/Data/Kaggle/imdb/imdb_train.csv"
TEST_FILE = "~/Data/Kaggle/imdb/imdb_test.csv"
MODEL_DIR = "~/Data/Kaggle/imdb/models/"

BS = 16
LR = 1e-4
EPOCHS = 5
EVAL_SAMPLES = 10000
TARGET_CLASSES = 2
HIDDEN_DIM = 64
BERT_MODEL_NAME = "bert-base-uncased"
BERT_DIM = 768
TEXT_COL = "review"
TARGET_COL = "target"


class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, text_col, target_col):
        self.data = data
        self.tokenizer = (tokenizer,)
        self.text_col = (text_col,)
        self.target_col = target_col

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        text = self.data[self.text_col].iloc[idx]
        target = self.data[self.target_col].iloc[idx]

        # tokenize data


class IMDBModel(nn.Module):
    def __init__(self, bert_model, bert_dim, hidden_dim, num_targets):
        self.bert_model = AutoModel()
        self.fc1 = nn.Linear(bert_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_targets)
        self.softmax = nn.Softmax()

    def forward(self, batch):
        pass


def train_model():

    # load data
    data = pd.read_csv(TRAIN_FILE)
    data = data.sample(frac=1.0)

    train_samples = data.shape[0] - EVAL_SAMPLES
    train_data = data.iloc[:train_samples, :]
    eval_data = data.iloc[train_samples:, :]

    # load tokenizer
    tokenizer = AutoTokenizer()

    # build datasets
    train_dataset = IMDBDataset(train_data, tokenizer, TEXT_COL, TARGET_COL)
    eval_dataset = IMDBDataset(eval_data, tokenizer, TEXT_COL, TARGET_COL)

    # build dataloader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, num_workers=4, batch_size=BS
    )
    eval_dataloader = DataLoader(eval_dataset, num_workers=4, batch_size=BS)

    # create model
    bert_model = IMDBModel(
        BERT_MODEL_NAME, BERT_DIM, HIDDEN_DIM, TARGET_CLASSES
    )

    # create logger

    # create lightning trainer

    # train model
    pass


if __name__ == "__main__":
    train_model()
