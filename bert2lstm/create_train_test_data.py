import os

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.20
SEED = 1729
FNAME = "~/Data/Kaggle/imdb/IMDB_Dataset.csv"
TRAIN_FILE = "~/Data/Kaggle/imdb/imdb_train.csv"
TEST_FILE = "~/Data/Kaggle/imdb/imdb_test.csv"


def split_dataset():

    # load data file
    df = pd.read_csv(FNAME)

    # encode target column
    df["target"] = df["sentiment"].apply(lambda x: 1 if x == "poitive" else 0)

    # split data into train and test set
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["sentiment"]
    )

    # display train and test set summray
    print("Train Data Summary: ")
    print(train["sentiment"].describe())

    print("Test Data Summray: ")
    print(test["sentiment"].describe())

    # save data
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)


if __name__ == "__main__":
    split_dataset()
