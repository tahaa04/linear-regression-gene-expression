import pandas as pd
import numpy as np
import os

#IMPORT CLEANED DATA
data = "data/processed_data/mgmt_cg12981137_lr_data.tsv"

data_df = pd.read_csv(data, sep="\t")

#CREATE RANDOMIZED INDICES
n = len(data_df)

indices = np.arange(n)

np.random.seed(22)
np.random.shuffle(indices)

#20:80 TEST-TRAIN SPLIT OF RANDOMIZED INDICES
test_size = int(0.2 * n)

test_indices = indices[:test_size]
train_indices = indices[test_size:]

#CREATE TRAIN AND TEST DATAFRAMES
train_data = data_df.iloc[train_indices].copy()
test_data = data_df.iloc[test_indices].copy()

#ZERO INDEX
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

#EXPORT TRAIN AND TEST FILES
train_data.to_csv(
    "data/processed_data/train_data.tsv",
    sep="\t",
    index=False
)

test_data.to_csv(
    "data/processed_data/test_data.tsv",
    sep="\t",
    index=False
)

