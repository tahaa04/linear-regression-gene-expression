import csv
import pandas as pd
import numpy as np

#IMPORT FILE
raw_file = "data/raw_data/denseDataOnlyDownload (1).tsv"

#OPEN RAW XENA FILE AS A PANDAS DATAFRAME -'data'
data = pd.read_csv(raw_file, sep="\t")

#SELECT REQUIRED COLUMNS FOR LINEAR REGRESSION
lr_data = data[["sample", "cg12981137", "MGMT"]].copy()

#RENAME COLUMNS
lr_data.columns = [
    "sample",
    "cg12981137_methylation",
    "MGMT_expression"
]

#CONVERT VALUES TO NUMBERS
lr_data["cg12981137_methylation"] = pd.to_numeric(
    lr_data["cg12981137_methylation"],
    errors="coerce"
)

lr_data["MGMT_expression"] = pd.to_numeric(
    lr_data["MGMT_expression"],
    errors="coerce"
)

#REMOVE ROWS WITH MISSING VALUES
lr_data = lr_data.dropna()

#REMOVE DUPLICATE SAMPLES IF PRESENT
lr_data = lr_data.drop_duplicates(subset="sample")

#ZERO INDEX
lr_data = lr_data.reset_index(drop=True)

#EXPORT WRANGLED FILE
lr_data.to_csv(
    "data/processed_data/mgmt_cg12981137_lr_data.tsv",
    sep="\t",
    index=False
)