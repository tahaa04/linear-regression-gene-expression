## Linear Regression from Scratch on MGMT Methylation and Gene Expression Data 

## Project Overview

This project implements simple linear regression from scratch in Python without using machine-learning libraries for model training. The goal is to understand the mathematics behind linear regression, including the line of best fit, prediction error, mean squared error, gradient descent, and model evaluation on unseen test data.

To make the project more meaningful than using synthetic data, I applied the model to real cancer genomics data from TCGA (The Cancer Genome Atlas) accessed through the UCSC Xena Browser. The dataset contains DNA methylation values for CpG probes associated with the MGMT gene and MGMT RNA expression values from glioma/glioblastoma tumor samples.

The prediction task is:

Can methylation at an MGMT-associated CpG site be used to predict MGMT RNA expression?

## Biological Background:

MGMT is a DNA repair gene that is especially important in glioblastoma research. In some tumors, methylation near the MGMT promoter is associated with reduced MGMT expression. Since promoter methylation can repress transcription, a negative relationship is expected: higher methylation should generally correspond to lower gene expression.

This project uses the CpG probe cg12981137 methylation level as the input feature. The dataset contained many MGMT-associated CpG probes, but methylation effects are region-specific, so averaging all probes could mix promoter-relevant CpGs with less relevant regions. The probe cg12981137 was selected because it is one of the CpG probes used in the published MGMT-STP27 model for MGMT promoter methylation assessment in glioma/glioblastoma datasets.

## Important Note:
This project is educational. It is not intended to build a clinical prediction model or discover a new biomarker. The purpose is to show how linear regression works under the hood using a real genomics dataset where the relationship between the variables has biological meaning.

## Dataset:
The final cleaned dataset contains two numerical columns:

Column	Description
cg12981137_methylation	DNA methylation beta value at the selected MGMT-associated CpG probe
MGMT_expression	MGMT RNA expression value

The methylation beta value ranges from 0 to 1:

0 means mostly unmethylated
1 means mostly methylated

The regression task is:

X = cg12981137 methylation
y = MGMT expression

## Exploratory Data Analysis
Before training the model, the relationship between cg12981137 methylation and MGMT expression was examined using a scatter plot. The scatter plot shows a generally negative relationship between cg12981137 methylation and MGMT expression, making the dataset suitable for demonstrating simple linear regression.

## Model:
The model is a simple linear regression model:

MGMT_expression = m × methylation + b

Where:

m is the slope/coefficient
b is the intercept
the model learns the line that minimizes prediction error

The model is trained from scratch using gradient descent. During training, the cost function is calculated using mean squared error.

## What This Project Demonstrates

Loading and cleaning real biological data
Selecting a biologically meaningful single input feature
Implementing linear regression mathematically from scratch
Training the model using gradient descent
Visualizing the regression line over a scatterplot
Plotting cost reduction over training iterations
Evaluating the model using train/test split 
Comparing the from-scratch implementation with scikit-learn’s LinearRegression

## Project Structure

```
data/
  raw_data/            Raw TCGA/Xena download (methylation + expression, all probes)
  processed_data/      Cleaned single-probe dataset, and the train/test split
notebooks/
  01_exploratory_data_analysis.ipynb   Scatter plot / EDA on the cleaned data
src/
  data_wrangling.py                 Cleans the raw file -> processed_data/mgmt_cg12981137_lr_data.tsv
  train_test_split.py               Splits the cleaned data -> train_data.tsv / test_data.tsv (80/20)
  linear_regression_manual.py       From-scratch gradient descent model, evaluation, and plots
  scikit_learn_linear_regression.py scikit-learn model, evaluation, plots, and model comparison
outputs/
  figures/     Regression lines, cost curve, actual-vs-predicted, residual plots (.png)
  metrics/     MSE/RMSE/MAE/R2 for each model, plus a combined model_comparison.csv
  predictions/ Per-sample test set predictions and residuals for each model
```

## How to Run

Install dependencies, then run the scripts in order from the project root (each one reads the previous one's output):

```
pip install -r requirements.txt

python src/data_wrangling.py
python src/train_test_split.py
python src/linear_regression_manual.py
python src/scikit_learn_linear_regression.py
```

Figures pop up via `plt.show()` as each script runs, and are also saved to `outputs/figures/`.

Optional EDA notebook: `jupyter notebook notebooks/01_exploratory_data_analysis.ipynb`

## Viewing Results

- Regression lines and diagnostic plots: `outputs/figures/`
- Numeric metrics (MSE, RMSE, MAE, R2) for each model: `outputs/metrics/manual_metrics.csv`, `outputs/metrics/sklearn_metrics.csv`, and the side-by-side `outputs/metrics/model_comparison.csv`
- Per-sample predictions and residuals on the held-out test set: `outputs/predictions/`

