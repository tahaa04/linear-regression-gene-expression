import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

#CREATE OUTPUT FOLDERS
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

#IMPORT TRAIN AND TEST DATA
train_data = "data/processed_data/train_data.tsv"
test_data = "data/processed_data/test_data.tsv"

train_df = pd.read_csv(train_data, sep="\t")
test_df = pd.read_csv(test_data, sep="\t")

#CONVERT COLUMNS TO NUMPY ARRAYS
x_train = train_df["cg12981137_methylation"].values
y_train = train_df["MGMT_expression"].values

x_test = test_df["cg12981137_methylation"].values
y_test = test_df["MGMT_expression"].values

#RESHAPE X FOR SCIKIT-LEARN
x_train_reshaped = x_train.reshape(-1, 1)
x_test_reshaped = x_test.reshape(-1, 1)

#EVALUATION FUNCTIONS
def mean_squared_error(y_actual, y_pred):
    error = y_pred - y_actual
    mse = np.mean(error ** 2)
    return mse

def mean_absolute_error(y_actual, y_pred):
    error = np.abs(y_actual - y_pred)
    mae = np.mean(error)
    return mae

def root_mean_squared_error(y_actual, y_pred):
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def r2_score(y_actual, y_pred):
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    return r2

#TRAIN SCIKIT-LEARN MODEL
model = LinearRegression()

model.fit(x_train_reshaped, y_train)

#GET WEIGHT AND BIAS
w = model.coef_[0]
b = model.intercept_

print("Scikit-learn Linear Regression")
print("Weight:", w)
print("Bias:", b)

#PREDICT TEST DATA
y_test_pred = model.predict(x_test_reshaped)

#EVALUATE ON TEST DATA
mse = mean_squared_error(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Scikit-learn Test MSE:", mse)
print("Scikit-learn Test RMSE:", rmse)
print("Scikit-learn Test MAE:", mae)
print("Scikit-learn Test R2:", r2)

#SAVE METRICS
sklearn_metrics = pd.DataFrame({
    "Model": ["Scikit-learn LinearRegression"],
    "Weight/Slope": [w],
    "Bias/Intercept": [b],
    "MSE": [mse],
    "RMSE": [rmse],
    "MAE": [mae],
    "R2": [r2]
})

sklearn_metrics.to_csv(
    "outputs/metrics/sklearn_metrics.csv",
    index=False
)

#SAVE PREDICTIONS
sklearn_predictions = pd.DataFrame({
    "sample": test_df["sample"],
    "cg12981137_methylation": x_test,
    "actual_MGMT_expression": y_test,
    "sklearn_predicted_MGMT_expression": y_test_pred,
    "sklearn_residual": y_test - y_test_pred
})

sklearn_predictions.to_csv(
    "outputs/predictions/sklearn_predictions.csv",
    index=False
)

#PLOT SCIKIT-LEARN REGRESSION LINE
x_line = np.linspace(x_train.min(), x_train.max(), 100)
y_line = model.predict(x_line.reshape(-1, 1))

plt.figure(figsize=(7, 5))

plt.scatter(x_train, y_train, label="Training data")
plt.plot(x_line, y_line, label="Scikit-learn regression line")

plt.xlabel("cg12981137 Methylation")
plt.ylabel("MGMT Expression")
plt.title("Scikit-learn Linear Regression")
plt.legend()

plt.savefig(
    "outputs/figures/sklearn_regression_line.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

#ACTUAL VS PREDICTED PLOT
plt.figure(figsize=(6, 6))

plt.scatter(y_test, y_test_pred)

min_value = min(y_test.min(), y_test_pred.min())
max_value = max(y_test.max(), y_test_pred.max())

plt.plot([min_value, max_value], [min_value, max_value])

plt.xlabel("Actual MGMT Expression")
plt.ylabel("Predicted MGMT Expression")
plt.title("Actual vs Predicted - Scikit-learn Model")

plt.savefig(
    "outputs/figures/sklearn_actual_vs_predicted.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

#COMBINE METRICS IF MANUAL METRICS FILE EXISTS
manual_metrics_file = "outputs/metrics/manual_metrics.csv"

if os.path.exists(manual_metrics_file):
    manual_metrics = pd.read_csv(manual_metrics_file)

    comparison = pd.concat(
        [manual_metrics, sklearn_metrics],
        ignore_index=True
    )

    comparison.to_csv(
        "outputs/metrics/model_comparison.csv",
        index=False
    )

    print("Model comparison saved.")