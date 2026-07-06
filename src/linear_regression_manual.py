import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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

#MODEL PREDICTION
def predict(x, w, b):
    y_pred = w * x + b
    return y_pred

#GRADIENT DESCENT
def gradient_descent_step(x_train, y_train, alpha, w, b):
    n = len(x_train)

    sum_dw = 0
    sum_db = 0

    for i in range(n):
        x_i = x_train[i]
        y_i = y_train[i]

        prediction = w * x_i + b
        error = prediction - y_i

        sum_dw = sum_dw + (error * x_i)
        sum_db = sum_db + error

    dw = (1 / n) * sum_dw
    db = (1 / n) * sum_db

    w = w - alpha * dw
    b = b - alpha * db

    return w, b


#EVALUATION METRICS:

#MEAN SQUARED ERROR
def mean_squared_error(y_actual, y_pred):
    error = y_pred - y_actual
    mse = np.mean(error ** 2)
    return mse

#MEAN ABSOLUTE ERROR
def mean_absolute_error(y_actual, y_pred):
    error = np.abs(y_actual - y_pred)
    mae = np.mean(error)
    return mae

#ROOT MEAN SQUARED ERROR
def root_mean_squared_error(y_actual, y_pred):
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    return rmse

#R2 SCORE
def r2_score(y_actual, y_pred):
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    return r2


#TRAIN
alpha = 0.1
iterations = 3000

w = 0
b = 0

cost_history = []
w_history = []
b_history = []


for i in range(iterations):
    y_train_pred = predict(x_train, w, b)
    cost = mean_squared_error(y_train, y_train_pred)

    cost_history.append(cost)
    w_history.append(w)
    b_history.append(b)

    w, b = gradient_descent_step(x_train, y_train, alpha, w, b)


print("Manual Linear Regression")
print("Final weight:", w)
print("Final bias:", b)
print("Final training cost:", cost_history[-1])

#PLOT COST FUNCTION
plt.figure(figsize=(7, 5))

plt.plot(cost_history)

plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Cost Reduction During Gradient Descent")

plt.savefig(
    "outputs/figures/manual_cost_reduction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#PLOT REGRESSION LINE
x_line = np.linspace(x_train.min(), x_train.max(), 100)
y_line = predict(x_line, w, b)

plt.figure(figsize=(7, 5))

plt.scatter(x_train, y_train, label="Training data")
plt.plot(x_line, y_line, label="Manual regression line")

plt.xlabel("cg12981137 Methylation")
plt.ylabel("MGMT Expression")
plt.title("Manual Linear Regression Using Gradient Descent")
plt.legend()

plt.savefig(
    "outputs/figures/manual_regression_line.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#PREDICT TEST DATA
y_test_pred = predict(x_test, w, b)

#EVALUATE ON TEST DATA
mse = mean_squared_error(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Manual Test MSE:", mse)
print("Manual Test RMSE:", rmse)
print("Manual Test MAE:", mae)
print("Manual Test R2:", r2)

#SAVE METRICS
manual_metrics = pd.DataFrame({
    "Model": ["Manual Linear Regression"],
    "Weight/Slope": [w],
    "Bias/Intercept": [b],
    "MSE": [mse],
    "RMSE": [rmse],
    "MAE": [mae],
    "R2": [r2]
})

manual_metrics.to_csv(
    "outputs/metrics/manual_metrics.csv",
    index=False
)

#SAVE PREDICTIONS
manual_predictions = pd.DataFrame({
    "sample": test_df["sample"],
    "cg12981137_methylation": x_test,
    "actual_MGMT_expression": y_test,
    "manual_predicted_MGMT_expression": y_test_pred,
    "manual_residual": y_test - y_test_pred
})

manual_predictions.to_csv(
    "outputs/predictions/manual_predictions.csv",
    index=False
)

#ACTUAL VS PREDICTED PLOT
plt.figure(figsize=(6, 6))

plt.scatter(y_test, y_test_pred)

min_value = min(y_test.min(), y_test_pred.min())
max_value = max(y_test.max(), y_test_pred.max())

plt.plot([min_value, max_value], [min_value, max_value])

plt.xlabel("Actual MGMT Expression")
plt.ylabel("Predicted MGMT Expression")
plt.title("Actual vs Predicted - Manual Model")

plt.savefig(
    "outputs/figures/manual_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#RESIDUAL PLOT
residuals = y_test - y_test_pred

plt.figure(figsize=(7, 5))

plt.scatter(y_test_pred, residuals)
plt.axhline(0)

plt.xlabel("Predicted MGMT Expression")
plt.ylabel("Residual")
plt.title("Residual Plot - Manual Model")

plt.savefig(
    "outputs/figures/manual_residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()