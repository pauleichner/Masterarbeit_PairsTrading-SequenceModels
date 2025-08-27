from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection    import TimeSeriesSplit
import pandas as pd
import numpy as np
import torch
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score, Accuracy


DF = pd.read_csv("COMPLETE_DATASET.csv", parse_dates=["date"])
DF = DF.dropna(subset=["spreadz_PNC_SCHW"]).reset_index(drop=True)
spread = DF["spreadz_PNC_SCHW"].values.astype("float32")

n_splits        = 5
forecast_horizon = 1
tscv            = TimeSeriesSplit(n_splits=n_splits)

rmse_list, mae_list, r2_list, acc_list = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(spread), 1):
    train       = spread[train_idx]
    test_values = spread[test_idx].tolist()

    model      = ARIMA(train, order=(1,0,0), trend="c").fit()
    res_update = model

    preds, trues, bases = [], [], []
    N = len(test_values) - forecast_horizon + 1
    for i in range(N):
        # h-step forecast
        fc   = res_update.forecast(steps=forecast_horizon)
        yhat = float(fc[forecast_horizon - 1])
        ytrue = test_values[i + forecast_horizon - 1]

        if i == 0:
            ybase = float(train[-1])
        else:
            ybase = test_values[i - 1]

        preds.append(yhat)
        trues.append(ytrue)
        bases.append(ybase)
        res_update = res_update.append([test_values[i]], refit=False)

    # convert to Tensor
    y_pred = torch.tensor(preds, dtype=torch.float32)
    y_true = torch.tensor(trues, dtype=torch.float32)
    y_base = torch.tensor(bases, dtype=torch.float32)

    # Conmpute ML Metrics
    mse  = MeanSquaredError()(y_pred, y_true)
    rmse = torch.sqrt(mse).item()
    mae  = MeanAbsoluteError()(y_pred, y_true).item()
    r2   = R2Score()(y_pred, y_true).item()

    # ACC
    pred_dir = (y_pred > y_base).long()
    true_dir = (y_true > y_base).long()
    acc      = Accuracy(task="binary")(pred_dir, true_dir).item()

    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)
    acc_list.append(acc)

print(f"RMSE = {np.mean(rmse_list):.4f} | {np.std(rmse_list):.4f}")
print(f"MAE  = {np.mean(mae_list):.4f} | {np.std(mae_list):.4f}")
print(f"R²   = {np.mean(r2_list):.4f} | {np.std(r2_list):.4f}")
print(f"ACC  = {np.mean(acc_list):.4f} | {np.std(acc_list):.4f}")


arma_values1 = {
    'RMSE_Valid': np.mean(rmse_list),
    'RMSE_Std'  : np.std(rmse_list),
    'MAE_Valid' : np.mean(mae_list),
    'MAE_Std'   : np.std(mae_list),
    'R2_Valid'  : np.mean(r2_list),
    'R2_Std'    : np.std(r2_list),
    'ACC_Valid' : np.mean(acc_list),
    'ACC_Std'   : np.std(acc_list),
}


print("=== Cross-Validation Results ===")
print(f"Average RMSE : {arma_values1['RMSE_Valid']:.4f} ± {arma_values1['RMSE_Std']:.4f}")
print(f"Average MAE  : {arma_values1['MAE_Valid']:.4f} ± {arma_values1['MAE_Std']:.4f}")
print(f"Average R²   : {arma_values1['R2_Valid']:.4f} ± {arma_values1['R2_Std']:.4f}")
print(f"Average Acc  : {arma_values1['ACC_Valid']:.4f} ± {arma_values1['ACC_Std']:.4f}")


print("\narma_values5 = {")
for key, val in arma_values1.items():
    print(f"    '{key}': {val:.4f},")

print("}")
