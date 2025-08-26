import os
import numpy as np
import pandas as pd
import logging
import time
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings("ignore")
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

CSV_FILE = "COMPLETE_DATASETLONG_LONG.csv"
SERIES_COLUMN = "spreadz_PNC_SCHW"
WINDOW_SIZE = 15
TEST_SIZE_RATIO = 0.2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("PredictSpreadLog.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# def prepare_windows(sequence, window_size, horizon):
#     X, y = [], []
#     # horizon-1, weil horizon=1 -> target direkt nach Window-Ende
#     offset = horizon - 1
#     for i in range(len(sequence) - window_size - offset):
#         X.append(sequence[i : i + window_size])
#         y.append(sequence[i + window_size + offset])
#     return np.array(X), np.array(y)

def prepare_windows(sequence: np.ndarray, window_size: int, horizon: int):
    X, y = [], []
    offset = horizon - 1
    n_samples, n_features = sequence.shape

    for i in range(n_samples - window_size - offset):
        window = sequence[i : i + window_size]
        X.append(window.flatten())
        y.append(sequence[i + window_size + offset, 0])
        # print(sequence[i + window_size + offset, 0])

    return np.array(X), np.array(y)


def PredictSpread(full_sequence, horizon, test_ratio=TEST_SIZE_RATIO):
    X, y = prepare_windows(full_sequence, WINDOW_SIZE, horizon)
    print(X[0])
    print(y[0])
    logger.info(f"Horizon={horizon}: {len(X)} Windows erzeugt.")

    n_windows = len(X)
    if test_ratio < 1:
        split_idx = int(n_windows * (1 - test_ratio))
    else:
        split_idx = test_ratio

    


    X_train, y_train = X[:split_idx], y[:split_idx]
    # print(X_train.shape)
    # print(y_train.shape)
    
    X_test,  y_test  = X[split_idx:],  y[split_idx:]
    logger.info(f"Train: {len(X_train)} Samples, Test: {len(X_test)} Samples")

    regressor = TabPFNRegressor(device="cpu")
    regressor.fit(X_train, y_train)
    
    logger.info(X_test[0])
    pred = regressor.predict(X_test)

    # logger.info(f"Test-True: {y_test[0]:.4f} | Test-Pred: {pred:.4f}")
    return y, regressor.predict(X), y_test, pred

if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting TabPFN Spread Prediction")

    df = pd.read_csv(CSV_FILE)
    df["date"] = pd.to_datetime(df["date"]) 
    df = df.set_index("date")

    FeautureArray = ["spreadz_PNC_SCHW", "Close_Bank"]

    df_clean = df.dropna(subset=FeautureArray)
    seq = np.column_stack([
        df_clean["spreadz_PNC_SCHW"].to_numpy(),
        df_clean["Close_Bank"].to_numpy(),
    ])

    IdxMask = df_clean.index.year == 2023
    ValidationIDX = np.where(IdxMask)[0]
    # print(ValidationIDX[0])  
    # print(df["close"].iloc[ValidationIDX[0]])

    Hori = [1, 2, 5, 10, 15]
    for horizon in Hori:
        true_all, pred_all, true_now, pred_now = PredictSpread(seq, horizon, test_ratio=ValidationIDX[0])
        
        df_results = pd.DataFrame({
            "True": true_all,
            "Pred": pred_all
        })
        output_path = f"TABPFN_spread_predictions_h{horizon}_{FeautureArray}.csv"
        df_results.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
        pred_now = pd.Series(pred_now)

        plt.figure(figsize=(10, 4))
        plt.plot(true_now, label="True (Backtest)")
        plt.plot(pred_now.shift(-horizon), label="Pred (Backtest)")
        plt.title(f"Horizon={horizon}")
        plt.legend()
        # plt.show()

