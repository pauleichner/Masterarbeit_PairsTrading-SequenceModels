import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import TimeSeriesSplit
from torchmetrics import Accuracy
from tabpfn import TabPFNClassifier
import torch
import warnings

warnings.filterwarnings("ignore")

plt.style.use('classic')
rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.prop_cycle': plt.cycler(color=[
        '#0072BD', '#D95319', '#EDB120', '#77AC30', '#A2142F',
    ])
})

from helperFunctions import createDataset, getFeaturesSequence

torch.cuda.manual_seed_all(42)

DF = pd.read_csv("COMPLETE_DATASET.csv")
DF["date"] = pd.to_datetime(DF["date"])

def performFEATURESearch(FeaturesTuple, NameSpread):
    NameParams  = ['LookBack', 'Horizon', 'HiddenSize', 'LR', 'Epochs']
    ValueParams = [lookback, horizon, hidden_Size, learningRate, epochs]
    dictCONF    = dict(zip(NameParams, ValueParams))

    feature_label = FeaturesTuple[0]
    Features      = FeaturesTuple[1:]
    dataFeatures  = np.stack(Features, axis=1).astype("float32")
    print("DONE - Created the Features DATA")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tscv   = TimeSeriesSplit(n_splits=5)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(dataFeatures), start=1):
        train_vals = dataFeatures[train_idx]
        test_vals  = dataFeatures[test_idx]

        X_train_full, Y_train_full = createDataset(train_vals, lookback, horizon)
        X_test_full,  Y_test_full  = createDataset(test_vals,  lookback, horizon)

        last_train  = X_train_full[:, -1, 0]
        last_test   = X_test_full[:,  -1, 0]
        Y_train_dir = (Y_train_full[:, -1, 0] > last_train).long().cpu().numpy()
        Y_test_dir  = (Y_test_full[:,  -1, 0] > last_test).long().cpu().numpy()

        n_tr = X_train_full.shape[0]
        n_te = X_test_full.shape[0]
        X_tr_np = X_train_full.cpu().numpy().reshape(n_tr, -1)
        X_te_np = X_test_full.cpu().numpy().reshape(n_te, -1)

        clf = TabPFNClassifier(device=device)
        clf.fit(X_tr_np, Y_train_dir)
        probs_val = clf.predict_proba(X_te_np)
        preds_val = (probs_val[:, 1] >= 0.5).astype(int)

        ACC_metric   = Accuracy(task="binary", threshold=0.5).to(device)
        acc_val   = ACC_metric(Y_test_dir, preds_val)


        DFMetrics = pd.DataFrame({
            'Accuracy':  [acc_val],
            **dictCONF
        })
        out_dir = "RESULTSADDFEATURES/BINARY"
        os.makedirs(out_dir, exist_ok=True)
        DFMetrics.to_csv(
            f"{out_dir}/TABPFN_h{horizon}_lb{lookback}_FOLD{fold}.csv",
            index=False, encoding='utf-8'
        )
        print(f"Fold {fold} results saved.")

    return {
        **dictCONF,
        'LastFold': fold,
        'LastValAcc': acc_val,
    }


if __name__ == "__main__":
    Horizon = [1, 2, 5]
    for ahor in Horizon:
        lookback     = 10
        horizon      = ahor
        hidden_Size  = 10      # only for logging
        learningRate = 20e-5    # ditto
        epochs       = 50       # ditto

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on", device)

        DataTUPLE = getFeaturesSequence(
            "spreadz_PNC_SCHW",
            False, False, False, False,
            False, False, False, False,
            False, False
        )
        performFEATURESearch(DataTUPLE, "spreadz_PNC_SCHW")
