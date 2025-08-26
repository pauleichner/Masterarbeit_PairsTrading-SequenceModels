import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
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
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from sklearn.metrics import brier_score_loss

#### Eigene Funktionen
from helperFunctions import createDataset, getFeaturesSequence

torch.cuda.manual_seed_all(42)


DF = pd.read_csv("COMPLETE_DATASET.csv")
DF["date"] = pd.to_datetime(DF["date"])


def performFEATURESearch(FeaturesTuple, NameSpread):
    # Hyperparameter als Dict
    NameParams  = ['LookBack', 'Horizon', 'HiddenSize', 'LR', 'Epochs']
    ValueParams = [lookback, horizon, hidden_Size, learningRate, epochs]
    dictCONF    = dict(zip(NameParams, ValueParams))

    feature_label = FeaturesTuple[0]
    Features      = FeaturesTuple[1:]
    FeatureLEN    = len(Features)
    dataFeatures  = np.stack(Features, axis=1).astype("float32")

    print("DONE - Created the Features DATA")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold, (train_index, test_index) in enumerate(tscv.split(dataFeatures), start=1):
        train_vals = dataFeatures[train_index]
        test_vals  = dataFeatures[test_index]

        X_train_full, Y_train_full = createDataset(train_vals, lookback, horizon)
        X_test_full,  Y_test_full  = createDataset(test_vals,  lookback, horizon)

        last_train   = X_train_full[:, -1, 0]
        last_test    = X_test_full[:,  -1, 0]
        Y_train_dir  = (Y_train_full[:, -1, 0] > last_train).float().unsqueeze(1) # boolean -> 1 and 0
        Y_test_dir   = (Y_test_full[:,  -1, 0] > last_test).float().unsqueeze(1)
        X_train = X_train_full.to(device)
        Y_train = Y_train_dir.to(device)
        X_val   = X_test_full.to(device)
        Y_val   = Y_test_dir.to(device)
        loader = data.DataLoader(data.TensorDataset(X_train, Y_train), batch_size=8, shuffle=True)

        class SpreadLSTM(nn.Module):
            def __init__(self, inputsize):
                super().__init__()
                self.lstm = nn.LSTM(input_size=inputsize,
                                    hidden_size=hidden_Size,
                                    num_layers=1,
                                    batch_first=True,
                                    dropout=0.2)
                self.linear = nn.Linear(hidden_Size, 1)

            def forward(self, x):
                seq_out, _   = self.lstm(x)
                last_hidden  = seq_out[:, -1, :]
                return self.linear(last_hidden)

        model = SpreadLSTM(inputsize=FeatureLEN).to(device)

        loss_fn   = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learningRate)

        ACC_metric   = Accuracy(task="binary", threshold=0.5).to(device)

        ACC_Train, ACC_Valid = [], []

        # Training über Epochen
        for epoch in range(1, epochs+1):
            model.train()
            for X_b, Y_b in loader:
                optimizer.zero_grad()
                logits = model(X_b)
                loss   = loss_fn(logits, Y_b)
                loss.backward()
                optimizer.step()

            # Evaluation pro Epoche
            model.eval()
            with torch.no_grad():
                # Train-Accuracy
                logits_tr = model(X_train)
                preds_tr  = (torch.sigmoid(logits_tr) >= 0.5).long()
                acc_tr    = ACC_metric(preds_tr, Y_train.long()).item()
                ACC_metric.reset()
                ACC_Train.append(acc_tr)

                # Val-Logits & -Probs
                logits_val = model(X_val)
                probs_val  = torch.sigmoid(logits_val)
                preds_val  = (probs_val >= 0.5).long()

                # Accuracy
                acc_val    = ACC_metric(preds_val, Y_val.long()).item()
                ACC_metric.reset()
                ACC_Valid.append(acc_val)

            if epoch % 10 == 0:
                print(
                    f"Fold {fold} Ep {epoch}: "
                    f"TrAcc={acc_tr:.4f} ValAcc={acc_val:.4f} "
                )

        DFMetrics = pd.DataFrame({
            'Epoch':       np.arange(1, epochs+1),
            'Acc_Train':   ACC_Train,
            'Acc_Valid':   ACC_Valid,
        })

        for k, v in dictCONF.items():
            DFMetrics[k] = v
        DFMetrics['Fold'] = fold

        DFMetrics.to_csv(
            f"RESULTSADDFEATURES/BINARY/"
            f"LSTM_h{horizon}_lb{lookback}"
            f"_hs{hidden_Size}_lr{learningRate}"
            f"_ep{epochs}_F-{feature_label}_FOLD{fold}.csv",
            index=False,
            encoding='utf-8'
        )

        if fold == n_splits:
            model_path = (
                f"MODELS/LSTM_BIN_h{horizon}_lb{lookback}"
                f"_hs{hidden_Size}_lr{learningRate}"
                f"_ep{epochs}.pth"
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'lookback': lookback,
                'horizon': horizon,
                'inputsize': FeatureLEN,
            }, model_path)
            print(f"Model saved to {model_path}")

    return {
        **dictCONF,
        'LastFold': fold,
        'LastValAcc': ACC_Valid[-1],
    }


########## MAIN ##########
Horizon = [1, 2, 5]
for ahor in Horizon:
    lookback     = 10
    horizon      = ahor
    hidden_Size  = 10
    learningRate = 20e-5
    epochs       = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    DataTUPLE = getFeaturesSequence(
        "spreadz_PNC_SCHW",
        False, False, False, False,
        False, False, False, False,
        False, False
    )
    performFEATURESearch(DataTUPLE, "spreadz_PNC_SCHW")
