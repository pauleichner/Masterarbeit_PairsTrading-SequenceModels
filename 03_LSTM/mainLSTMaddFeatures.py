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
        '#0072BD',  # blau
        '#D95319',  # orange
        '#EDB120',  # gelb
        '#77AC30',  # grün
        '#A2142F',  # rot
    ])
})
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score, Accuracy
#### Eigene Funktionen
from helperFunctions import createDataset, getFeaturesSequence
######################

torch.cuda.manual_seed_all(42)
DF = pd.read_csv("COMPLETE_DATASET.csv")
DF["date"] = pd.to_datetime(DF["date"])





def performFEATURESearch(FeaturesTuple, NameSpread):

    NameParams   = ['LookBack', 'Horizon', 'HiddenSize', 'LR', 'Epochs']
    ValueParams  = [lookback, horizon, hidden_Size, learningRate, epochs]
    dictCONF     = dict(zip(NameParams, ValueParams))
    feature_label = FeaturesTuple[0] 
    Features      = FeaturesTuple[1:]
    Spread      = Features[0]
    other_feats = Features[1:]

    # print(NameSpread)
    Spread     = Features[0]
    FeatureLEN = len(Features)
    dataFeatures = np.stack(Features, axis=1).astype("float32")

    print("DONE - Created the Features DATA")

    total_n    = len(Spread)
    used_n     = int(total_n * TotalUsed)
    features_used   = dataFeatures[:used_n]
    train_size   = int(used_n * TrainTestS)
    train_vals      = features_used[:train_size]
    test_vals       = features_used[train_size:]

    train_size  = int(len(Spread) * 0.8)
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold = 0
    for train_index, test_index in tscv.split(dataFeatures):
        fold += 1
        train_vals = dataFeatures[train_index]
        test_vals  = dataFeatures[test_index]
        X_train_full, Y_train_full  = createDataset(train_vals, lookback, horizon)
        X_test_full,  Y_test_full  = createDataset(test_vals,  lookback, horizon)
        
        print(f"Fold {fold}: train Größe = {len(train_index)}, test Größe = {len(test_index)}")
        with torch.no_grad():
            # letzter bekannter Spread in jedem Test-Sample
            naive_pred = X_test_full[:, -1, 0]
            true_vals  = Y_test_full[:, -1, 0]

            # MSE / RMSE
            mse_naive  = torch.mean((naive_pred - true_vals) ** 2).item()
            rmse_naive = np.sqrt(mse_naive)

            # MAE
            mae_naive = torch.mean(torch.abs(naive_pred - true_vals)).item()

            # R^2
            var_true = torch.var(true_vals, unbiased=False).item()
            r2_naive = 1 - mse_naive / var_true

            # Accuracy (immer "Up" als Baseline)
            true_dir  = (true_vals > naive_pred).long()
            pred_dir  = torch.ones_like(true_dir)          # immer Up
            acc_naive = (pred_dir == true_dir).float().mean().item()

    
        print(f"Naive   RMSE={rmse_naive:.4f}, MAE={mae_naive:.4f}, R²={r2_naive:.4f}, Acc={acc_naive:.4f}")

        # jetzt die echten Y-Vektoren bauen
        Y_train = Y_train_full[:, -1, 0].unsqueeze(1)
        Y_val   = Y_test_full[:,  -1, 0].unsqueeze(1)

        # 2) Move to GPU
        X_train = X_train_full.to(device)
        Y_train = Y_train.to(device)
        X_val   = X_test_full.to(device)
        Y_val   = Y_val.to(device)

        # DataLoader
        loader = data.DataLoader(data.TensorDataset(X_train, Y_train),batch_size=8,shuffle=True)

        # Modell erstellen und auf GPU schieben
        class SpreadLSTM(nn.Module):
            def __init__(self, inputsize):
                super().__init__()
                self.lstm   = nn.LSTM(input_size=inputsize,
                                    hidden_size=hidden_Size,
                                    num_layers=1,
                                    batch_first=True,
                                    dropout=0.2)
                self.linear = nn.Linear(hidden_Size, 1) # Matrixmuliplikation (Dimensionreduktion, Kombination der Features) 

            def forward(self, x):
                out_seq, _   = self.lstm(x)         # Abfolge der Hidden State Vektoren, _ (Hidden, Cell werden ignoriert)
                last_hidden = out_seq[:, -1, :]     # Nur den letzten versteckten Zustand
                return self.linear(last_hidden)        


        # print(FeatureLEN)
        model     = SpreadLSTM(inputsize=FeatureLEN).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learningRate)
        loss_fn   = nn.MSELoss()
        MSE_metric = MeanSquaredError().to(device)
        MAE_metric = MeanAbsoluteError().to(device)
        R2S_metric = R2Score().to(device)
        ACC_metric = Accuracy(task="binary", threshold=0.5).to(device)

        # Training
        RMSE_Train, RMSE_Validation = [], []
        MAE_Train, MAE_Validation = [], []
        R2S_Train  , R2S_Validation   = [], []
        ACC_Train, ACC_Validation = [], []


        for epoch in range(1, epochs+1):
            model.train()
            for X_b, Y_b in loader:
                optimizer.zero_grad()
                y_hat = model(X_b)                
                y_true = Y_b[:, -1].unsqueeze(1)     
                loss   = loss_fn(y_hat, y_true)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                # --- Train ---
                PR_Train = model(X_train).squeeze(-1)            
                GT_Train  = Y_train.squeeze(-1)

                MSE_metric.update(PR_Train, GT_Train)
                rmse_tr = torch.sqrt(MSE_metric.compute()).item()
                MSE_metric.reset()
                RMSE_Train.append(rmse_tr)

                # MAE
                MAE_metric.update(PR_Train, GT_Train)
                mae_tr = MAE_metric.compute().item()
                MAE_metric.reset()
                MAE_Train.append(mae_tr)        

                # R2
                R2S_metric.update(PR_Train, GT_Train)
                r2_tr = R2S_metric.compute().item()
                R2S_metric.reset()
                R2S_Train.append(r2_tr)

                # ACC
                Prev_Train = X_train_full[:, -1, 0].to(device)

                # binary: 1 wenn nächster Spread > letzter Spread
                pred_sig_tr = (PR_Train > Prev_Train).long()
                actual_sig_tr = (GT_Train > Prev_Train).long()

                ACC_metric.update(pred_sig_tr, actual_sig_tr)
                acc_tr = ACC_metric.compute().item()
                ACC_metric.reset()
                ACC_Train.append(acc_tr)


                # --- Test ---
                PR_Validation = model(X_val).squeeze(-1)
                GT_Validation = Y_val.squeeze(-1).to(device)

                # RMSE
                MSE_metric.update(PR_Validation, GT_Validation)
                rmse_te = torch.sqrt(MSE_metric.compute()).item()
                MSE_metric.reset()
                RMSE_Validation.append(rmse_te)

                # MAE
                MAE_metric.update(PR_Validation, GT_Validation)
                mae_te = MAE_metric.compute().item()
                MAE_metric.reset()
                MAE_Validation.append(mae_te)

                # R2
                R2S_metric.update(PR_Validation, GT_Validation)
                r2_te = R2S_metric.compute().item()
                R2S_metric.reset()
                R2S_Validation.append(r2_te)

                Prev_Val = X_test_full[:, -1, 0].to(device)

                pred_sig_val = (PR_Validation > Prev_Val).long()
                actual_sig_val = (GT_Validation > Prev_Val).long()

                ACC_metric.update(pred_sig_val, actual_sig_val)
                acc_val = ACC_metric.compute().item()
                ACC_metric.reset()
                ACC_Validation.append(acc_val)



            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train RMSE {rmse_tr:.4f}, Test RMSE {rmse_te:.4f}")
        
        
        # === Werte für CSV vorbereiten ===
        dictVALUESLast = {
            'RMSE_Train': RMSE_Train[-1],
            'RMSE_Valid':   RMSE_Validation[-1],
            'MAE_Train':  MAE_Train[-1],
            'MAE_Valid':    MAE_Validation[-1],
            'R2_Train':   R2S_Train[-1],
            'R2_Valid':     R2S_Validation[-1],
        }
        CONF = {** dictCONF, ** dictVALUESLast}

        dictVALUES = {
            'RMSE_Train': RMSE_Train,
            'RMSE_Valid':   RMSE_Validation,
            'MAE_Train':  MAE_Train,
            'MAE_Valid':    MAE_Validation,
            'R2_Train':   R2S_Train,
            'R2_Valid':     R2S_Validation,
            'ACC_Train': ACC_Train,
            'ACC_Valid': ACC_Validation,
        }
        DFMetrics = pd.DataFrame(dictVALUES)
        DFMetrics.to_csv(f"RESULTSADDFEATURES/BIAS/LSTM_RWDA_h-{horizon}_lookb-{lookback}_hiddSize-{hidden_Size}_lr-{learningRate}_ep-{epochs}_F-{feature_label}_FOLD_{fold}.csv", index=False, encoding='utf-8')

        # Modell abspeichern
        if fold == 5:
            model_path = f"MODELS/LSTM_h{horizon}_lb{lookback}_hs{hidden_Size}_lr{learningRate}_ep{epochs}_{feature_label}_{NameSpread}_BACKTEST.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'lookback': lookback,
                'horizon': horizon,
                'inputsize': FeatureLEN,
            }, model_path)
            print(f"Model saved to {model_path}")



    return CONF



########## MAIN ##########

# Hyperparameter
SpreadArray = ["spreadz_KB_RF"  ,"spreadz_SAN_DB","spreadz_KB_KEY","spreadz_PNC_SCHW" ,"spreadz_KEY_HBAN","spreadz_ALLY_DB","spreadz_CMA_TFC"]
for aSpread in SpreadArray:
    Horizon = [40]
    for ahor in Horizon: 
        
        lookback = 10
        horizon = ahor
        hidden_Size = 15
        learningRate = 20e-5
        epochs = 70

        TotalUsed  = 1
        TrainTestS = 0.8

        PLOT = False


        ## Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on", device)


        DataTUPLE = getFeaturesSequence(aSpread, False, False, False, False, False, False, False, False, False, False)

        performFEATURESearch(DataTUPLE, aSpread)



    # feature_names = ["CLOSE", "SP500" ,"MACD", "ATR", "TENY", "CRESPREAD", "BANK", "NEWS" ,"IV10", "IV20"]

    # combinations = []
    # combinations.append([False] * len(feature_names))
    # for i in range(len(feature_names)):
    #     combo = [False] * len(feature_names)
    #     combo[i] = True
    #     combinations.append(combo)
    # combinations.append([True] * len(feature_names))

    # # print(combinations)
    # SP500=False, TENY=False, CRESPREAD=False, BANK=False, IV10=False, IV20=False, CLOSE=True, NEWS=False,

    # all_results = {}
    # for flags in combinations: 
    #     DataTUPLE = getFeaturesSequence("spreadz_PNC_SCHW",  *flags,  lookahead_bias=True)
    #     feature_label = DataTUPLE[0]
    #     print(f"\n=== Running with: {feature_label} ===")
    #     # Modell trainieren / auswerten
    #     conf = performFEATURESearch(DataTUPLE, "spreadz_PNC_SCHW")
    #     all_results[feature_label] = conf

    # for label, conf in all_results.items():
    #     print(f"{label}: {conf}")