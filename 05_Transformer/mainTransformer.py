import math
import pandas as pd
from itertools import product
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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score, Accuracy
from sklearn.model_selection import TimeSeriesSplit
#### Eigene Funktionen
from helperFunctions import createDataset, getFeaturesSequence
######################
torch.cuda.manual_seed_all(42)



DF = pd.read_csv("COMPLETE_DATASET.csv")
DF["date"] = pd.to_datetime(DF["date"])



def performFEATURESearch(FeaturesTuple, SpreadName):

    SpreadNameplot = SpreadName


    NameParams  = ['LookBack', 'Horizon', 'DModel', 'NHead', 'NumLayers', 'LR', 'Epochs']
    ValueParams = [lookback, horizon, DModel, NHead, NUMLayers, learningRate, epochs]
    dictCONF = dict(zip(NameParams, ValueParams))

    feature_label = FeaturesTuple[0] 
    Features = FeaturesTuple[1:]

    Spread = Features[0]
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


        Y_train = Y_train_full[:, -1, 0].unsqueeze(1)
        Y_val   = Y_test_full[:,  -1, 0].unsqueeze(1)

        # 2) Move to GPU
        X_train = X_train_full.to(device)
        Y_train = Y_train.to(device)
        X_val   = X_test_full.to(device)
        Y_val   = Y_val.to(device)

        # DataLoader
        loader = data.DataLoader(data.TensorDataset(X_train, Y_train),batch_size=8,shuffle=True,)

        # Modell erstellen und auf GPU schieben
        class PositionalEncodingSINCOS(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term) # gerade Ind
                pe[:, 1::2] = torch.cos(position * div_term) # ungerade Ind
                pe = pe.unsqueeze(1)
                self.register_buffer('pe', pe) # nicht mitlernen

            def forward(self, x):
                seq_len = x.size(0)
                x = x + self.pe[:seq_len]
                return x
            

        class SpreadTransformer(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.1):
                super().__init__()
                # Feature-Embedding
                self.input_proj = nn.Linear(input_size, d_model)
                # PE
                self.pos_encoder = PositionalEncodingSINCOS(d_model)
                # Encoder
                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, dim_feedforward=d_model*4)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                # Head for Regression
                self.output_layer = nn.Linear(d_model, 1)

            def forward(self, x):
                x = x.permute(1, 0, 2)
                x = self.input_proj(x)
                x = self.pos_encoder(x)
                x = self.transformer_encoder(x)
                last = x[-1, :, :]
                out = self.output_layer(last)
                return out 

                
        model = SpreadTransformer(input_size=FeatureLEN, d_model=DModel, nhead=NHead, num_layers=NUMLayers, dropout=0.1).to(device)
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
        print(f"Train RMSE : {dictVALUESLast['RMSE_Train']:.4f}")
        print(f"Valid RMSE : {dictVALUESLast['RMSE_Valid']:.4f}")
        print(f"Train MAE  : {dictVALUESLast['MAE_Train']:.4f}")
        print(f"Valid MAE  : {dictVALUESLast['MAE_Valid']:.4f}")

        print(f"Train R²   : {dictVALUESLast['R2_Train']:.4f}")
        print(f"Valid R²   : {dictVALUESLast['R2_Valid']:.4f}")
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
        DFMetrics.to_csv(f"RESULTSADDFEATURES/RAWData/TRANSFORMER_RWDA_h-{horizon}dModel-{DModel}NumHeads-{NHead}_NumLay-{NUMLayers}_lr-{learningRate}_ep-{epochs}_F-{feature_label}_FOLD_{fold}_{SpreadNameplot}.csv", index=False, encoding='utf-8')

        # Modell abspeichern
        if fold == 5:
            model_path = f"MODELS/TRANSFORMER_h-{horizon}dModel-{DModel}NumHeads-{NHead}_NumLay-{NUMLayers}_lr-{learningRate}_ep-{epochs}_F-{feature_label}_{SpreadNameplot}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'lookback': lookback,
                'horizon': horizon,
                'inputsize': FeatureLEN,
            }, model_path)
            print(f"Model saved to {model_path}")



    return CONF



########## MAIN ##########
Horion = [15]

for aHor in Horion: 
    lookback = 10
    horizon  = aHor
    epochs   = 60
    PLOT     = True


    # Hyperparameter
    DModel = 16
    NHead = 8
    if (DModel % NHead != 0):
        print("Muss teilbar sein !!")
        quit()
        

    NUMLayers = 1
    learningRate = 50e-4

    lookback = 15

    PLOT = False

    ## Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    # spreadz_PNC_SCHW
    # spreadz_KB_RF
    TotalUsed  = 1
    TrainTestS = 0.8

    SpreadArray = [
                "spreadz_KB_RF","spreadz_SAN_DB","spreadz_KB_KEY",
                "spreadz_PNC_SCHW","spreadz_KEY_HBAN","spreadz_ALLY_DB","spreadz_CMA_TFC"
            ]

    for aSpread in SpreadArray:
        DataTUPLE = getFeaturesSequence(
            aSpread,
            False,  # SP500
            False,  # TENY
            False,  # CRESPREAD
            False,  # BANK
            False,  # IV10
            False,  # IV20
            False,   # NEWS
            False,  # CLOSE
            False,   # ATR
            False    # MACD
        )

        performFEATURESearch(DataTUPLE, aSpread)













# Grid
param_grid = {
    'NUMLayers': [1, 2, 3],
    'learningRate': [1e-3, 1e-4, 1e-5],
    'DModel': [8, 12, 16, 32],
    'NHead': [2, 4, 8],
}

results = []

for NL, LR, DM, NH in product(param_grid['NUMLayers'],
                              param_grid['learningRate'],
                              param_grid['DModel'],
                              param_grid['NHead']):

    NUMLayers   = NL
    learningRate = LR
    DModel      = DM
    NHead       = NH
    
    if DModel % NHead != 0:
        continue
    
    conf = performFEATURESearch(DataTUPLE)  
    results.append({
        'NUMLayers':   NL,
        'learningRate': LR,
        'DModel':      DM,
        'NHead':       NH,
        'MAE_Valid':   conf['MAE_Valid'],
        'RMSE_Valid':  conf['RMSE_Valid'],
        'R2_Valid':    conf['R2_Valid']
    })

df_results = pd.DataFrame(results)

best = df_results.loc[df_results['MAE_Valid'].idxmin()]
df_results.to_csv("grid_search_results.csv", index=False, encoding="utf-8")
print("Beste Konfiguration nach Valid-MAE:")
print(best.to_frame().T)