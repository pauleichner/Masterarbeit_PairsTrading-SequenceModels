import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random

from sklearn.model_selection import train_test_split
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score, Accuracy

#### Eigene Funktionen ################
from helperFunctions import createDataset, getFeaturesSequence
#######################################################

torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

DF = pd.read_csv("COMPLETE_DATASET.csv")
DF["date"] = pd.to_datetime(DF["date"])

def load_spread_sequence(spread_name, use_features=False):
    DataTUPLE = getFeaturesSequence(spread_name, *( [True]*10 if use_features else [False]*10 ))
    feature_label = DataTUPLE[0]
    Features = DataTUPLE[1:]
    spread_features = np.stack(Features, axis=1).astype("float32")
    return spread_features, feature_label


def prepare_splits_for_spread(spread_name, use_features=False, lookback=15, horizon=1):
    features, feature_label = load_spread_sequence(spread_name, use_features)
    X_full, Y_full = createDataset(features, lookback, horizon) # Anfang bei lookback +1 | Ende bei +Horizon
    indices = np.arange(len(X_full))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, shuffle=False)
    return X_full, Y_full, feature_label, train_idx, test_idx

class MultiSpreadDataset(data.Dataset):
    def __init__(self, spread_names, spread_features, spread_labels, splits_idx):
        self.mapping = []
        self.spread_features = spread_features
        self.spread_labels   = spread_labels

        for name in spread_names:
            for t in splits_idx[name]:
                self.mapping.append((name, int(t)))

    def __len__(self):
        print(len(self.mapping))
        return len(self.mapping)


    def __getitem__(self, i):
        spread, t = self.mapping[i]
        # print(spread)
        # print(t)
        X = self.spread_features[spread][t]
        y = self.spread_labels[spread][t, -1, 0]
        return X.float(), y.float()

    
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

def performMultiSpreadTraining(SpreadArray,lookback,horizon,learningRate,epochs):
    spread_feats, spread_labs = {}, {}
    train_idxs, test_idxs     = {}, {}
    for spread in SpreadArray:
        X_full, Y_full, feat_lbl, trn, tst = prepare_splits_for_spread(
            spread, use_features=False,
            lookback=lookback, horizon=horizon
        )
        # print(f"{spread}: {len(trn)} Train‐Windows, {len(tst)} Test‐Windows (von insgesamt {X_full.shape[0]} Fenstern)")
        spread_feats[spread] = X_full
        spread_labs[spread]  = Y_full
        train_idxs[spread]   = trn
        test_idxs[spread]    = tst

    train_ds = MultiSpreadDataset(SpreadArray, spread_feats, spread_labs, train_idxs)
    test_ds  = MultiSpreadDataset(SpreadArray, spread_feats, spread_labs, test_idxs)

    train_loader = data.DataLoader(
        train_ds, batch_size=32, shuffle=True,
        drop_last=False, pin_memory=True
    )
    test_loader  = data.DataLoader(
        test_ds, batch_size=32, shuffle=False,
        drop_last=False, pin_memory=True
    )

    FeatureLEN = next(iter(spread_feats.values())).shape[2]
    model = SpreadTransformer(
        input_size=FeatureLEN,
        d_model=DModel,
        nhead=NHead,
        num_layers=NUMLayers,
        dropout=0.1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    mse_met = MeanSquaredError().to(device)
    mae_met = MeanAbsoluteError().to(device)
    r2_met  = R2Score().to(device)
    acc_met = Accuracy(task="binary", threshold=0.5).to(device)

    history = {
        'rmse_train': [], 'rmse_test': [],
        'mae_train':  [], 'mae_test':  [],
        'r2_train':   [], 'r2_test':   [],
        'acc_train':  [], 'acc_test':  []
    }

    for epoch in range(1, epochs+1):
        model.train()
        for X_b, y_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(X_b)
            loss  = loss_fn(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            # -- Train --
            all_p, all_t = [], []
            for X_b, y_b in train_loader:
                X_b = X_b.to(device)
                p   = model(X_b).squeeze(1).cpu()
                all_p.append(p)
                all_t.append(y_b)
            P_tr = torch.cat(all_p)
            T_tr = torch.cat(all_t)

            rmse_tr = torch.sqrt(mse_met(P_tr, T_tr)).item()
            mae_tr  = mae_met(P_tr, T_tr).item()
            r2_tr   = r2_met (P_tr, T_tr).item()

            prevs_tr = []
            for spread, idxs in train_idxs.items():
                feats = spread_feats[spread][idxs]
                prevs_tr.append(feats[:, -1, 0])
            prevs_tr = torch.cat(prevs_tr)
            acc_tr = acc_met((P_tr > prevs_tr), (T_tr > prevs_tr)).item()

            # -- Test --
            all_p, all_t = [], []
            for X_b, y_b in test_loader:
                X_b = X_b.to(device)
                p   = model(X_b).squeeze(1).cpu()
                all_p.append(p)
                all_t.append(y_b)
            P_te = torch.cat(all_p)
            T_te = torch.cat(all_t)

            rmse_te = torch.sqrt(mse_met(P_te, T_te)).item()
            mae_te  = mae_met(P_te, T_te).item()
            r2_te   = r2_met (P_te, T_te).item()

            prevs_te = []
            for spread, idxs in test_idxs.items():
                feats = spread_feats[spread][idxs]
                prevs_te.append(feats[:, -1, 0])
            prevs_te = torch.cat(prevs_te)
            acc_te = acc_met((P_te > prevs_te), (T_te > prevs_te)).item()

        history['rmse_train'].append(rmse_tr)
        history['rmse_test'].append(rmse_te)
        history['mae_train'].append(mae_tr)
        history['mae_test'].append(mae_te)
        history['r2_train'].append(r2_tr)
        history['r2_test'].append(r2_te)
        history['acc_train'].append(acc_tr)
        history['acc_test'].append(acc_te)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train RMSE {rmse_tr:.4f} | Test RMSE {rmse_te:.4f} | Test R² {r2_te:.4f}")

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(
        f"RESULTS/MULTI_TRANSFORMER_h{horizon}_lookb-{lookback}_hiddSize-{NHead}_lr-{learningRate}_ep-{epochs}.csv",
        index=False
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'lookback': lookback,
        'horizon': horizon,
        'inputsize': FeatureLEN,
        'spread_columns': SpreadArray,
    }, f"MODELS/MULTI_TRANSFORMER_h{horizon}_lookb-{lookback}_hiddSize-{NHead}_lr-{learningRate}_ep-{epochs}.pth")
    print("Model und History gespeichert.")

    return {
        'RMSE_Test': history['rmse_test'][-1],
        'R2_Test':  history['r2_test'][-1],
        'RMSE_Train': history['rmse_train'][-1],
        'R2_Train':  history['r2_train'][-1],
    }

########## MAIN ##########

# Hyperparameter
DModel = 16
NHead = 8
if (DModel % NHead != 0):
    print("Muss teilbar sein !!")
    quit()

lookback = 15

NUMLayers = 1
learningRate = 50e-4
epochs = 100
PLOT = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

# Alle verfügbaren Spreads
SpreadArray = [
    "spreadz_KB_RF", "spreadz_SAN_DB", "spreadz_KB_KEY", 
    "spreadz_PNC_SCHW", "spreadz_KEY_HBAN", "spreadz_ALLY_DB", 
    "spreadz_CMA_TFC"
]

print(f"Training with {len(SpreadArray)} spreads: {SpreadArray}")

# Verschiedene Horizonte testen
Horizon = [10, 15]

for horizon in Horizon:
    print(f"\n{'='*50}")
    print(f"TRAINING FOR HORIZON = {horizon}")
    print(f"{'='*50}")
    
    print("\n--- Training without additional features ---")
    conf_no_features = performMultiSpreadTraining(SpreadArray,lookback,horizon,learningRate,epochs)
    
    print(f"\n--- COMPARISON FOR HORIZON {horizon} ---")
    print(f"Without features - Test RMSE: {conf_no_features['RMSE_Test']:.4f}, Test R²: {conf_no_features['R2_Test']:.4f}")
    # print(f"With features    - Test RMSE: {conf_with_features['RMSE_Test']:.4f}, Test R²: {conf_with_features['R2_Test']:.4f}")


print("\n" + "="*50)
print("MULTI-SPREAD TRAINING COMPLETED!")
print("="*50)