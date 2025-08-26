import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch.nn.functional as F
import math
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
        Y_train_dir  = (Y_train_full[:, -1, 0] > last_train).float().unsqueeze(1)
        Y_test_dir   = (Y_test_full[:,  -1, 0] > last_test).float().unsqueeze(1)

        X_train = X_train_full.to(device)
        Y_train = Y_train_dir.to(device)
        X_val   = X_test_full.to(device)
        Y_val   = Y_test_dir.to(device)

        # DataLoader
        loader = data.DataLoader(data.TensorDataset(X_train, Y_train),
                                 batch_size=8, shuffle=True)


        class sLSTMCell(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int):
                super(sLSTMCell, self).__init__()
                self.W_f = nn.Linear(input_dim, hidden_dim, bias=False)
                self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.b_f = nn.Parameter(torch.zeros(hidden_dim))

                self.W_i = nn.Linear(input_dim, hidden_dim, bias=False)
                self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.b_i = nn.Parameter(torch.zeros(hidden_dim))

                self.W_g = nn.Linear(input_dim, hidden_dim, bias=False)
                self.U_g = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.b_g = nn.Parameter(torch.zeros(hidden_dim))

                self.W_o = nn.Linear(input_dim, hidden_dim, bias=False)
                self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.b_o = nn.Parameter(torch.zeros(hidden_dim))

            def forward(
                self,
                x_zo: torch.Tensor,
                x_if: torch.Tensor,
                h_prev: torch.Tensor,
                c_prev: torch.Tensor,
                n_prev: torch.Tensor,
                m_prev: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                i_raw = self.W_i(x_if) + self.U_i(h_prev) + self.b_i
                f_raw = self.W_f(x_if) + self.U_f(h_prev) + self.b_f

                z_raw = self.W_g(x_zo) + self.U_g(h_prev) + self.b_g
                o_raw = self.W_o(x_zo) + self.U_o(h_prev) + self.b_o

                max_term = torch.maximum(f_raw + m_prev, i_raw)
                m_t      = max_term
                i_t      = torch.exp(i_raw - m_t)
                f_t      = torch.exp(f_raw + m_prev - m_t)

                z_t      = torch.tanh(z_raw)
                n_t      = f_t * n_prev + i_t
                c_t      = f_t * c_prev + i_t * z_t

                o_t      = torch.sigmoid(o_raw)
                h_tilde  = c_t.div(n_t.clamp(min=1e-8))
                h_t      = o_t * h_tilde

                return h_t, c_t, n_t, m_t


        class sLSTMCell(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int):
                super(sLSTMCell, self).__init__()
                self.W_f = nn.Linear(input_dim, hidden_dim, bias=False)
                self.r_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.b_f = nn.Parameter(torch.zeros(hidden_dim))

                self.W_i = nn.Linear(input_dim, hidden_dim, bias=False)
                self.r_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.b_i = nn.Parameter(torch.zeros(hidden_dim))

                self.W_z = nn.Linear(input_dim, hidden_dim, bias=False)
                self.r_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.b_z = nn.Parameter(torch.zeros(hidden_dim))

                self.W_o = nn.Linear(input_dim, hidden_dim, bias=False)
                self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.b_o = nn.Parameter(torch.zeros(hidden_dim))

            def forward(
                self,
                x_zo: torch.Tensor,
                x_if: torch.Tensor,
                h_prev: torch.Tensor,
                c_prev: torch.Tensor,
                n_prev: torch.Tensor,
                m_prev: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                i_raw = self.W_i(x_if) + self.r_i(h_prev) + self.b_i
                f_raw = self.W_f(x_if) + self.r_f(h_prev) + self.b_f

                z_raw = self.W_z(x_zo) + self.r_z(h_prev) + self.b_z
                o_raw = self.W_o(x_zo) + self.U_o(h_prev) + self.b_o

                max_term = torch.maximum(f_raw + m_prev, i_raw)
                m_t      = max_term
                i_t      = torch.exp(i_raw - m_t)
                f_t      = torch.exp(f_raw + m_prev - m_t)

                z_t      = torch.tanh(z_raw)
                n_t      = f_t * n_prev + i_t
                c_t      = f_t * c_prev + i_t * z_t

                o_t      = torch.sigmoid(o_raw)
                h_tilde  = c_t.div(n_t.clamp(min=1))
                h_t      = o_t * h_tilde

                return h_t, c_t, n_t, m_t


        class mLSTMCell(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.W_q = nn.Linear(input_dim, hidden_dim, bias=True)   # Query
                self.W_k = nn.Linear(input_dim, hidden_dim, bias=True)   # Key
                self.W_v = nn.Linear(input_dim, hidden_dim, bias=True)   # Value
            
                self.w_i = nn.Linear(input_dim, 1, bias=True)
                self.w_f = nn.Linear(input_dim, 1, bias=True)
                self.w_o = nn.Linear(input_dim, 1, bias=True)
                nn.init.constant_(self.w_f.bias, 1.0)

            def forward(self, x: torch.Tensor, C_prev: torch.Tensor, n_prev: torch.Tensor, m_prev: torch.Tensor,):
                batch_size = x.size(0)
                d = self.hidden_dim
                # + b_* wird intern in der Klasse gemacht
                q_t = self.W_q(x)
                k_t = (1.0 / math.sqrt(d)) * self.W_k(x)
                v_t = self.W_v(x)
                i_logit = self.w_i(x)
                f_logit = self.w_f(x)
                o_logit = self.w_o(x)
                m_t = torch.maximum(f_logit + m_prev, i_logit)
                i_t = torch.exp(i_logit - m_t)
                f_t = torch.exp(f_logit + m_prev - m_t)
                outer_vk = v_t.unsqueeze(2) * k_t.unsqueeze(1)
                C_t = f_t.view(batch_size, 1, 1) * C_prev + i_t.view(batch_size, 1, 1) * outer_vk
                n_t = f_t * n_prev.unsqueeze(2).view(batch_size, d) + i_t.view(batch_size, 1) * k_t
                o_t = torch.sigmoid(o_logit)
                Cq = torch.bmm(C_t, q_t.unsqueeze(2)).squeeze(2)
                inner = torch.sum(n_t * q_t, dim=1, keepdim=True)
                denom = torch.maximum(torch.abs(inner), torch.ones_like(inner))
                h_t = o_t.view(batch_size, 1) * (Cq / denom)
                return C_t, n_t, m_t, h_t


        class sLSTMBlock(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4, conv_kernel_size: int = 4, mlp_factor: float = 4/3):
                super().__init__()
                self.pre_norm = nn.LayerNorm(input_dim)
                if conv_kernel_size > 1:
                    self.conv_if = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size-1, bias=True)
                    self.swish = nn.SiLU()
                else:
                    self.conv_if = None
                self.cell = sLSTMCell(input_dim, hidden_dim)
                self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=hidden_dim)
                up_dim = int(hidden_dim * mlp_factor)
                self.mlp_up = nn.Linear(hidden_dim, up_dim)
                self.mlp_down = nn.Linear(up_dim, input_dim)
                self.gelu = nn.GELU()
                self.o_ext = nn.Linear(input_dim, input_dim, bias=True)

            def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor, n_prev: torch.Tensor, m_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                x_norm = self.pre_norm(x_t)
                if self.conv_if is not None:
                    conv_input = x_norm.unsqueeze(2)
                    conv_out = self.conv_if(conv_input)
                    conv_out = conv_out[..., :1].squeeze(2)
                    x_for_gates = self.swish(conv_out)
                else:
                    x_for_gates = x_norm

                h_t, c_t, n_t, m_t = self.cell(x_norm, x_for_gates, h_prev, c_prev, n_prev, m_prev)

                h_norm = self.group_norm(h_t)
                u = self.mlp_up(h_norm)
                u = self.gelu(u)
                d = self.mlp_down(u)
                o_ext = torch.sigmoid(self.o_ext(x_norm))
                gated_out = o_ext * d
                y_t = x_t + gated_out

                return h_t, c_t, n_t, m_t, y_t


        class mLSTMBlock(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4, conv_kernel_size: int = 4, mlp_up_factor: float = 2.0):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.num_heads = num_heads
                self.pre_norm = nn.LayerNorm(input_dim)

                up_dim = int(input_dim * mlp_up_factor)
                self.up_proj = nn.Linear(input_dim, up_dim)
                self.W_o_ext = nn.Linear(input_dim, input_dim, bias=True)

                if conv_kernel_size > 1:
                    self.conv_kv = nn.Conv1d(
                        in_channels = up_dim - input_dim,
                        out_channels = hidden_dim,
                        kernel_size = conv_kernel_size,
                        padding = conv_kernel_size - 1,
                        bias = True
                    )
                    self.swish = nn.SiLU()
                else:
                    self.conv_kv = None

                self.skip_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
                self.cell = mLSTMCell(hidden_dim, hidden_dim)
                self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=hidden_dim)
                self.down_proj = nn.Linear(hidden_dim, input_dim)

            def forward(self, x_t: torch.Tensor, C_prev: torch.Tensor, n_prev: torch.Tensor, m_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                x_norm = self.pre_norm(x_t)
                up = self.up_proj(x_norm)
                o_ext_logits = self.W_o_ext(up[:, :self.input_dim])
                o_ext = torch.sigmoid(o_ext_logits)

                m_input = up[:, self.input_dim:]

                if self.conv_kv is not None:
                    conv_in = m_input.unsqueeze(2)
                    conv_out = self.conv_kv(conv_in)
                    conv_out = conv_out[..., :1].squeeze(2)
                    conv_out = self.swish(conv_out)
                    m_lstm_in = conv_out
                else:
                    m_lstm_in = F.pad(m_input, (0, self.hidden_dim - m_input.size(1)), "constant", 0.0)

                skip = self.skip_proj(m_lstm_in)
                C_t, n_t, m_t, h_t_high = self.cell(m_lstm_in, C_prev, n_prev, m_prev)
                h_norm = self.group_norm(h_t_high)
                merged = skip + h_norm
                down = self.down_proj(merged)
                gated = o_ext * down
                y_t = x_t + gated

                return C_t, n_t, m_t, y_t


        class XLSTM(nn.Module):
            def __init__( self, input_dim: int, hidden_dim: int, n_s: int, n_m: int, num_heads: int = 4, conv_kernel_size: int = 4, mlp_factor_s: float = 4/3, mlp_up_factor_m: float = 2.0):
                super().__init__()
                self.input_dim  = input_dim
                self.hidden_dim = hidden_dim
                self.n_s        = n_s
                self.n_m        = n_m

                self.s_blocks = nn.ModuleList([
                    sLSTMBlock(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        conv_kernel_size=conv_kernel_size,
                        mlp_factor=mlp_factor_s)
                    for _ in range(n_s)])

                self.m_blocks = nn.ModuleList([
                    mLSTMBlock(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        conv_kernel_size=conv_kernel_size,
                        mlp_up_factor=mlp_up_factor_m)
                    for _ in range(n_m)])


            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, D = x.size()
                out = x

                for block in self.s_blocks:
                    h_prev = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
                    c_prev = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
                    n_prev = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
                    m_prev = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)

                    y_seq = []
                    for t in range(T):
                        x_t = out[:, t, :]  # (B, D)
                        h_t, c_t, n_t, m_t, y_t = block(x_t, h_prev, c_prev, n_prev, m_prev)
                        y_seq.append(y_t.unsqueeze(1))  # (B, 1, D)

                        h_prev, c_prev, n_prev, m_prev = h_t, c_t, n_t, m_t

                    out = torch.cat(y_seq, dim=1)

                for block in self.m_blocks:
                    C_prev = torch.zeros(
                        B, self.hidden_dim, self.hidden_dim,
                        device=x.device, dtype=x.dtype
                    )
                    n_prev_m = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
                    m_prev_m = torch.zeros(B, 1, device=x.device, dtype=x.dtype)

                    y_seq = []
                    for t in range(T):
                        x_t = out[:, t, :]  # (B, D)
                        C_t, n_t_m, m_t_m, y_t = block(x_t, C_prev, n_prev_m, m_prev_m)
                        y_seq.append(y_t.unsqueeze(1))

                        C_prev, n_prev_m, m_prev_m = C_t, n_t_m, m_t_m

                    out = torch.cat(y_seq, dim=1)
                return out


        class SpreadLSTM(nn.Module):
            def __init__(
                self,
                input_dim: int,
                hidden_dim: int,
                n_s: int,
                n_m: int,
                num_heads: int = 4,
                conv_kernel_size: int = 4,
                mlp_factor_s: float = 4/3,
                mlp_up_factor_m: float = 2.0
            ):
                super().__init__()
                self.xlstm = XLSTM(
                    input_dim      = input_dim,
                    hidden_dim     = hidden_dim,
                    n_s            = n_s,
                    n_m            = n_m,
                    num_heads      = num_heads,
                    conv_kernel_size = conv_kernel_size,
                    mlp_factor_s   = mlp_factor_s,
                    mlp_up_factor_m = mlp_up_factor_m
                )
                self.head = nn.Linear(input_dim, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # XLSTM liefert (batch, seq_len, input_dim)
                out_seq = self.xlstm(x)
                # Den letzten Zeitschritt extrahieren:
                last_h = out_seq[:, -1, :]
                return self.head(last_h)     

        model = SpreadLSTM(
            input_dim    = FeatureLEN,
            hidden_dim   = hidden_Size,
            n_s          = 1,
            n_m          = 1,
            num_heads      = 4,
            conv_kernel_size = 4,
            mlp_factor_s  = 4/3,
            mlp_up_factor_m = 2.0
        ).to(device)

        # Loss & Optimizer
        loss_fn   = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learningRate)

        # Metriken initialisieren
        ACC_metric   = Accuracy(task="binary", threshold=0.5).to(device)
        ACC_Train, ACC_Valid = [], []


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

        # DataFrame mit Epochenverlauf
        DFMetrics = pd.DataFrame({
            'Epoch':       np.arange(1, epochs+1),
            'Acc_Train':   ACC_Train,
            'Acc_Valid':   ACC_Valid,})

        for k, v in dictCONF.items():
            DFMetrics[k] = v
        DFMetrics['Fold'] = fold

        DFMetrics.to_csv(
            f"RESULTSADDFEATURES/BINARY/"
            f"xLSTM_h{horizon}_lb{lookback}"
            f"_hs{hidden_Size}_lr{learningRate}"
            f"_ep{epochs}_F-{feature_label}_FOLD{fold}.csv",
            index=False,
            encoding='utf-8'
        )

        if fold == n_splits:
            model_path = (
                f"MODELS/xLSTM_BIN_h{horizon}_lb{lookback}"
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
        'LastValAcc': ACC_Valid[-1],}


########## MAIN ##########
Horizon = [1, 2, 5]
for ahor in Horizon:
    lookback = 10
    num_layers = 1
    horizon = ahor
    hidden_Size = 16
    learningRate = 0.0030
    epochs = 50

    TotalUsed  = 1
    TrainTestS = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    DataTUPLE = getFeaturesSequence(
        "spreadz_PNC_SCHW",
        False, False, False, False,
        False, False, False, False,
        False, False
    )
    performFEATURESearch(DataTUPLE, "spreadz_PNC_SCHW")
