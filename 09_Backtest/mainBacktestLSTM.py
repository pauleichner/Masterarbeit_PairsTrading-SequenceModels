import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import matplotlib.patches as patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## MODELS
hidden_Size = 15
lookback = 10

plt.style.use('classic')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    # "grid.linestyle": "--",
    "grid.alpha": 0.6,
})


class SpreadLSTM(nn.Module):
    def __init__(self, inputsize):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=inputsize,
                            hidden_size=hidden_Size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.2)
        self.linear = nn.Linear(hidden_Size, 1)

    def forward(self, x):
        out_seq, _   = self.lstm(x)
        last_hidden = out_seq[:, -1, :]
        return self.linear(last_hidden)  



## =================================================


def getHedgeRatioBETA(PA: pd.Series, PB: pd.Series) -> float:
    df = pd.concat([PA, PB], axis=1).dropna()
    X = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values
    model = LinearRegression().fit(X, y)
    return model.coef_[0]

def getTradeSignalSeriesDefault(Spread : pd.Series, StdThreshold : float) -> pd.Series:
    Signals = pd.Series
    Signals = np.where(Spread > StdThreshold, 1, np.where(Spread < -StdThreshold, -1, 0))
    return Signals


class Backtest:
    
    InitEquity = 10000
    # leverage_factor = 2

    def __init__(self, AssetA, AssetB, Spread, Signal, beta, Dates, Benchmark, Horizon, SpreadName):
        self.Dates = Dates
        self.SpreadName = SpreadName
        self.AssetA = AssetA
        self.AssetB = AssetB
        self.Spread = Spread
        self.Signal = Signal
        self.hedgeR = beta

        self.Equity = self.InitEquity
        self.GraphEquity = []
        self.TradeACTIVESpreadLONG = False
        self.TradeACTIVESpreadSHORT = False

        # Asset spezifisch
        self.entryPriceA = 0.0
        self.entryPriceB = 0.0
        self.currentSharesA = 0
        self.currentSharesB = 0
        
        
        # DeepLearning
        self.Horizon = Horizon
        self.Model = SpreadLSTM(inputsize=1).to(device=device)
        ckpt = torch.load(f"MODEL_LSTM/LSTM_h{self.Horizon}_lb10_hs15_lr0.0002_ep70_SpreadOnly_{self.SpreadName}_BACKTEST.pth", map_location=device)
        self.Model.load_state_dict(ckpt['model_state_dict'])
        self.Model.to(device)
        self.Model.eval()
        print(f"Loaded Mode: MODEL_LSTM/LSTM_h{self.Horizon}_lb10_hs15_lr0.0002_ep70_SpreadOnly_{self.SpreadName}_BACKTEST.pth" )

        self.Benchmark = Benchmark
        self.tradeEntries = []
        self.tradeExits   = []
        

    def performPredBacktest(self):
        for i, currentSpread in enumerate(self.Spread):
            if i < lookback:
                self.GraphEquity.append(self.Equity)
                continue
            
            self.GraphEquity.append(self.Equity)

            print(f"[INFO] Trade Day: {i} | Spread: {currentSpread}")

            if self.TradeACTIVESpreadSHORT:
                # Manage Trade
                if currentSpread < 0:
                    # Asset A wird verkauft | Asset B wird gekauft
                    PnLLong = self.__calcShortPnL(self.entryPriceA, self.AssetA[i], self.currentSharesA)
                    PnLShort = self.__calcLongPnL(self.entryPriceB, self.AssetB[i], self.currentSharesB)
                    PnLGes = PnLLong + PnLShort
                    self.Equity = self.Equity + PnLGes
                    self.TradeACTIVESpreadSHORT = False
                    self.tradeExits.append((i, currentSpread))
                    print("[TRADE] Short Spread Trade was closed")
            elif self.TradeACTIVESpreadLONG:
                if currentSpread > 0:
                    # Asset A wird gekauft | Asset B wird verkauft
                    PnLLong = self.__calcLongPnL(self.entryPriceA, self.AssetA[i], self.currentSharesA)
                    PnLShort = self.__calcShortPnL(self.entryPriceB, self.AssetB[i], self.currentSharesB)
                    PnLGes = PnLLong + PnLShort
                    self.Equity = self.Equity + PnLGes
                    self.TradeACTIVESpreadLONG = False
                    self.tradeExits.append((i, currentSpread))
                    print("[TRADE] LONG Spread Trade was closed")
            

            else:
                # Check for new Entries
                if self.Spread[i] > 1.5:
                    # Spread über 1.5 -> Short Asset A | Long Asset B
                    window = self.Spread.values[i-lookback:i].reshape(lookback, 1)  # np.ndarray (lookback, 1)
                    signal = self.LSTMPred(window)
                    print(f"Tag {i}: LSTM sagt {signal=}")
                    if signal == -1:
                        # print("[INFO] : Model predicted falling Spread")
                        self.entryPriceA = self.AssetA[i]
                        self.entryPriceB = self.AssetB[i]
                        self.currentSharesA = self.__getSharesA(self.AssetA[i])
                        self.currentSharesB = self.__getSharesB(self.AssetB[i])
                        self.TradeACTIVESpreadSHORT = True
                        self.tradeEntries.append((i, currentSpread))

                    
                elif self.Spread[i] < -1.5:
                    # Spread unter -1.5 -> Long Asset A | Short Asset B
                    window = self.Spread.values[i-lookback:i].reshape(lookback, 1)  # np.ndarray (lookback, 1)
                    signal = self.LSTMPred(window)
                    print(f"Tag {i}: LSTM sagt {signal=}")
                    if signal == 1:
                        # print("[INFO] : Model predicted rising Spread")
                        self.entryPriceA = self.AssetA[i]
                        self.entryPriceB = self.AssetB[i]
                        self.currentSharesA = self.__getSharesA(self.AssetA[i])
                        self.currentSharesB = self.__getSharesB(self.AssetB[i])
                        self.TradeACTIVESpreadLONG = True
                        self.tradeEntries.append((i, currentSpread))
                else:
                    continue        


    def performSimpleBacktest(self):
        
        for i, currentSpread in enumerate(self.Spread):
            self.GraphEquity.append(self.Equity)

            # print(f"[INFO] Trade Day: {i} | Spread: {currentSpread}")

            if self.TradeACTIVESpreadSHORT:
                # Manage Trade
                if currentSpread < 0:
                    # Asset A wird verkauft | Asset B wird gekauft
                    PnLLong = self.__calcShortPnL(self.entryPriceA, self.AssetA[i], self.currentSharesA)
                    PnLShort = self.__calcLongPnL(self.entryPriceB, self.AssetB[i], self.currentSharesB)
                    PnLGes = PnLLong + PnLShort
                    self.Equity = self.Equity + PnLGes
                    self.TradeACTIVESpreadSHORT = False
                    self.tradeExits.append((i, currentSpread))
                    # print("[TRADE] Short Spread Trade was closed")
            elif self.TradeACTIVESpreadLONG:
                if currentSpread > 0:
                    # Asset A wird gekauft | Asset B wird verkauft
                    PnLLong = self.__calcLongPnL(self.entryPriceA, self.AssetA[i], self.currentSharesA)
                    PnLShort = self.__calcShortPnL(self.entryPriceB, self.AssetB[i], self.currentSharesB)
                    PnLGes = PnLLong + PnLShort
                    self.Equity = self.Equity + PnLGes
                    self.TradeACTIVESpreadLONG = False
                    self.tradeExits.append((i, currentSpread))
                    # print("[TRADE] LONG Spread Trade was closed")
            

            else:
                # Check for new Entries
                if self.Signal[i] == 1:
                    # Spread über 1.5 -> Short Asset A | Long Asset B
                    self.entryPriceA = self.AssetA[i]
                    self.entryPriceB = self.AssetB[i]
                    self.currentSharesA = self.__getSharesA(self.AssetA[i])
                    self.currentSharesB = self.__getSharesB(self.AssetB[i])
                    self.TradeACTIVESpreadSHORT = True
                    self.tradeEntries.append((i, currentSpread))
                    
                elif self.Signal[i] == -1:
                    # Spread unter -1.5 -> Long Asset A | Short Asset B
                    self.entryPriceA = self.AssetA[i]
                    self.entryPriceB = self.AssetB[i]
                    self.currentSharesA = self.__getSharesA(self.AssetA[i])
                    self.currentSharesB = self.__getSharesB(self.AssetB[i])
                    self.TradeACTIVESpreadLONG = True
                    self.tradeEntries.append((i, currentSpread))
                else:
                    continue



    
    def __getSharesA(self, price_A: float) -> float:
        dollar_exp_A = self.Equity / 2
        return (dollar_exp_A / price_A) #* self.leverage_factor
    
    def __getSharesB(self, price_B: float) -> float:
        dollar_exp_B = (self.Equity / 2 ) * self.hedgeR
        return (dollar_exp_B / price_B) #* self.leverage_factor

    def __calcLongPnL(self, entryPrice, exitPrice, shares):
        return (exitPrice - entryPrice) * shares

    def __calcShortPnL(self, entryPrice, exitPrice, shares):
        return (entryPrice - exitPrice) * shares


    # change to antother by just implementing different self.Model
    def LSTMPred(self, window: np.ndarray) -> int: 
        # +1 -> Spread steigt weiter | -1 -> Spred fällt
        x    = torch.tensor(window, dtype=torch.float32, device=next(self.Model.parameters()).device).unsqueeze(0)
        pred = self.Model(x).cpu().item()
        return 1 if pred > window[-1,0] else -1    





    # =======================================================================================================
    # =============================================== Metrics ===============================================
    # =======================================================================================================
    def getPNL(self):
        print(f"Equity: {self.Equity} $")
        return self.Equity
    
    def plotEquityGraph(self):
        dates = pd.to_datetime(self.Dates)
        fig, ax = plt.subplots()
        ax.plot(dates, self.GraphEquity, label="Pairs Trading")
        BenchmarkShares = self.InitEquity / self.Benchmark.iloc[0]
        BaH_Equity = self.Benchmark * BenchmarkShares
        ax.plot(dates, BaH_Equity, label="Benchmark")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.show()

    def showTradesPlot(self):
        # Zeitachse
        dates = pd.to_datetime(self.Dates)
        fig, (ax_spread, ax_a, ax_b) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        numTrades = min(len(self.tradeEntries), len(self.tradeExits))

        ax_spread.plot(dates, self.Spread, color='black', label='Spread')
        if self.tradeEntries:
            entry_idx, _ = zip(*self.tradeEntries)
            ax_spread.scatter(dates[list(entry_idx)],
                              [self.Spread.iloc[i] for i in entry_idx],
                              marker='o', s=80, facecolors='none', edgecolors='blue',
                              label='Entry')
        if self.tradeExits:
            exit_idx, _ = zip(*self.tradeExits)
            ax_spread.scatter(dates[list(exit_idx)],
                              [self.Spread.iloc[i] for i in exit_idx],
                              marker='s', s=80, facecolors='none', edgecolors='red',
                              label='Exit')
        ax_spread.set_ylabel('Spread (z-Score)')
        ax_spread.grid(True)
        ax_spread.legend(loc='upper left', scatterpoints=1)

        # 2) Asset A Kurs
        ax_a.plot(dates, self.AssetA, label='Asset A', color="black")
        ax_a.set_ylabel('KB')
        ax_a.grid(True)

        # 3) Asset B Kurs
        ax_b.plot(dates, self.AssetB, label='Asset B', color="black")
        ax_b.set_ylabel('RF')
        ax_b.grid(True)

        for idx in range(numTrades):
            entry_i, _ = self.tradeEntries[idx]
            exit_i, _ = self.tradeExits[idx]
            if exit_i <= entry_i:
                continue

            entry_date, exit_date = dates[entry_i], dates[exit_i]
            priceA_entry, priceA_exit = self.AssetA[entry_i], self.AssetA[exit_i]
            priceB_entry, priceB_exit = self.AssetB[entry_i], self.AssetB[exit_i]
            sig = self.Signal[entry_i]

            marker_A = 'v' if sig == 1 else '^'
            color_A = 'red' if sig == 1 else 'green'
            marker_B = '^' if sig == 1 else 'v'
            color_B = 'green' if sig == 1 else 'red'

            ax_a.scatter(entry_date, priceA_entry, marker=marker_A, s=80, color=color_A, label='Entry A' if idx == 0 else '')
            ax_b.scatter(entry_date, priceB_entry, marker=marker_B, s=80, color=color_B, label='Entry B' if idx == 0 else '')

            ax_a.scatter(exit_date, priceA_exit, marker='s', s=80, edgecolors='red', facecolors='none', label='Exit A' if idx == 0 else '')
            ax_b.scatter(exit_date, priceB_exit, marker='s', s=80, edgecolors='red', facecolors='none', label='Exit B' if idx == 0 else '')

            ax_a.plot([entry_date, exit_date], [priceA_entry, priceA_exit], linestyle='--', color='blue', linewidth="0.7")
            ax_b.plot([entry_date, exit_date], [priceB_entry, priceB_exit], linestyle='--', color='blue', linewidth="0.7")

        ax_b.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()

        fig.suptitle(f"Entered {numTrades} Trades | LSTM", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



    def spreadGegen(self):
        dates = pd.to_datetime(self.Dates)
        end_date = dates.max()

        entries_lstm = list(self.tradeEntries)
        exits_lstm   = list(self.tradeExits)

        self.tradeEntries = []
        self.tradeExits   = []
        self.performSimpleBacktest()
        entries_def = list(self.tradeEntries)
        exits_def   = list(self.tradeExits)

        fig, (ax_spread, ax_Default) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax_spread.plot(dates, self.Spread, color='black', label='Spread')
        if entries_lstm:
            idx, _ = zip(*entries_lstm)
            ax_spread.scatter(
                dates[list(idx)],
                [self.Spread.iloc[i] for i in idx],
                marker='o', s=80,
                facecolors='none', edgecolors='springgreen',
                label='Entry'
            )
        if exits_lstm:
            idx, _ = zip(*exits_lstm)
            ax_spread.scatter(
                dates[list(idx)],
                [self.Spread.iloc[i] for i in idx],
                marker='s', s=80,
                facecolors='none', edgecolors='deeppink',
                label='Exit'
            )

        ax_spread.set_ylabel('Spread (z-Score)')
        ax_spread.set_title("LSTM Entry Logic | Pair: KB-RF")
        ax_spread.grid(True)
        ax_spread.set_xlim(pd.Timestamp('2023-01-19'), end_date)
        ax_spread.legend(loc='upper left', scatterpoints=1)

        ax_Default.plot(dates, self.Spread, color='black', label='Spread')
        if entries_def:
            idx, _ = zip(*entries_def)
            ax_Default.scatter(
                dates[list(idx)],
                [self.Spread.iloc[i] for i in idx],
                marker='o', s=80,
                facecolors='none', edgecolors='springgreen',
                label='Entry'
            )

        exits_def_sorted = sorted(exits_def, key=lambda tpl: tpl[0])
        exits_def_to_plot = exits_def_sorted[1:]
        first_exit_idx = min(i for i, _ in exits_def)
        exits_def_to_plot = [
            (i, val)
            for (i, val) in exits_def
            if i != first_exit_idx
        ]
        if exits_def_to_plot:
            idx, _ = zip(*exits_def_to_plot)
            ax_Default.scatter(
                dates[list(idx)],
                [self.Spread.iloc[i] for i in idx],
                marker='s', s=80,
                facecolors='none', edgecolors='deeppink',
                label='Exit'
            )

        ax_Default.set_ylabel('Spread (z-Score)')
        ax_Default.set_title("Default Entry Logic | Pair: KB-RF")
        ax_Default.grid(True)
        ax_Default.set_xlim(pd.Timestamp('2023-01-19'), end_date)




        x0, y0 = pd.Timestamp('2023-01-25'), -2.8
        dx, dy = pd.Timedelta('30 days'), 2

        for ax in (ax_spread, ax_Default):
            rect = patches.Rectangle(
                (x0, y0), dx, dy,
                linewidth=2, edgecolor='darkviolet', facecolor='none',
                transform=ax.transData
            )
            ax.add_patch(rect)
            txt_x = x0 + pd.Timedelta('32 days')
            txt_y = y0 + dy / 2
            ax.text(txt_x, txt_y,"1",fontsize=14, fontweight='bold',va='center', ha='left',color='darkviolet',transform=ax.transData)



        x0, y0 = pd.Timestamp('2024-04-17'), 1.5
        dx, dy = pd.Timedelta('30 days'), 2

        for ax in (ax_spread, ax_Default):
            rect = patches.Rectangle(
                (x0, y0), dx, dy,
                linewidth=2, edgecolor='darkviolet', facecolor='none',
                transform=ax.transData
            )
            ax.add_patch(rect)
            txt_x = x0 + pd.Timedelta('32 days')
            txt_y = y0 + dy / 2
            ax.text(txt_x, txt_y,"2",fontsize=14, fontweight='bold',va='center', ha='left',color='darkviolet',transform=ax.transData)


        handles, labels = ax_Default.get_legend_handles_labels()

        plt.show()

        filename = 'Backtest_spreadComp.pdf'
        fig.savefig(
            filename,
            dpi=700,
            bbox_inches='tight'
        )



    def plot_all_model_entries(self, lookback: int = 10, std_threshold: float = 1.5):

        horizons = [1, 2, 5, 10, 15]
        colors = {
            1: "cyan", 2: "dodgerblue", 5: "lightcoral",
            10: "mediumspringgreen", 15: "indigo", "Default": "black"
        }
        shapes = {
            1: "o", 2: "s", 5: "^",
            10: "v", 15: "D", "Default": "x"
        }
        entry_idx = {}

        entry_idx["Default"] = np.where(
            (self.Spread > std_threshold) | (self.Spread < -std_threshold)
        )[0]

        for h in horizons:
            model = SpreadLSTM(inputsize=1).to(device)
            ckpt = torch.load(
                f"MODEL_LSTM/LSTM_h{h}_lb{lookback}_hs{hidden_Size}_lr0.0002_ep70_SpreadOnly_{self.SpreadName}_BACKTEST.pth",
                map_location=device
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            idxs = []
            spread_vals = self.Spread.values
            for i in range(lookback, len(spread_vals)):
                window = spread_vals[i - lookback : i].reshape(lookback, 1)
                x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
                pred = model(x).cpu().item()
                if (spread_vals[i] > std_threshold and pred < window[-1, 0]) or (spread_vals[i] < -std_threshold and pred > window[-1, 0]):
                    idxs.append(i)
            entry_idx[h] = np.array(idxs)

        dates = pd.to_datetime(self.Dates)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, self.Spread.values, color="gray", label="Spread (z-score)")

        for key, idxs in entry_idx.items():
            plt.scatter(
                dates[idxs],
                self.Spread.values[idxs],
                c=colors[key],
                marker=shapes[key],
                edgecolors="w",
                s=80,
                label=str(key)
            )

        plt.axhline(std_threshold, color="red", linestyle="--", linewidth=1)
        plt.axhline(-std_threshold, color="red", linestyle="--", linewidth=1)
        plt.legend(title="Model Horizon / Default")
        plt.title("Entry Points per Model vs. Default Threshold")
        plt.xlabel("Time")
        plt.ylabel("Spread (z-score)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def __getEquitySeries(self) -> pd.Series:
        return pd.Series(self.GraphEquity, index=pd.to_datetime(self.Dates))

    def computeTotalReturn(self, eq: pd.Series) -> float:
        return eq.iloc[-1] / eq.iloc[0] - 1

    def computeCAGR(self, eq: pd.Series) -> float:
        total_ret = self.computeTotalReturn(eq)
        days = (eq.index[-1] - eq.index[0]).days
        years = days / 365.25
        return (1 + total_ret)**(1/years) - 1

    def computeVolatility(self, eq: pd.Series) -> float:
        daily_rets = eq.pct_change().dropna()
        return daily_rets.std() * np.sqrt(252)

    def computeMaxDrawdown(self, eq: pd.Series) -> float:
        running_max = eq.cummax()
        drawdowns = (eq - running_max) / running_max
        return drawdowns.min()

    def computeSharpeRatio(self, eq: pd.Series, rf: float = 0.0) -> float:
        cagr = self.computeCAGR(eq)
        vol = self.computeVolatility(eq)
        return (cagr - rf) / vol if vol != 0 else np.nan

    def computeCalmarRatio(self, eq: pd.Series) -> float:
        cagr = self.computeCAGR(eq)
        max_dd = abs(self.computeMaxDrawdown(eq))
        return cagr / max_dd if max_dd != 0 else np.nan

    def computeProfitFactor(self, eq: pd.Series) -> float:
        pnl = eq.diff().dropna()
        gross_win  = pnl[pnl > 0].sum()
        gross_loss = -pnl[pnl < 0].sum()
        return gross_win / gross_loss if gross_loss != 0 else np.nan

    def computeReturnToMaxDD(self, eq: pd.Series) -> float:
        total_ret = self.computeTotalReturn(eq)
        max_dd = abs(self.computeMaxDrawdown(eq))
        return total_ret / max_dd if max_dd != 0 else np.nan

    def printPerformanceMetrics(self, rf: float = 0.0):
        eq = self.__getEquitySeries()

        ren = (eq.iloc[-1] / self.InitEquity ) -1
        cagr   = self.computeCAGR(eq)
        vol    = self.computeVolatility(eq)
        mdd    = self.computeMaxDrawdown(eq)
        sharpe = self.computeSharpeRatio(eq, rf)
        calmar = self.computeCalmarRatio(eq)
        pf     = self.computeProfitFactor(eq)
        r2mdd  = self.computeReturnToMaxDD(eq)

        print("=== Performance Metrics ===")
        print(f"AbsReturn:                           {ren*100:6.2f}%")
        print(f"CAGR:                           {cagr*100:6.2f}%")
        print(f"Volatilität (ann.):            {vol*100:6.2f}%")
        print(f"Maximaler Drawdown:            {mdd*100:6.2f}%")
        print(f"Sharpe-Ratio (rf={rf*100:.2f}%):    {sharpe:6.2f}")
        print(f"Calmar-Ratio:                  {calmar:6.2f}")
        print(f"Profit Factor:                 {pf:6.2f}")
        print(f"Return / MaxDD:                {r2mdd:6.2f}")
        print("=============================")

        return (ren, cagr, vol, mdd, sharpe, r2mdd)


if __name__ == "__main__":
    # Spreads =  ["spreadz_KB_RF" ,"spreadz_KB_KEY","spreadz_PNC_SCHW" ,"spreadz_KEY_HBAN","spreadz_ALLY_DB","spreadz_CMA_TFC"]
    Spreads = ["spreadz_KB_RF"]
    for aSpread in Spreads:
        print(aSpread)
        parts = aSpread.split("_")
        TickerA = parts[1]
        TickerB = parts[2]

        Hors = [1, 2, 5, 10, 15]
        Hors = [1]
        for aH in Hors:

            ModelHorizon = aH
            DefaultThreshold = 1.5

            data = pd.read_csv("COMPLETE_DATASETLONG.csv")
            CloseA = data[f"close_{TickerA}"]
            CloseB = data[f"close_{TickerB}"]
            Spread = data[f"spreadz_{TickerA}_{TickerB}"]
            Dates = data["date"]
            BankIndex = pd.read_csv("COMPLETE_DATASETLONG.csv")["Close_Bank"]
            hedge_ratio = getHedgeRatioBETA(CloseA, CloseB)
            print(f"Hedge Ratio (Beta): {hedge_ratio:.4f}")
            print(f"1 Share of {TickerA} and {hedge_ratio:.4f} Shares of {TickerB}")

            TradeSignal = getTradeSignalSeriesDefault(Spread, DefaultThreshold)
            # print("Spread Long: " + str(sum(np.where(TradeSignal > 0, 1, 0))))
            # print("Spread Short : " + str(sum(np.where(TradeSignal < 0, 1, 0))))

            BacktestEngine = Backtest(CloseA, CloseB, Spread, TradeSignal, hedge_ratio, Dates, BankIndex, ModelHorizon, aSpread)
            
            # BacktestEngine.plot_all_model_entries(lookback, DefaultThreshold)
            
            BacktestEngine.performPredBacktest()
            # BacktestEngine.plotEquityGraph()
            # MetricsTuple = BacktestEngine.printPerformanceMetrics(rf=0.01)
            # BacktestEngine.showTradesPlot()
            BacktestEngine.spreadGegen()



            # # Save Results
            # metrics_cols = ['AbsReturn', 'cagr', 'vola', 'mdd', 'sharpe', 'r2mdd']
            # all_cols = ['Horizon'] + metrics_cols
            # df = pd.DataFrame([[aH] + list(MetricsTuple)], columns=all_cols)
            # filename = f"Auswertung_LSTM/AuswertungBacktest_{TickerA}_{TickerB}.csv"
            # df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

