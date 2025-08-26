import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.style.use('classic')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.titlesize": 21,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.6,
})

DF = pd.read_csv("COMPLETE_DATASET.csv", parse_dates=["date"])
DF = DF.dropna(subset=["spreadz_PNC_SCHW"]).reset_index(drop=True)
spread = DF["spreadz_PNC_SCHW"].astype("float32")

arma_mod = ARIMA(spread, order=(1, 0, 0), trend='c')
res = arma_mod.fit()
resid = res.resid
print("Koeffizienten (params):")
print(res.params)

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

plot_acf(spread, lags=40, ax=axes1[0])
axes1[0].set_title("ACF original Spread")
axes1[0].grid(True)
axes1[0].set_ylim(-1.1, 1.1)
plot_pacf(spread, lags=40, ax=axes1[1])
axes1[1].set_title("PACF original Spread")
axes1[1].grid(True)
axes1[1].set_ylim(-1.1, 1.1)

plt.tight_layout()
filename = f'APACF_orig.pdf'
fig1.savefig(
    filename,
    dpi=700,
    bbox_inches='tight'
)


fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

plot_acf(resid, lags=40, ax=axes2[0])
axes2[0].set_title("ACF of Residuals (ARMA(1,0))")
axes2[0].grid(True)
axes2[0].set_ylim(-1.1, 1.1)
plot_pacf(resid, lags=40, ax=axes2[1])
axes2[1].set_title("PACF of Residuals (ARMA(1,0))")
axes2[1].grid(True)
axes2[1].set_ylim(-1.1, 1.1)
plt.tight_layout()
filename = f'APACF_residuals.pdf'
fig2.savefig(
    filename,
    dpi=700,
    bbox_inches='tight'
)

