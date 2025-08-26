import torch
import pandas as pd
import numpy as np
import re
import math
from datetime import datetime, date

def createDataset(dataset, lookback, horizon):
    X, y = [], []
    for i in range(len(dataset) - lookback - horizon + 1):
        feature = dataset[i : i + lookback]
        target  = dataset[i + lookback : i + lookback + horizon]
        X.append(feature)
        y.append(target)
    
    # print(X)
    # print(y)
    return torch.tensor(X), torch.tensor(y)

def getChange(data):
    X = []
    # erster Tag: kein Vorgänger → 0 oder np.nan
    X.append(0.0)
    for i in range(1, len(data)):
        ch = (data[i] - data[i-1]) / data[i-1]
        X.append(ch)
    return X




from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, logging
logging.set_verbosity_error()

MODELName = "yiyanghkust/finbert-tone"  
Tokenizer = AutoTokenizer.from_pretrained(MODELName)
Model = AutoModelForSequenceClassification.from_pretrained(MODELName)
FinBERT = pipeline(task="sentiment-analysis", model=Model, tokenizer=Tokenizer, framework="pt")

def createNewsSentiment(newsline):
    if not isinstance(newsline, str):
        # print("Need String!")
        return 0

    # print(FinBERT(newsline))
    RESULT = FinBERT(newsline)[0]["label"]
    # print(RESULT)
    if RESULT in ("Neutral", "´None"):
        return 0.0
    elif RESULT == "Negative":
        return -1.0
    elif RESULT == "Positive":
        return +1.0
    else:
        return 0.0



class ExpDecaySentiment:
    def __init__(self, half_life_days: float = 5.0):
        self._lambda = math.log(2) / half_life_days
        self._last_value = 0.0
        self._last_date: date | None = None

    def update(self, newsline: str, today: date | None = None) -> float:
        if today is None:
            today = datetime.now().date()
        if self._last_date is not None:
            delta_days = (today - self._last_date).days
            decay = math.exp(-self._lambda * delta_days)
            self._last_value *= decay

        s = createNewsSentiment(newsline)
        if s != 0.0:
            self._last_value = s
            self._last_date = today

        return self._last_value

    def current(self, as_of: date | None = None) -> float:

        if self._last_date is None:
            return 0.0
        if as_of is None:
            as_of = datetime.now().date()
        delta_days = (as_of - self._last_date).days
        return self._last_value * math.exp(-self._lambda * delta_days)





def createNEWSDataset(DataSet):
    temp = []
    for aNews in DataSet:
        if aNews is not None:
            tempSentiment =  createNewsSentiment(aNews)
            temp.append(tempSentiment) 

    return temp



# sample_news = ["Tesla beats quarterly Earnings by a landslide ; Truist Delivers Rare Sector Miss"]
# sentiments = createNEWSDataset(sample_news)
# print(sentiments)

def createDecayedNewsSeries(dates, headlines, half_life_days=5.0):
    tracker = ExpDecaySentiment(half_life_days=half_life_days)
    decayed = []
    for d, h in zip(dates, headlines):
        today = d.date() if hasattr(d, "date") else d
        decayed.append(tracker.update(h, today=today))
    # zurückgeben als np.array mit dtype float32
    return np.array(decayed, dtype=np.float32)



def compute_ATR(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    # True Range: max(High-Low, abs(High - PrevClose), abs(Low - PrevClose))
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def compute_MACD(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram})


def getFeaturesSequence(
    SpreadName: str,
    SP500: bool,
    TENY: bool,
    CRESPREAD: bool,
    BANK: bool,
    IV10: bool,
    IV20: bool,
    NEWS: bool,
    CLOSE: bool,
    ATR: bool = False,
    MACD: bool = False
):
    if not isinstance(SpreadName, str):
        print("Please provide a valid SpreadName")
        return

    flags = [
       ("SP500",     SP500),
       ("TENY",      TENY),
       ("CRESPREAD", CRESPREAD),
       ("BANK",      BANK),
       ("IV10",      IV10),
       ("IV20",      IV20),
       ("NEWS",      NEWS),
       ("CLOSE",     CLOSE),
       ("ATR",       ATR),
       ("MACD",      MACD),
    ]
    active = [name for name, on in flags if on]
    feature_str = "_".join(active) if active else "SpreadOnly"

    # Ticker extrahieren
    m1 = re.search(r"spreadz_([^_]+)", SpreadName)
    TICKER1 = m1.group(1)
    temp = f"{TICKER1}_"
    m2 = re.search(rf"{temp}([^_]+)", SpreadName)
    TICKER2 = m2.group(1)

    # Daten einlesen
    DF = pd.read_csv("COMPLETE_DATASET.csv")
    DF["date"] = pd.to_datetime(DF["date"])
    required = [SpreadName, "close", "DGS10", "HighYield", "InvGradeYield", "Close_Bank",
                f"close_{TICKER1}", f"close_{TICKER2}"]
    if ATR or MACD:
        required += [f"high_{TICKER1}", f"low_{TICKER1}", f"high_{TICKER2}", f"low_{TICKER2}"]
    DF = DF.dropna(subset=required).reset_index(drop=True)

    # Basis-Arrays
    Spread      = DF[SpreadName].astype("float32").values
    SandP       = DF["close"].astype("float32").values
    TenyTreas   = DF["DGS10"].astype("float32").values
    HighY       = DF["HighYield"].astype("float32").values
    InGra       = DF["InvGradeYield"].astype("float32").values
    CreditSpread = HighY - InGra
    Bank        = DF["Close_Bank"].astype("float32").values

    ImpVolTicker_1_10 = DF[f"ivmean10_{TICKER1}_IV"].astype("float32").values
    ImpVolTicker_1_20 = DF[f"ivmean20_{TICKER1}_IV"].astype("float32").values
    ImpVolTicker_2_10 = DF[f"ivmean10_{TICKER2}_IV"].astype("float32").values
    ImpVolTicker_2_20 = DF[f"ivmean20_{TICKER2}_IV"].astype("float32").values

    CloseTicker1 = DF[f"close_{TICKER1}"]
    CloseTicker2 = DF[f"close_{TICKER2}"]

    # Basis-Features
    SPChange     = getChange(SandP)
    TenYT        = getChange(TenyTreas)
    CredSpread   = getChange(CreditSpread)
    BankIndex    = getChange(Bank)
    Im1_10       = getChange(ImpVolTicker_1_10)
    Im1_20       = getChange(ImpVolTicker_1_20)
    Im2_10       = getChange(ImpVolTicker_2_10)
    Im2_20       = getChange(ImpVolTicker_2_20)
    CloseVals1   = getChange(CloseTicker1)
    CloseVals2   = getChange(CloseTicker2)

    # News
    NewsLookAheadBias = False
    NewsTicker1  = DF[f"title_{TICKER1}_news"]
    NewsTicker2  = DF[f"title_{TICKER2}_news"]
    NewsList1    = createNEWSDataset(NewsTicker1)
    NewsList2    = createNEWSDataset(NewsTicker2)
    NewsList1 = np.asarray(NewsList1, dtype=np.float32)
    NewsList2 = np.asarray(NewsList2, dtype=np.float32)
    # NewsList1 = createDecayedNewsSeries(DF["date"], DF[f"title_{TICKER1}_news"], half_life_days=5.0)
    # NewsList2 = createDecayedNewsSeries(DF["date"], DF[f"title_{TICKER2}_news"], half_life_days=5.0)


    if NewsLookAheadBias:
        # 1-Tag-Look-Ahead
        news1_shifted = np.roll(NewsList1, -1)
        news2_shifted = np.roll(NewsList2, -1)
        # letzes Element auf 0 setzen
        news1_shifted[-1] = 0.0
        news2_shifted[-1] = 0.0

        NewsList1, NewsList2 = news1_shifted, news2_shifted

    # ATR und MACD
    if ATR:
        # High/Low/Close als Series
        atr1 = compute_ATR(DF[f"high_{TICKER1}"], DF[f"low_{TICKER1}"], CloseTicker1)
        atr2 = compute_ATR(DF[f"high_{TICKER2}"], DF[f"low_{TICKER2}"], CloseTicker2)
    if MACD:
        macd1 = compute_MACD(CloseTicker1)
        macd2 = compute_MACD(CloseTicker2)

    # Feature-Liste zusammenstellen
    features = [Spread]
    if SP500:      features.append(SPChange)
    if TENY:       features.append(TenYT)
    if CRESPREAD:  features.append(CredSpread)
    if BANK:       features.append(BankIndex)
    if IV10:
        features.extend([Im1_10, Im2_10])
    if IV20:
        features.extend([Im1_20, Im2_20])
    if NEWS:
        features.extend([NewsList1, NewsList2])
    if CLOSE:
        features.extend([CloseVals1, CloseVals2])
    if ATR:
        features.extend([atr1.values.astype("float32"), atr2.values.astype("float32")])
    if MACD:
        features.extend([
            macd1['macd_line'].values.astype("float32"),
            macd1['signal_line'].values.astype("float32"),
            macd1['histogram'].values.astype("float32"),
            macd2['macd_line'].values.astype("float32"),
            macd2['signal_line'].values.astype("float32"),
            macd2['histogram'].values.astype("float32")
        ])

    lengths = [len(arr) for arr in features]
    if len(set(lengths)) != 1:
        raise ValueError(f"Unterschiedliche Längen in Features: {lengths}")

    return (feature_str, *tuple(features))