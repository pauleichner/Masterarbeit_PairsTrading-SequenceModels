import csv
import requests
from urllib.request import urlopen
import ssl
import certifi
import json
import time

def formURL(Param, Ticker, Key, from_date=None, to_date=None):
    url = f"https://financialmodelingprep.com/api/v3/{Param}/{Ticker}?apikey={Key}"
    if from_date and to_date:
        url += f"&from={from_date}&to={to_date}"
    return url

def formNewsURL(api_key, ticker, page, from_date, to_date):
    return (f"https://financialmodelingprep.com/api/v3/stock_news?"
            f"tickers={ticker}&page={page}&from={from_date}&to={to_date}&apikey={api_key}")

class FMPDataGetter():
    apiKEY = "" # fill in own API-Key
    startDate = "2005-01-01"
    endDate = "2025-01-01"

    def __init__(self, Parameter, Ticker):
        self.Parameter = Parameter
        self.Ticker = Ticker

    def getPriceData(self):
        resp = requests.get(formURL(self.Parameter, self.Ticker, self.apiKEY, self.startDate, self.endDate))

        if resp.status_code == 200:
            data = resp.json()
            print(data)
            if self.Parameter == 'historical-price-full':
                if "historical" in data:
                    historical_data = data["historical"]
                    filename = f"DataKrypto/{self.Ticker}_{self.Parameter}.csv"
                    
                    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["date", "open", "high", "low", "close", "volume"])
                        for day in historical_data:
                            writer.writerow([
                                day.get("date"),
                                day.get("open"),
                                day.get("high"),
                                day.get("low"),
                                day.get("close"),
                                day.get("volume")
                            ])
                    print(f"CSV-Datei '{filename}' wurde erfolgreich erstellt.")
                else:
                    print("Keine historischen Daten gefunden.")
        else:
            print("Fehler bei Datenbeschaffung")
            print(resp)

    def getNewsData(self):
        news_items = []
        page = 0
        context = ssl.create_default_context(cafile=certifi.where())
        
        while True:
            url = formNewsURL(self.apiKEY, self.Ticker, page, self.startDate, self.endDate)
            try:
                response = urlopen(url, context=context)
            except Exception as e:
                print("Fehler beim Abrufen der URL:", e)
                break
            
            data_str = response.read().decode("utf-8")
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                print(f"Fehler beim Dekodieren der JSON-Daten auf Seite {page}.")
                break
            if not data:
                print(f"Keine weiteren Nachrichten auf Seite {page}.")
                break
            
            print(f"Seite {page} hat {len(data)} Nachrichten.")
            news_items.extend(data)
            page += 1
            time.sleep(0.1) # for Rate Limiting
        
        filename = f"DataKrypto/{self.Ticker}_news.json"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(news_items, f, ensure_ascii=False, indent=4)
            print(f"Gespeicherte Nachrichten in '{filename}'.")
        except Exception as e:
            print("Fehler beim Schreiben der JSON-Datei:", e)

def main():
    symbols = ['MS', 'DB', 'ALLY', 'BAC', 'BMO', 'C', 'CMA', 'COF', 'FITB', 'GS', 'HBAN', 'JPM', 'KB', 'KEY', 'MTB', 'PNC', 'RF', 'SAN', 'SCHW', 'TD', 'TFC', 'USB', 'WFC']
    for Sym in symbols:
        DataGetter = FMPDataGetter('historical-price-full', Sym)
        DataGetter.getPriceData()
        DataGetter.getNewsData()

if __name__ == "__main__":
    main()
