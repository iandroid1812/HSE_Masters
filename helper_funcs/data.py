import pandas as pd
import yfinance as yf
from os.path import exists


def load_market():
    if exists('Datasets/market/market.csv'):
        return pd.read_csv('Datasets/market/market.csv')

    # id_to_ticker_path = "Datasets/kaggle/Company_Tweet.csv"
    ticker_to_name_path = "Datasets/kaggle/Company.csv"

    # tickers_df = pd.read_csv(id_to_ticker_path)
    company_name_df = pd.read_csv(ticker_to_name_path)
    tickers = company_name_df.ticker_symbol.tolist()

    market_data = yf.download((' ').join(tickers), start="2014-12-31", end="2020-01-01", group_by = 'ticker')

    market_data = market_data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)

    market_data = market_data.dropna()
    market_data = market_data.reset_index()

    market_data.to_csv('Datasets/market/market.csv', index=False)

    return market_data

def cleaned_market():
    market_data= load_market()

    market_data = market_data[market_data.Date > '2014.12.31']

    return market_data.reset_index(drop=True)