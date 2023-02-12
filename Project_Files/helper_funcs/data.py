import pandas as pd
# import yfinance as yf
from os.path import exists


def load_market():
    """This function uses yfinanace library to download market data for tickers
    listed inside Company.csv for time period from beggining of 2015 to the end of 2019.
    Doesn't require any arguments because data does not change depending on a task

    Returns
    -------
    pandas DataFrame 
        contains market features High, Low, Open, Close, Volume ...
    """
    prefix = "../../"

    if exists(prefix + 'Datasets/market/market.csv'):
        return pd.read_csv(prefix + 'Datasets/market/market.csv')

    ticker_to_name_path = prefix + "Datasets/kaggle/Company.csv"

    company_name_df = pd.read_csv(ticker_to_name_path)
    tickers = company_name_df.ticker_symbol.tolist()

    # loads data 1 month before the required date to have moving average 
    # values already calculated from the beggining of dataset instead of 0 values
    market_data = yf.download((' ').join(tickers), start="2014-12-01", end="2020-01-01", group_by = 'ticker')

    market_data = market_data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)

    market_data = market_data.dropna()
    market_data = market_data.reset_index()

    market_data.loc[:, 'Date'] = pd.to_datetime(market_data['Date'])

    market_data = market_data.reset_index(drop=True)

    # save dataframe to csv for later usage
    market_data.to_csv(prefix + 'Datasets/market/market.csv', index=False)

    return market_data

def cleaned_market():
    """Returns data for the required period synchronized with twitter dataset

    Returns
    -------
    pandas DataFrame 
        contains market features High, Low, Open, Close, Volume ...
    """

    market_data = load_market()

    market_data = market_data[market_data.Date > '2014.12.31']

    return market_data.reset_index(drop=True)
