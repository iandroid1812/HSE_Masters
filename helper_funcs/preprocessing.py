from helper_funcs.data import load_market
from os.path import exists
import pandas as pd
import ta
import pickle
from darts import TimeSeries
import numpy as np


# fraction of negative tweets in all tweets
def frac_by_all(negative, positive):
    if negative == 0:
        return 0
    return negative / (negative + positive)

def frac(negative, positive):
    if negative == 0:
        return 0
    elif positive == 0:
        return 1
    return negative / positive


def market_init(freq='B', EXP_MA=15):
    data = load_market()
    tickers_df = pd.read_csv("Datasets/kaggle/Company_Tweet.csv")
    tickers = tickers_df.ticker_symbol.value_counts().index.to_list()
    tickers.sort()


    data['Volatility'] = ta.volatility.AverageTrueRange(
        high=data.High, low=data.Low, close=data.Close, window=EXP_MA).average_true_range()

    data = data[data.Date > '2014.12.31'].reset_index(drop=True)

    for ticker in tickers:
        df = data[data.Ticker == ticker]
        df = df.set_index('Date').resample(freq).ffill().reset_index()
    
        if ticker == 'AAPL':
            market_df = df
        else:
            market_df = pd.concat([market_df, df])
    
    market_df = market_df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)

    return market_df


def sentiment_init(freq='B', EXP_MA=15):
    tickers_df = pd.read_csv("Datasets/kaggle/Company_Tweet.csv")
    df_sentiment = pd.read_pickle("Datasets/results/preprocessing/sentiment_finetuned.pkl")
    tickers = tickers_df.ticker_symbol.value_counts().index.to_list()
    tickers.sort()

    df_sentiment = df_sentiment.merge(tickers_df, on='tweet_id', how='inner')
    df_sentiment = df_sentiment[['post_date', 'sentiment_score', 'ticker_symbol']]
    df_sentiment.loc[:, 'post_date'] = df_sentiment.post_date.apply(lambda x: x.to_pydatetime().date())
    df_sentiment = df_sentiment.groupby(['ticker_symbol', 'post_date']).value_counts().unstack(fill_value=0)

    for ticker in tickers:
        # first we add missing dates and fill thew with 0
        company = df_sentiment.xs(ticker).reindex(pd.date_range('2015-01-02', '2019-12-31'), fill_value=0)

        # then resample to business days and add up dropped past values up to the date that is not dropped
        company = company.resample(rule=freq, origin='end').sum().rename(columns = {0:'Negative', 1:'Positive'})

        # taking fractions of negative tweeets to calculate the final score 
        company['sentiment_score_1'] = company.apply(lambda row: frac_by_all(row['Negative'], row['Positive']), axis=1)
        company['sentiment_score_2'] = company.apply(lambda row: frac(row['Negative'], row['Positive']), axis=1)

        # calculating moving average
        company['sentiment_score_1'] = company.sentiment_score_1.ewm(span=EXP_MA).mean()
        company['sentiment_score_2'] = company.sentiment_score_2.ewm(span=EXP_MA).mean()

        company['Ticker'] = ticker
        company = company.reset_index().rename(
            columns={
                'index': 'Date'
                })
        
        if ticker == 'AAPL':
            sentiment_df = company
        else:
            sentiment_df = pd.concat([sentiment_df, company])
    
    sentiment_df = sentiment_df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    sentiment_df.columns.name = None

    return sentiment_df


def embeddings_init(freq='B'):
    with open('Datasets/results/preprocessing/embeddings_2.pkl', "rb") as file:
        embeddings = pickle.load(file)
        file.close()
    tickers_df = pd.read_csv("Datasets/kaggle/Company_Tweet.csv")
    tickers = tickers_df.ticker_symbol.value_counts().index.to_list()
    tickers.sort()

    df = pd.read_pickle("Datasets/results/preprocessing/final.pkl")
    df = df[['tweet_id', 'post_date']]
    df = df.merge(tickers_df, on='tweet_id', how='inner').rename(
        columns={
            'post_date': 'Date',
            'ticker_symbol': 'Ticker'
        })

    df['Date'] = df['Date'].apply(lambda x: x.date())
    df = df.reset_index()
    df = df.groupby(['Ticker', 'Date']).agg({'index': lambda x: embeddings[list(x)].mean(axis=0).tolist()})
    df = df.rename(columns={
        'index': 'embeddings'
    })

    df.embeddings = df.embeddings.apply(np.array)

    for ticker in tickers:
        # first we add missing dates and fill thew with 0
        company = df.xs(ticker).reindex(pd.date_range('2015-01-02', '2019-12-31'), fill_value=np.zeros(384))

        # then resample to business days and add up dropped past values up to the date that is not dropped
        company = company.resample(rule=freq, origin='end').mean()

        company['Ticker'] = ticker
        
        if ticker == 'AAPL':
            embeddings_df = company
        else:
            embeddings_df = pd.concat([embeddings_df, company])
    
    embeddings_df = embeddings_df.reset_index().rename(
            columns={
                'index': 'Date'
                })
    embeddings_df = embeddings_df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    
    return embeddings_df


def total_init():
    if exists('Datasets/results/total_df.pkl'):
        return pd.read_pickle('Datasets/results/total_df.pkl')

    market = market_init()
    sentiment = sentiment_init()
    embeddings = embeddings_init()

    total = pd.merge(market, sentiment, on=['Date', 'Ticker'])
    total = pd.merge(total, embeddings, on=['Date', 'Ticker'])
    total= total.join(pd.DataFrame(total['embeddings'].to_list()))

    total.to_pickle('Datasets/results/total_df.pkl')

    return total


def total_timeseries(market=True, sentiment=True, embeddings=True):

    market_columns = []
    if market is True:
        market_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Volatility']

    sentiment_columns = []
    if sentiment is True:
        sentiment_columns = ['Negative', 'Positive', 'sentiment_score_1', 'sentiment_score_2']
    
    embeddings_columns = []
    if embeddings is True:
        embeddings_columns = np.arange(384).tolist()

    value_columns = market_columns + sentiment_columns + embeddings_columns
    
    data = total_init()

    timeseries_total = TimeSeries.from_group_dataframe(
        df=data,
        time_col='Date',
        group_cols='Ticker',
        static_cols=[],
        value_cols=value_columns,
        freq='B'
    )

    for i in range(len(timeseries_total)):
        timeseries_total[i] = timeseries_total[i].add_holidays(country_code='US')

    return timeseries_total


def get_covariates(data, past_covariates, embeddings=False):
    train = data[0]
    val = data[1]
    emb = np.arange(384).tolist()
    emb = list(map(str, emb))

    if embeddings:
       past_covariates = past_covariates + emb

    target_train, past_train, future_train,\
        target_val, past_val, future_val, target_test, past_test, future_test = [[] for _ in range(9)]

    for series_1, series_2 in zip(train, val):
        target_train.append(series_1['Close'])
        past_train.append(series_1[past_covariates])
        future_train.append(series_1['holidays'])

        target_val_, target_test_ = series_2['Close'].split_before(258)
        past_val_, past_test_ = series_2[past_covariates].split_before(258)
        future_val_, future_test_ = series_2['holidays'].split_before(258)

        target_val.append(target_val_)
        past_val.append(past_val_)
        future_val.append(future_val_)

        target_test.append(target_test_)
        past_test.append(past_test_)
        future_test.append(future_test_)


    return target_train, past_train, future_train,\
        target_val, past_val, future_val, target_test, past_test, future_test