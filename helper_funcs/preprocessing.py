from helper_funcs.data import cleaned_market
from os.path import exists
import pandas as pd
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values


def timeseries_init(time_col, static_cols,\
    value_cols, freq, fill_missing, type, ticker=None, group_col=None):

    if type=='MULTI':
        data = cleaned_market()
    elif type=='UNI':
        data = cleaned_market()
    elif type=='Sentiment':
        try:
            data = pd.read_pickle('Datasets/market/market_sentiment.pkl')
        except:
            timeseries = timeseries_init(
                time_col='Date',
                static_cols=[],
                value_cols=[
                    'Adj Close',
                    'Close',
                    'High',
                    'Low',
                    'Open',
                    'Volume'],
                freq='B',
                fill_missing=True,
                group_col='Ticker',
                type='MULTI'
                )
 
            dictionary = sentiment_init(timeseries[0])

            data = pd.DataFrame([])
            for i in range(6):
                df = timeseries[i].pd_dataframe().reset_index()
                df['Ticker'] = timeseries[i].static_covariates_values()[0][0]
                data = pd.concat([data, df], axis=0)
            data = data.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
            data.Date = data.Date.dt.date.apply(lambda x: str(x))

            data['sentiment'] = None
            for key in list(dictionary.keys()):
                data['sentiment'] = data['sentiment'].combine_first(
                    pd.merge(data, dictionary[key], how ='left', on=['Date', 'Ticker']).sentiment_score
                    )
            data.to_pickle('Datasets/market/market_sentiment.pkl')
    

    if type=='MULTI':
        timeseries = TimeSeries.from_group_dataframe(
            df=data,
            time_col=time_col,
            group_cols=group_col,  # individual time series are extracted by grouping `df` by `group_cols`
            static_cols=static_cols,
            value_cols=value_cols,
            freq=freq,
            fill_missing_dates=fill_missing
            )

        for i in range(len(timeseries)):
            timeseries[i] = fill_missing_values(timeseries[i], method='ffill', limit_direction='forward')
            timeseries[i] = timeseries[i].add_holidays(country_code='US')

        

    elif type=='Sentiment':
        timeseries = TimeSeries.from_group_dataframe(
            df=data,
            time_col=time_col,
            group_cols=group_col,  # individual time series are extracted by grouping `df` by `group_cols`
            static_cols=static_cols,
            value_cols=value_cols,
            freq=freq
            )

        for i in range(len(timeseries)):
            # timeseries[i] = fill_missing_values(timeseries[i], method='ffill', limit_direction='forward')
            timeseries[i] = timeseries[i].add_holidays(country_code='US')
    
    else:
        timeseries = None


    return timeseries


def get_covariates(type, data, target_col, past_cov, future_cov):

    if type=='MULTI':
        target_train, past_train, future_train,\
            target_val, past_val, future_val = [[] for _ in range(6)]


        for series in data['train']:
            target_train.append(series[target_col])
            past_train.append(series[past_cov])
            future_train.append(series[future_cov])

        for series in data['val']:
            target_val.append(series[target_col])
            past_val.append(series[past_cov])
            future_val.append(series[future_cov])
    else:
        return None


    return target_train, past_train, future_train,\
        target_val, past_val, future_val

def frac(neg, pos):
    if neg == 0:
        return 1
    if pos == 0:
        return 0
    return neg / pos

def sentiment_init(timeseries):
    df = pd.read_pickle("Datasets/results/preprocessing/sentiment_finetuned.pkl")
    tickers_df = pd.read_csv("Datasets/kaggle/Company_Tweet.csv")

    df = df.merge(tickers_df, on='tweet_id', how='inner')

    df = df[['post_date', 'sentiment_score', 'ticker_symbol']]

    df_positive = df[df.sentiment_score==1]
    df_positive['post_date'] = df_positive['post_date'].apply(lambda x: x.to_pydatetime().date())
    df_positive = df_positive.groupby(['ticker_symbol', 'post_date']).count()

    df_negative = df[df.sentiment_score==0]
    df_negative['post_date'] = df_negative['post_date'].apply(lambda x: x.to_pydatetime().date())
    df_negative = df_negative.groupby(['ticker_symbol', 'post_date']).count()

    business_days = pd.DataFrame(
        {'Date': timeseries.time_index.to_frame().reset_index(drop=True)['Date'].dt.date})

    sentiment_dict = {}
    tickers = ['AAPL', 'GOOG', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
    
    for ticker in tickers:
        negative = df_negative.xs(ticker).reindex(pd.date_range('2015-01-01', '2019-12-31'), fill_value=0)
        negative.index = negative.index.date
        positive = df_positive.xs(ticker).reindex(pd.date_range('2015-01-01', '2019-12-31'), fill_value=0)
        positive.index = positive.index.date

        lst_negative = []
        i = 0
        for date_b in business_days.Date:
            sum = 0
            for date in negative.index[i:]:
                i += 1
                sum += negative.loc[date].sentiment_score
                if date_b == date:
                    lst_negative.append(sum)
                    break

        lst_positive = []
        i = 0
        for date_b in business_days.Date:
            sum = 0
            for date in positive.index[i:]:
                i += 1
                sum += positive.loc[date].sentiment_score
                if date_b == date:
                    lst_positive.append(sum)
                    break
            
        new_df = pd.concat(
            [business_days.Date, pd.Series(lst_negative), pd.Series(lst_positive)],
            axis=1
            ).rename(columns = {0:'Negative', 1:'Positive'})

        sentiment_score = pd.DataFrame(
            {
                'sentiment_score': new_df.apply(lambda row: frac(row['Negative'], row['Positive']), axis=1),
                'Date': business_days.Date.apply(lambda x: str(x))
                }
                )
        sentiment_score['Ticker'] = ticker
        
        sentiment_dict[ticker] = sentiment_score
    
    return sentiment_dict