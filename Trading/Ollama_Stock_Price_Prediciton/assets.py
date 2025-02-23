#############
##LIBRARIES##
#############

import re
import math

import yfinance as yf
from finvizfinance.quote import finvizfinance

from statsmodels.tsa.statespace.sarimax import SARIMAX

import plotly.graph_objects as go
import pandas as pd
import numpy as np

import holidays

from langchain_community.llms import Ollama


import streamlit as st

llm = Ollama(model='llama3')

def classify_sentiment(title):
    output = llm.invoke(f"Classify the sentiment as 'POSITIVE' or 'NEGATIVE' or 'NEUTRAL' with just that one")
    return output.strip()



def classify_sentiment_batch(titles, batch_size = 10):
    print(f"Clasificando {len(titles)} títulos de noticias.")


    valid_sentiments = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
    sentiments = ["NEUTRAL"] * len(titles)

    num_batches = math.ceil(len(titles)/batch_size)

    for i in range(num_batches):
        print(f"Batch {i}")
        batch_titles = titles[i * batch_size:(i + 1) * batch_size]

        prompt = (
            "For each news title below, classify the sentiment as 'POSITIVE', 'NEGATIVE' or 'NEUTRAL'.\n"
            "Return exactly one sentiment per title, and a number with the order of the titles in the same order as the titles, and NOTHING ELSE.\n"
            "The answer can only contain a number with the order of the title and the words POSITIVE, NEGATIVE or NEUTRAL.\n"
            "Example:\n"
            "1 - POSITIVE\n"
            "2 - NEGATIVE\n"
            "3 - NEUTRAL\n"
        )

        prompt += "\n".join(f"{idx+1} - {title}" for idx, title in enumerate(batch_titles))

        output = llm.invoke(prompt)
        print(f"🔹 Respuesta de Ollama para el batch {i + 1}/{num_batches}:\n{output}\n")

        for line in output.split("\n"):
            line = line.strip().upper()
            match = re.match(r"(\d+)\s*-\s*(POSITIVE|NEGATIVE|NEUTRAL)", line)
            
            if match:
                index = int(match.group(1)) - 1 + (i * batch_size)  # Convertir a índice global
                sentiment = match.group(2)
                
                if 0 <= index < len(sentiments):  # Verificar que el índice sea válido
                    sentiments[index] = sentiment


        
    while len(sentiments) < len(titles):
        sentiments.append("NEUTRAL")
        
    if len(sentiments) > len(titles):
        sentiments = sentiments[:len(titles)]

    print(f"La longitud de los sentiments es {len(sentiments)}")
    print(f"Classification completed!:\n{sentiments}")
    return sentiments


def get_news_ticker(ticker):
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()
    return news_df

# Function to get and process news data
def get_news_data(news_df_original):
    
    news_df =news_df_original.copy()
    news_df['Title'] = news_df['Title'].str.lower()


    news_df['sentiment'] = classify_sentiment_batch(news_df['Title'].tolist())

    news_df_sent = news_df.copy()
    news_df_sent = news_df_sent[news_df_sent['sentiment'] != 'NEUTRAL'].copy()

    news_df_sent['Date'] = pd.to_datetime(news_df_sent['Date'])
    news_df_sent['DateOnly'] = news_df_sent['Date'].dt.date


    return news_df_sent


# Function to group and process sentiment data
def process_sentiment_data(news_df):
    """
    Agrupa las noticias por día de cotización y calcula el sentimiento promedio en los últimos 7 días hábiles.
    """
    print(f"Procesamos los datos de news_df con columnas: {news_df.columns}")

    grouped = news_df.groupby(['Trading_Day', 'sentiment']).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=['POSITIVE', 'NEGATIVE'], fill_value=0)

    print("Grouped inicial")
    print(grouped)

   
    all_trading_days = pd.date_range(start=grouped.index.min(), end=grouped.index.max(), freq='B')
    grouped = grouped.reindex(all_trading_days, fill_value=0)

    grouped['7day_avg_positive'] = grouped['POSITIVE'].rolling('7D', min_periods=1).sum()
    grouped['7day_avg_negative'] = grouped['NEGATIVE'].rolling('7D', min_periods=1).sum()

    grouped['7day_pct_positive'] = grouped['POSITIVE'].expanding().sum() / (grouped['POSITIVE'].expanding().sum() + grouped['NEGATIVE'].expanding().sum())

    result_df = grouped.reset_index().rename(columns={'index': 'Trading_Day'})

    print("Final result_df")
    print(result_df)

    return result_df

# Function to fetch and process stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)  
    stock_data['Pct_Change'] = stock_data['Close'].pct_change() * 100 
    return stock_data


#We fill the weekends too
def fill_missing_stock_dates(stock_data):
    all_dates = pd.date_range(start=stock_data.index.min(), end=stock_data.index.max(), freq="D")
    
    stock_data = stock_data.reindex(all_dates)
    return stock_data



def next_trading_day(stock_dates, news_date):
    news_date = np.datetime64(news_date) 

    pos = np.searchsorted(stock_dates, news_date)
    if pos == len(stock_dates): 
        return stock_dates[-1]
    
    return stock_dates[pos]

def trading_day(stock_data, result_df):
    stock_dates = np.array(stock_data.index)

    result_df['Trading_Day'] = result_df['DateOnly'].apply(lambda date: next_trading_day(stock_dates, date))
    return result_df


def preprocess_stock_data(stock_data):
    stock_data.columns = stock_data.columns.droplevel(1)

    stock_data.columns.name = None

    stock_data.index.name = None

    return stock_data




# Function to combine sentiment and stock data
def combine_data(result_df, stock_data):
    combined_df = result_df.set_index('Trading_Day').join(stock_data[['Pct_Change']], how='inner')
    combined_df['lagged_7day_pct_positive'] = combined_df['7day_pct_positive'].shift(1)  # Lag sentiment feature

    return combined_df

# Function to calculate Pearson correlation
def calculate_correlation(combined_df):
    correlation_pct_change = combined_df[['lagged_7day_pct_positive', 'Pct_Change']].corr().iloc[0, 1]
    return correlation_pct_change

# Function to get future dates excluding weekends and holidays
def get_future_dates(start_date, num_days):
    print("get future dates")
    us_holidays = holidays.US()
    future_dates = []
    current_date = start_date

    while len(future_dates) < num_days:
        if current_date.weekday() < 5 and current_date not in us_holidays:
            future_dates.append(current_date)
        current_date += pd.Timedelta(days=1)

    return future_dates


# Function to get future dates excluding weekends or holidays from the next day
def get_future_dates_next_day(combined_df, num_days):

    print("get future dates")
    us_holidays = holidays.US()
    future_dates = []


    last_real_date = combined_df.dropna(subset=['Pct_Change']).index[-1]
    current_date = last_real_date + pd.Timedelta(days=1)

    while len(future_dates) < num_days:
        if current_date.weekday() < 5 and current_date not in us_holidays:
            if current_date not in combined_df.index:
                future_dates.append(current_date)
        current_date += pd.Timedelta(days=1)

    return future_dates



#Function that takes the dataframe with the data from the news and predicts the percentage change for the next days
def fit_and_forecast(combined_df, function_future_dates=get_future_dates ,forecast_steps=3):
    endog = combined_df['Pct_Change'].dropna() 
    exog = combined_df['lagged_7day_pct_positive'].dropna() 
    print("GOING FOR THE ARIMAX MODEL")
    endog = endog.tail(200)
    exog = exog.loc[endog.index]  

    model = SARIMAX(endog, exog=exog, order=(1, 1, 1))
    fit = model.fit(disp=False, maxiter=50) 

    if function_future_dates == get_future_dates_next_day:
        future_dates = function_future_dates(combined_df, forecast_steps)
    else:
        print(combined_df)
        future_dates = function_future_dates(combined_df.index[-1], forecast_steps)
    
    future_exog = []
    for date in future_dates:
        if date in combined_df.index:
            future_exog.append(combined_df.loc[date, 'lagged_7day_pct_positive'])
        else:
            future_exog.append(combined_df['lagged_7day_pct_positive'].iloc[-1])
    
    future_exog = np.array(future_exog).reshape(-1, 1)

    forecast = fit.get_forecast(steps=forecast_steps, exog=future_exog)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    return forecast_mean, forecast_ci, future_dates



def preprocessing_data(combined_df):

    last_real_date = combined_df.dropna(subset=['Pct_Change']).index[-1]
    last_real_value = combined_df['Pct_Change'].dropna().iloc[-1]

    forecast_mean_from_start = pd.Series(
    [last_real_value] + forecast_mean.tolist(),
    index=[last_real_date] + forecast_index
    )

    forecast_index_from_start = forecast_mean_from_start.index

    forecast_mean_from_start = pd.concat([
        pd.Series([last_real_value], index=[last_real_date]),
        forecast_mean
    ])

    forecast_ci_first_point = pd.DataFrame({
        'lower Pct_Change': [last_real_value], 
        'upper Pct_Change': [last_real_value]
    }, index=[last_real_date])

    forecast_ci_from_start = pd.concat([forecast_ci_first_point, forecast_ci])

    pct_change_mean = combined_df['Pct_Change'].mean()
    pct_change_std = combined_df['Pct_Change'].std()

    forecast_mean_from_start_std = (forecast_mean_from_start - pct_change_mean) / pct_change_std

    forecast_ci_from_start_std = pd.DataFrame({
        'lower Pct_Change': (forecast_ci_from_start['lower Pct_Change'] - pct_change_mean) / pct_change_std,
        'upper Pct_Change': (forecast_ci_from_start['upper Pct_Change'] - pct_change_mean) / pct_change_std
    }, index=forecast_ci_from_start.index)

    return combined_df, forecast_mean_from_start_std, forecast_ci_from_start_std, forecast_index_from_start



# Function to create and display plot
def create_plot(combined_df, forecast_mean, forecast_ci, forecast_index):


    sentiment_std = (combined_df['7day_pct_positive'] - combined_df['7day_pct_positive'].mean()) / combined_df['7day_pct_positive'].std()

    pct_change_std = (combined_df['Pct_Change'] - combined_df['Pct_Change'].mean()) / combined_df['Pct_Change'].std()


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=sentiment_std,
        name='Standardized Sentiment Proportion',
        line=dict(color='blue'),
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=pct_change_std,
        name='Stock Pct Change (Standardized)',
        line=dict(color='orange'),
        mode='lines+markers'
    ))


    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_mean,
        name='Forecasted Stock Pct Change',
        line=dict(color='red'),
        mode='lines+markers'
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast_index, forecast_index[::-1]]),
        y=np.concatenate([forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1][::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        title='Sentiment Proportion and Stock Percentage Change with Forecast',
        xaxis = dict(
            title='Date',
            tickmode='linear',
            dtick=86400000.0,
            tickangle=90,
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text='Standardized Sentiment Proportion', font=dict(color='orange'))
        ),
        yaxis2=dict(
            title=dict(text='Stock Pct Change', font=dict(color='orange')),
            overlaying='y',
            side='right'
        ),
        template='plotly_dark'
    )



    st.plotly_chart(fig)


# PART 3
# STREAMLIT

st.sidebar.title("News Sentiment for Stock Prices Prediction")
ticker = st.sidebar.text_input("Enter stock ticker. For instance choose SBUX:", value='SBUX')

if st.sidebar.button("Get News Sentiment Data"):
    news_df_original = get_news_ticker(ticker)
    news_df = get_news_data(news_df_original)

    st.session_state["news_df"] = news_df

    st.write("News Sentiment Data obtained successfully!")
    st.write(news_df.head())

if st.sidebar.button("Run Full Analysis"):
    if "news_df" in st.session_state:
        news_df = st.session_state["news_df"]

        start_date = news_df['DateOnly'].min().strftime('%Y-%m-%d')
        end_date = news_df['DateOnly'].max().strftime('%Y-%m-%d')
        stock_data = get_stock_data(ticker, start_date, end_date)

        stock_data = fill_missing_stock_dates(stock_data)

        news_df = trading_day(stock_data, news_df)

        result_df = process_sentiment_data(news_df)

        stock_data = preprocess_stock_data(stock_data)

        combined_df = combine_data(result_df, stock_data)
        
        correlation_pct_change = calculate_correlation(combined_df)

        st.write(f"Correlation: {correlation_pct_change}")

        forecast_mean, forecast_ci, forecast_index = fit_and_forecast(combined_df, get_future_dates_next_day)

        combined_df, forecast_mean, forecast_ci, forecast_index = preprocessing_data(combined_df)
        create_plot(combined_df, forecast_mean, forecast_ci, forecast_index)

    else:
        st.warning("Please run 'Get News Sentiment Data' first before running the full analysis.")



