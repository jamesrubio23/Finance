#############
##LIBRARIES##
#############

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



def classify_sentiment_batch(titles):
    print(f"🔹 Clasificando {len(titles)} títulos de noticias.")

    prompt = (
        "For each news title below, classify the sentiment as 'POSITIVE', 'NEGATIVE' or 'NEUTRAL'.\n"
        "Return exactly one sentiment per title, in the same order as the titles, and NOTHING ELSE.\n"
        "Make sure to return exactly the same number of lines as the number of news titles.\n"
    )

    prompt += "\n".join(f"- {title}" for title in titles)

    output = llm.invoke(prompt)
    print(f"🔹 Respuesta de Ollama:\n{output}\n")

    valid_sentiments = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
    sentiments = []

    # Dividir la salida en líneas y limpiar los espacios
    for line in output.split("\n"):
        print("Otro")
        line = line.strip().upper()

        sentiment = next((s for s in valid_sentiments if s in line), None)

        if sentiment:
            sentiments.append(sentiment)
        else:
            sentiments.append('NEUTRAL') 

    # Si el número de clasificaciones no coincide con el número de títulos, corregir
    if len(sentiments) != len(titles):
        print(f"ERROR: Ollama devolvió {len(sentiments)} sentimientos en lugar de {len(titles)}")
        
        # Rellenar con "NEUTRAL" si faltan clasificaciones
        while len(sentiments) < len(titles):
            sentiments.append("NEUTRAL")
    print(f"La longitud de los sentiments es {len(sentiments)}")
    print(sentiments)
    return sentiments


# Function to get and process news data
def get_news_data(ticker):
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()

    print("Conseguimos las noticias")
    news_df['Title'] = news_df['Title'].str.lower()

    # Enviar todas las noticias en un solo prompt para evitar múltiples llamadas lentas
    news_df['sentiment'] = classify_sentiment_batch(news_df['Title'].tolist())

    print("Despues de analizar los sentimientos, los añadimos")
    news_df_sent = news_df.copy()
    news_df_sent = news_df_sent[news_df_sent['sentiment'] != 'NEUTRAL'].copy()
    print(f"La longitud de news_df_sent es {len(news_df_sent)}")
    print("Seguimos con las fechas")
    news_df_sent['Date'] = pd.to_datetime(news_df_sent['Date'])
    news_df_sent['DateOnly'] = news_df_sent['Date'].dt.date

    print("Tenemos news_df_sent")
    print(news_df_sent)

    return news_df_sent


# Function to group and process sentiment data
def process_sentiment_data(news_df):
    # Reshape data to have df with columns: Date, # of positive Articles, # of negative Articles
    print(f"Procesamos los datos de news_df con columnas: {news_df.columns}")
    grouped = news_df.groupby(['DateOnly', 'sentiment']).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=['POSITIVE', 'NEGATIVE'], fill_value=0)
    print("Que es grouped")
    print(f"{grouped}")

    # Create rolling averages that count number of positive and negative sentiment articles within past 7 days
    grouped['7day_avg_positive'] = grouped['POSITIVE'].rolling(window=7, min_periods=1).sum()
    grouped['7day_avg_negative'] = grouped['NEGATIVE'].rolling(window=7, min_periods=1).sum()

    # Create "Percent Positive" by creating percentage measure
    grouped['7day_pct_positive'] = grouped['POSITIVE'] / (grouped['POSITIVE'] + grouped['NEGATIVE'])

    result_df = grouped.reset_index()

    print("Que es result_df")
    print(f"{result_df}")

    return result_df

# Function to fetch and process stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)  # Pull ticker data
    stock_data['Pct_Change'] = stock_data['Close'].pct_change() * 100  # Transform closing value to percent change
    return stock_data

# Function to combine sentiment and stock data
def combine_data(result_df, stock_data):
    combined_df = result_df.set_index('DateOnly').join(stock_data[['Pct_Change']], how='inner')
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

def fit_and_forecast(combined_df, forecast_steps=3):
    print("A predecir")
    endog = combined_df['Pct_Change'].dropna()  # Variable dependiente
    exog = combined_df['lagged_7day_pct_positive'].dropna()  # Variable predictora
    print("GOING FOR THE ARIMAX MODEL")
    # Tomar solo los últimos 200 datos (ajusta según el rendimiento)
    endog = endog.tail(200)
    exog = exog.loc[endog.index]  # Alinear

    model = SARIMAX(endog, exog=exog, order=(1, 1, 1))
    fit = model.fit(disp=False, maxiter=50)  # Reduce iteraciones para acelerar

    future_dates = get_future_dates(combined_df.index[-1], forecast_steps)
    future_exog = combined_df['lagged_7day_pct_positive'][-forecast_steps:].values.reshape(-1, 1)

    forecast = fit.get_forecast(steps=forecast_steps, exog=future_exog)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    return forecast_mean, forecast_ci, future_dates



# Function to create and display plot
def create_plot(combined_df, forecast_mean, forecast_ci, forecast_index):
    # Standardize the sentiment proportion
    sentiment_std = (combined_df['7day_pct_positive'] - combined_df['7day_pct_positive'].mean()) / combined_df['7day_pct_positive'].std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=sentiment_std,
        name='Standardized Sentiment Proportion',
        line=dict(color='blue'),
        mode='lines'
    ))

    # Percentage change
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Pct_Change'],
        name='Stock Pct Change',
        line=dict(color='green'),
        yaxis='y2',
        mode='lines'
    ))

    # Añado el forecast percentage
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_mean,
        name='Forecasted Stock Pct Change',
        line=dict(color='red'),
        mode='lines'
    ))

    # Add forecast confidence interval
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
        xaxis_title='Date',
        yaxis=dict(
            title='Standardized Sentiment Proportion',
            titlefont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Stock Pct Change',
            titlefont=dict(color='green'),
            overlaying='y',
            side='right'
        ),
        template='plotly_dark'
    )

    st.plotly_chart(fig)


# PART 3
# STREAMLIT

# Streamlit app
st.sidebar.title("Predicting Stock Prices by News Sentiment")
ticker = st.sidebar.text_input("Enter stock ticker, SBUX?:", value='SBUX')
run_button = st.sidebar.button("Run Analysis")

if run_button:
    news_df = get_news_data(ticker)
    result_df = process_sentiment_data(news_df)
    start_date = result_df['DateOnly'].min().strftime('%Y-%m-%d')
    end_date = result_df['DateOnly'].max().strftime('%Y-%m-%d')
    stock_data = get_stock_data(ticker, start_date, end_date)
    combined_df = combine_data(result_df, stock_data)
    correlation_pct_change = calculate_correlation(combined_df)

    st.write(f"Pearson correlation between lagged sentiment score and stock percentage change: {correlation_pct_change}")

    forecast_mean, forecast_ci, forecast_index = fit_and_forecast(combined_df)
    create_plot(combined_df, forecast_mean, forecast_ci, forecast_index)