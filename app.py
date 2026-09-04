import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import joblib

st.set_page_config(page_title="BTC Algo Trading", layout="wide")
st.title("🤖 Trading BTC - Prédiction 1h")

@st.cache_resource
def load_model():
    model = joblib.load('btc_multioutput_rf.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

@st.cache_data(ttl=300)
def fetch_data():
    df = yf.download('BTC-USD', period='7d', interval='5m', progress=False)
    if df.empty:
        return None, None
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('5min', method='ffill')
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(20).std()
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df['Close_lag5'] = df['Close'].shift(5)
    df['High_low_ratio'] = df['High'] / df['Low']
    df.dropna(inplace=True)
    return df, df[['RSI', 'SMA_5', 'SMA_20', 'SMA_50', 'Volatility', 
                   'Close_lag1', 'Close_lag2', 'Close_lag5', 'High_low_ratio', 'Close']]

def detect_supports_resistances(df, order=5):
    local_max_idx = argrelextrema(df['High'].values, np.greater, order=order)[0]
    local_min_idx = argrelextrema(df['Low'].values, np.less, order=order)[0]
    resistances = df.iloc[local_max_idx]['High'].tail(5).tolist()
    supports = df.iloc[local_min_idx]['Low'].tail(5).tolist()
    return supports, resistances

st.markdown("""<meta http-equiv="refresh" content="300">""", unsafe_allow_html=True)

df, features = fetch_data()

if df is not None and not df.empty:
    last_price = df['Close'].iloc[-1]
    last_time = df.index[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Prix", f"${last_price:.2f}")
    col2.metric("📊 RSI", f"{df['RSI'].iloc[-1]:.2f}")
    col3.metric("📈 Volatilité", f"{df['Volatility'].iloc[-1]:.4f}")
    col4.metric("⏰ MAJ", last_time.strftime("%H:%M:%S"))

    supports, resistances = detect_supports_resistances(df)
    
    features_scaled = scaler.transform(features.iloc[-1:].values)
    preds = model.predict(features_scaled)[0]
    future_times = [df.index[-1] + timedelta(minutes=5*(i+1)) for i in range(12)]
    
    future_price = preds[-1]
    pct_change = (future_price - last_price) / last_price
    if pct_change > 0.003:
        signal = "ACHAT 🟢"
    elif pct_change < -0.003:
        signal = "VENTE 🔴"
    else:
        signal = "ATTENDRE 🟡"
    
    st.markdown(f"## Signal : {signal} (Variation 1h : {pct_change*100:.2f}%)")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index[-50:], open=df['Open'].iloc[-50:], high=df['High'].iloc[-50:],
                                 low=df['Low'].iloc[-50:], close=df['Close'].iloc[-50:], name="Prix"), row=1, col=1)
    for s in supports:
        fig.add_hline(y=s, line_dash="dash", line_color="green", opacity=0.7, row=1, col=1)
    for r in resistances:
        fig.add_hline(y=r, line_dash="dash", line_color="red", opacity=0.7, row=1, col=1)
    fig.add_trace(go.Scatter(x=future_times, y=preds, mode='lines+markers', name='Prédiction 1h', line=dict(color='cyan', width=3, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-50:], y=df['SMA_20'].iloc[-50:], mode='lines', name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-50:], y=df['RSI'].iloc[-50:], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🔍 Détails prédictions"):
        st.dataframe(pd.DataFrame({"Heure": [t.strftime("%H:%M") for t in future_times], "Prix": [round(p,2) for p in preds]}))
else:
    st.warning("Attente des données...")
