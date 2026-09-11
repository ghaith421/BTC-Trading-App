
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from datetime import timedelta
from zoneinfo import ZoneInfo
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
import json
import urllib.parse
import urllib.request

warnings.filterwarnings("ignore")

# ============================================================
# BTC INTELLIGENCE - V3 DESIGN
# Educational / simulation dashboard - no order execution
# ============================================================

st.set_page_config(
    page_title="BTC Intelligence — V3",
    page_icon="₿",
    layout="wide",
)

TUNIS_TZ = ZoneInfo("Africa/Tunis")
MODEL_FILE = "btc_multioutput_rf.pkl"
SCALER_FILE = "scaler.pkl"

FEATURES = [
    "RSI",
    "SMA_5",
    "SMA_20",
    "SMA_50",
    "Volatility",
    "Close_lag1",
    "Close_lag2",
    "Close_lag5",
    "High_low_ratio",
    "Close",
]

HORIZONS = 12
BAR_MINUTES = 5
DEFAULT_THRESHOLD = 0.003
DEFAULT_FEE = 0.001

# V3: Binance Spot public market data as the single reference source
MARKET_SOURCE = "Binance Spot"
BINANCE_BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "5m"


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler


try:
    model, scaler = load_model()
except Exception as e:
    st.error("Impossible de charger le modèle/scaler.")
    st.exception(e)
    st.stop()


# ============================================================
# DATA + FEATURES
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def fetch_binance_klines(limit=1000):
    """
    Public Binance market-data endpoints.
    data-api.binance.vision is preferred for public market data because
    Binance documents it specifically for public NONE-security endpoints.
    Multiple official endpoints are tried as fallbacks.
    """
    params = urllib.parse.urlencode({
        "symbol": BINANCE_SYMBOL,
        "interval": BINANCE_INTERVAL,
        "limit": limit,
    })

    errors = []

    for base_url in BINANCE_BASE_URLS:
        url = f"{base_url}/api/v3/klines?{params}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 BTC-Algo-Trading-V3",
                "Accept": "application/json",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))

            if not isinstance(payload, list) or not payload:
                raise ValueError("Réponse vide ou invalide.")

            rows = []
            for k in payload:
                rows.append({
                    "Open time": pd.to_datetime(k[0], unit="ms", utc=True),
                    "Open": float(k[1]),
                    "High": float(k[2]),
                    "Low": float(k[3]),
                    "Close": float(k[4]),
                    "Volume": float(k[5]),
                    "Close time": pd.to_datetime(k[6], unit="ms", utc=True),
                    "Quote volume": float(k[7]),
                    "Trades": int(k[8]),
                })

            df = pd.DataFrame(rows).set_index("Open time")
            df.index = df.index.tz_convert(TUNIS_TZ)
            df = df[~df.index.duplicated(keep="last")].sort_index()

            return df

        except Exception as e:
            errors.append(f"{base_url}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Impossible d'accéder aux endpoints publics Binance depuis "
        "l'environnement Streamlit Cloud. Tentatives: "
        + " | ".join(errors)
    )


@st.cache_data(ttl=5, show_spinner=False)
def fetch_binance_ticker():
    """
    Current BTC/USDT price from Binance public market-data endpoints.
    """
    params = urllib.parse.urlencode({"symbol": BINANCE_SYMBOL})
    errors = []

    for base_url in BINANCE_BASE_URLS:
        url = f"{base_url}/api/v3/ticker/price?{params}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 BTC-Algo-Trading-V3",
                "Accept": "application/json",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))

            return float(payload["price"])

        except Exception as e:
            errors.append(f"{base_url}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Prix live Binance indisponible. Tentatives: "
        + " | ".join(errors)
    )


def add_features(df):
    df = df.copy()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std()

    df["Close_lag1"] = df["Close"].shift(1)
    df["Close_lag2"] = df["Close"].shift(2)
    df["Close_lag5"] = df["Close"].shift(5)

    df["High_low_ratio"] = df["High"] / df["Low"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df


@st.cache_data(ttl=30, show_spinner=False)
def fetch_data():
    df = fetch_binance_klines(limit=1000)
    return add_features(df)


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def detect_supports_resistances(df, order=5):
    if len(df) < 2 * order + 1:
        return [], []

    local_max_idx = argrelextrema(
        df["High"].values, np.greater, order=order
    )[0]
    local_min_idx = argrelextrema(
        df["Low"].values, np.less, order=order
    )[0]

    resistances = df.iloc[local_max_idx]["High"].tail(5).tolist()
    supports = df.iloc[local_min_idx]["Low"].tail(5).tolist()

    return supports, resistances


# ============================================================
# CURRENT PREDICTION
# ============================================================

def predict_current(df):
    # The ML model receives a COMPLETED 5-minute candle.
    # This avoids changing features while the current candle is still forming.
    X = df[FEATURES].iloc[-1:].values
    Xs = scaler.transform(X)
    preds = np.asarray(model.predict(Xs)[0], dtype=float)

    model_reference_price = float(df["Close"].iloc[-1])

    future_times = [
        df.index[-1] + timedelta(minutes=BAR_MINUTES * (i + 1))
        for i in range(HORIZONS)
    ]

    return preds, future_times, model_reference_price


# ============================================================
# SIGNAL SCORE
# ============================================================

def calculate_signal(df, preds, threshold=DEFAULT_THRESHOLD):
    last = df.iloc[-1]
    price = float(last["Close"])

    p1h = float(preds[-1])
    ml_change = (p1h - price) / price

    score = 0

    # ML
    if ml_change >= threshold:
        score += 50
    elif ml_change <= -threshold:
        score -= 50

    # Trend
    if last["SMA_5"] > last["SMA_20"]:
        score += 25
    elif last["SMA_5"] < last["SMA_20"]:
        score -= 25

    # RSI
    rsi = float(last["RSI"])
    if rsi < 30:
        score += 15
    elif rsi > 70:
        score -= 15
    elif rsi < 45:
        score += 7
    elif rsi > 55:
        score -= 7

    if score >= 50:
        signal = "ACHAT 🟢"
        strength = "Forte"
    elif score <= -50:
        signal = "VENTE 🔴"
        strength = "Forte"
    else:
        signal = "ATTENDRE 🟡"
        strength = "Neutre"

    return signal, strength, score, ml_change


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================
# Important:
# The original training script was not supplied. Inspection of
# btc_multioutput_rf.pkl shows:
# - MultiOutputRegressor
# - 12 estimators
# - each estimator is RandomForestRegressor
# - n_estimators=25, max_depth=8, random_state=42
# We therefore reconstruct the most likely target definition:
# y[t] = Close[t+1] ... Close[t+12].
#
# This is an inference from the saved model + the application.
# It should be verified against the original training script
# if that script becomes available.

def make_walk_forward_model():
    base = RandomForestRegressor(
        n_estimators=25,
        max_depth=8,
        n_jobs=-1,
        random_state=42,
    )
    return MultiOutputRegressor(base)


@st.cache_data(ttl=900, show_spinner=False)
def run_walk_forward(
    data_hash,
    close_values,
    feature_values,
    index_values,
    train_size,
    test_size,
    retrain_every,
    threshold,
    fee,
):
    # Rebuild arrays from immutable inputs for Streamlit caching
    close = np.asarray(close_values, dtype=float)
    X_all = np.asarray(feature_values, dtype=float)
    idx = pd.DatetimeIndex(index_values)

    n = len(close)

    # Need 12 future bars to create a target
    usable_n = n - HORIZONS
    if usable_n <= train_size + 20:
        return None

    # Use the latest part as chronological test set
    test_start = max(train_size, usable_n - test_size)
    test_end = usable_n

    predictions = np.full((test_end - test_start, HORIZONS), np.nan)
    actual = np.full((test_end - test_start, HORIZONS), np.nan)

    # Expanding-window walk-forward:
    # retrain periodically, then predict the next bars.
    model_wf = None
    scaler_wf = None
    last_fit_i = -10**9

    for pos in range(test_start, test_end):
        if model_wf is None or (pos - last_fit_i) >= retrain_every:
            train_end = pos

            X_train = X_all[:train_end]
            y_train = np.column_stack(
                [close[h:h + train_end] for h in range(1, HORIZONS + 1)]
            )

            scaler_wf = StandardScaler()
            X_train_scaled = scaler_wf.fit_transform(X_train)

            model_wf = make_walk_forward_model()
            model_wf.fit(X_train_scaled, y_train)
            last_fit_i = pos

        X_now = scaler_wf.transform(X_all[pos:pos + 1])
        p = np.asarray(model_wf.predict(X_now)[0], dtype=float)

        row = pos - test_start
        predictions[row] = p
        actual[row] = close[pos + 1:pos + HORIZONS + 1]

    # 1-hour prediction = horizon 12
    pred_1h = predictions[:, -1]
    actual_1h = actual[:, -1]
    current_close = close[test_start:test_end]

    valid = np.isfinite(pred_1h) & np.isfinite(actual_1h)

    pred_1h = pred_1h[valid]
    actual_1h = actual_1h[valid]
    current_close = current_close[valid]

    pred_change = (pred_1h - current_close) / current_close
    actual_change = (actual_1h - current_close) / current_close

    # Position based on prediction.
    # Shift by one bar to avoid using the current prediction before it exists.
    position = np.where(
        pred_change > threshold, 1,
        np.where(pred_change < -threshold, -1, 0)
    )
    position = np.roll(position, 1)
    position[0] = 0

    market_ret = actual_change
    strategy_gross = position * market_ret

    turnover = np.abs(np.diff(np.r_[0, position]))
    costs = turnover * fee

    strategy_ret = strategy_gross - costs

    equity = np.cumprod(1 + strategy_ret)
    buy_hold = np.cumprod(1 + market_ret)

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1

    trades = int(np.sum(turnover > 0))

    winning = strategy_ret[strategy_ret > 0]
    win_rate = float(np.mean(strategy_ret > 0)) if len(strategy_ret) else 0.0

    ret_std = float(np.std(strategy_ret))
    sharpe = (
        float(np.mean(strategy_ret) / ret_std * np.sqrt(365 * 24))
        if ret_std > 0 else 0.0
    )

    mae = mean_absolute_error(actual_1h, pred_1h)
    rmse = np.sqrt(mean_squared_error(actual_1h, pred_1h))

    direction_pred = np.sign(pred_change)
    direction_actual = np.sign(actual_change)
    direction_acc = float(np.mean(direction_pred == direction_actual))

    # Horizon metrics
    horizon_rows = []
    for h in range(HORIZONS):
        pp = predictions[:, h][valid]
        aa = actual[:, h][valid]
        cc = current_close

        mae_h = mean_absolute_error(aa, pp)
        rmse_h = np.sqrt(mean_squared_error(aa, pp))

        pc = (pp - cc) / cc
        ac = (aa - cc) / cc
        dir_h = float(np.mean(np.sign(pc) == np.sign(ac)))

        horizon_rows.append({
            "Horizon": f"{(h + 1) * 5} min",
            "MAE": mae_h,
            "RMSE": rmse_h,
            "Direction accuracy": dir_h,
        })

    return {
        "index": idx[test_start:test_end][valid],
        "pred": pred_1h,
        "actual": actual_1h,
        "position": position,
        "strategy_ret": strategy_ret,
        "market_ret": market_ret,
        "equity": equity,
        "buy_hold": buy_hold,
        "drawdown": drawdown,
        "trades": trades,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "mae": mae,
        "rmse": rmse,
        "direction_acc": direction_acc,
        "horizon_metrics": pd.DataFrame(horizon_rows),
        "final_multiple": float(equity[-1]) if len(equity) else 1.0,
        "bh_multiple": float(buy_hold[-1]) if len(buy_hold) else 1.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }


# ============================================================
# CHARTS
# ============================================================

def current_chart(df, preds, future_times, supports, resistances):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    recent = df.tail(80)

    fig.add_trace(
        go.Candlestick(
            x=recent.index,
            open=recent["Open"],
            high=recent["High"],
            low=recent["Low"],
            close=recent["Close"],
            name="BTC",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["SMA_20"],
            mode="lines",
            name="SMA 20",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=preds,
            mode="lines+markers",
            name="Prévision ML",
            line=dict(width=3, dash="dot"),
        ),
        row=1,
        col=1,
    )

    for s in supports:
        fig.add_hline(
            y=s,
            line_dash="dash",
            opacity=0.5,
            row=1,
            col=1,
        )

    for r in resistances:
        fig.add_hline(
            y=r,
            line_dash="dash",
            opacity=0.5,
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["RSI"],
            mode="lines",
            name="RSI",
        ),
        row=2,
        col=1,
    )

    fig.add_hline(y=70, line_dash="dash", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", opacity=0.5, row=2, col=1)

    fig.update_layout(
        height=680,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.55)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(148,163,184,0.08)",
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(148,163,184,0.08)",
        zeroline=False
    )

    return fig


def backtest_chart(result):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result["index"],
            y=result["equity"],
            mode="lines",
            name="Stratégie WF",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=result["index"],
            y=result["buy_hold"],
            mode="lines",
            name="Buy & Hold",
        )
    )

    fig.update_layout(
        title="Performance normalisée",
        height=430,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.55)",
        yaxis_title="Capital / Capital initial",
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)", zeroline=False)

    return fig


def drawdown_chart(result):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result["index"],
            y=result["drawdown"] * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
        )
    )

    fig.update_layout(
        title="Drawdown",
        height=320,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.55)",
        yaxis_title="Drawdown (%)",
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)", zeroline=False)

    return fig


# ============================================================

# ============================================================
# V3 — PREMIUM UI
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(247,147,26,.10), transparent 28%),
        radial-gradient(circle at 92% 8%, rgba(59,130,246,.08), transparent 24%),
        #070b12;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
}

section[data-testid="stSidebar"] {
    background: #0b1018;
    border-right: 1px solid rgba(148,163,184,.10);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.hero {
    padding: 1.4rem 1.6rem;
    border: 1px solid rgba(148,163,184,.13);
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(10,15,24,.88));
    box-shadow: 0 20px 60px rgba(0,0,0,.22);
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 2.05rem;
    font-weight: 800;
    letter-spacing: -.04em;
    margin: 0;
}

.hero-subtitle {
    color: #94a3b8;
    margin-top: .25rem;
    font-size: .92rem;
}

.live-pill {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .38rem .72rem;
    border-radius: 999px;
    background: rgba(34,197,94,.09);
    border: 1px solid rgba(34,197,94,.20);
    color: #86efac;
    font-size: .78rem;
    font-weight: 700;
}

.kpi {
    min-height: 118px;
    padding: 1rem 1.05rem;
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,.12);
    background: linear-gradient(145deg, rgba(17,24,39,.94), rgba(10,15,24,.94));
    box-shadow: 0 12px 35px rgba(0,0,0,.16);
}

.kpi-label {
    color: #94a3b8;
    font-size: .72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.kpi-value {
    color: #f8fafc;
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: .35rem;
}

.kpi-value.btc { color: #f7931a; }
.kpi-value.green { color: #4ade80; }
.kpi-value.red { color: #f87171; }
.kpi-value.yellow { color: #facc15; }

.kpi-note {
    color: #64748b;
    font-size: .70rem;
    margin-top: .22rem;
}

.section-title {
    font-size: 1.08rem;
    font-weight: 800;
    color: #e5e7eb;
    margin: 1rem 0 .65rem;
}

.section-caption {
    color: #64748b;
    font-size: .78rem;
    margin-top: -.35rem;
    margin-bottom: .7rem;
}

.signal-card {
    border-radius: 18px;
    padding: 1rem 1.15rem;
    border: 1px solid rgba(148,163,184,.13);
    background: #0d131d;
    margin: .2rem 0 1rem;
}

.signal-main {
    font-size: 1.35rem;
    font-weight: 800;
}

.signal-meta {
    color: #94a3b8;
    font-size: .78rem;
    margin-top: .3rem;
}

.score-track {
    height: 7px;
    background: #1e293b;
    border-radius: 99px;
    overflow: hidden;
    margin-top: .6rem;
}

.score-fill {
    height: 100%;
    border-radius: 99px;
}

.info-strip {
    padding: .65rem .9rem;
    border-radius: 12px;
    background: rgba(30,41,59,.45);
    border: 1px solid rgba(148,163,184,.09);
    color: #94a3b8;
    font-size: .76rem;
    margin-bottom: .8rem;
}

div[data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid rgba(148,163,184,.10);
}

button[data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: .75rem 1rem;
}

div[data-testid="stMetric"] {
    background: rgba(17,24,39,.75);
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 14px;
    padding: .7rem .8rem;
}

.stButton > button {
    border-radius: 11px;
    font-weight: 700;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

hr {
    border-color: rgba(148,163,184,.08);
}

.footer {
    text-align: center;
    color: #475569;
    font-size: .70rem;
    padding-top: 1.4rem;
}
</style>
""", unsafe_allow_html=True)

# Sidebar = compact control center
with st.sidebar:
    st.markdown("## ₿ BTC Intelligence")
    st.caption("V3 • Market analytics")
    st.divider()
    st.markdown("### ⚙️ Configuration")
    refresh_info = st.select_slider(
        "Actualisation des données",
        options=["5s", "15s", "30s", "60s"],
        value="30s",
    )
    auto_refresh = st.toggle("Actualisation automatique", value=False)
    st.divider()
    st.markdown("### 📡 Source")
    st.markdown(f"**{MARKET_SOURCE}**")
    st.caption(f"{BINANCE_SYMBOL} • bougies {BAR_MINUTES} min")
    st.divider()
    st.markdown("### 🧠 Modèle")
    st.caption("Random Forest • 12 horizons")
    st.caption("Référence : dernière bougie clôturée")
    st.divider()
    st.caption("Dashboard éducatif / simulation uniquement.")

if auto_refresh:
    refresh_seconds = {"5s": 5, "15s": 15, "30s": 30, "60s": 60}[refresh_info]
    st.markdown(
        f'<meta http-equiv="refresh" content="{refresh_seconds}">',
        unsafe_allow_html=True
    )

st.markdown("""
<div class="hero">
    <div class="live-pill">● MARKET DATA CONNECTED</div>
    <div class="hero-title">₿ BTC Intelligence</div>
    <div class="hero-subtitle">
        Market overview · Machine Learning · Walk-Forward Analytics
    </div>
</div>
""", unsafe_allow_html=True)

try:
    df = fetch_data()
except Exception as e:
    st.error("⚠️ Impossible de récupérer les données de marché.")
    st.warning(
        "La connexion aux endpoints publics Binance a échoué. "
        "Le modèle ML n'est pas en cause."
    )
    with st.expander("Détails techniques"):
        st.code(str(e))
    st.stop()

if df is None or df.empty:
    st.error("Aucune donnée BTC disponible.")
    st.stop()

try:
    live_price = fetch_binance_ticker()
    data_status = "🟢 Connecté"
except Exception:
    live_price = float(df["Close"].iloc[-1])
    data_status = "🟠 Dernière clôture"

preds, future_times, model_reference_price = predict_current(df)
supports, resistances = detect_supports_resistances(df)
signal, strength, score, pct_change = calculate_signal(df, preds)
live_gap = (live_price - model_reference_price) / model_reference_price
rsi = float(df["RSI"].iloc[-1])
volatility = float(df["Volatility"].iloc[-1])
last_time = df.index[-1]
now_tunis = pd.Timestamp.now(tz=TUNIS_TZ)

signal_class = "green" if score >= 50 else "red" if score <= -50 else "yellow"
signal_icon = "↑" if score >= 50 else "↓" if score <= -50 else "→"
score_pct = min(100, max(0, int(50 + score / 2)))

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">BTC / USDT</div>
      <div class="kpi-value btc">${live_price:,.2f}</div>
      <div class="kpi-note">{data_status} • live</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">Prévision +1H</div>
      <div class="kpi-value {'green' if pct_change >= 0 else 'red'}">${preds[-1]:,.2f}</div>
      <div class="kpi-note">{pct_change:+.2%} vs clôture modèle</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">Signal ML</div>
      <div class="kpi-value {signal_class}">{signal_icon} {signal.split()[0]}</div>
      <div class="kpi-note">Force : {strength}</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">RSI 14</div>
      <div class="kpi-value">{rsi:.1f}</div>
      <div class="kpi-note">Volatilité : {volatility:.2%}</div>
    </div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">Score</div>
      <div class="kpi-value {signal_class}">{score:+d} / 100</div>
      <div class="kpi-note">Indice combiné ML + tendance + RSI</div>
    </div>""", unsafe_allow_html=True)

# Signal panel
st.markdown(f"""
<div class="signal-card">
  <div class="signal-main">{signal_icon} {signal} <span style="color:#94a3b8;font-size:.85rem;">• {strength}</span></div>
  <div class="signal-meta">Score de contexte : {score:+d} · Projection 1h : {pct_change:+.2%}</div>
  <div class="score-track"><div class="score-fill" style="width:{score_pct}%;background:#f7931a;"></div></div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f'<div class="info-strip">📡 <b>{MARKET_SOURCE}</b> · {BINANCE_SYMBOL} · '
    f'Dernière bougie clôturée : <b>{last_time.strftime("%d/%m/%Y %H:%M")}</b> · '
    f'Heure locale : <b>{now_tunis.strftime("%H:%M:%S")}</b> · '
    f'Écart live / clôture modèle : <b>{live_gap:+.3%}</b></div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "📈  OVERVIEW",
    "🧪  WALK-FORWARD",
    "🤖  ML ANALYTICS",
])

with tab1:
    st.markdown('<div class="section-title">📈 Marché & prévision</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">80 dernières bougies · SMA 20 · RSI · supports/résistances · projection ML</div>',
        unsafe_allow_html=True
    )

    fig = current_chart(df, preds, future_times, supports, resistances)
    st.plotly_chart(fig, use_container_width=True, config={
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    })

    left, right = st.columns([1.7, 1])

    with left:
        st.markdown('<div class="section-title">🔮 Projection multi-horizon</div>', unsafe_allow_html=True)
        pred_table = pd.DataFrame({
            "Horizon": [f"{5 * (i + 1)} min" for i in range(HORIZONS)],
            "Heure": [t.strftime("%H:%M:%S") for t in future_times],
            "Prix prédit": [round(float(p), 2) for p in preds],
            "Variation": [
                f"{((p - model_reference_price) / model_reference_price):+.2%}"
                for p in preds
            ],
        })
        st.dataframe(
            pred_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Prix prédit": st.column_config.NumberColumn(format="$%.2f"),
                "Variation": st.column_config.TextColumn(),
            },
        )

    with right:
        st.markdown('<div class="section-title">🎯 Zones techniques</div>', unsafe_allow_html=True)
        if supports:
            st.markdown("**Support**")
            for x in reversed(supports[-3:]):
                st.markdown(f"`$ {x:,.2f}`")
        else:
            st.caption("Aucun support détecté.")
        st.markdown("**Résistance**")
        if resistances:
            for x in reversed(resistances[-3:]):
                st.markdown(f"`$ {x:,.2f}`")
        else:
            st.caption("Aucune résistance détectée.")

    st.markdown(
        '<div class="info-strip">ℹ️ Le modèle utilise uniquement la dernière bougie 5 minutes <b>clôturée</b>. '
        'Le prix live est affiché séparément pour éviter de mélanger une bougie en formation avec les features ML.</div>',
        unsafe_allow_html=True
    )

with tab2:
    st.markdown('<div class="section-title">🧪 Walk-Forward Backtest</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Évaluation chronologique avec réentraînement périodique du modèle.</div>',
        unsafe_allow_html=True
    )

    with st.expander("⚙️ Paramètres du test", expanded=True):
        b1, b2, b3, b4 = st.columns(4)
        train_size = b1.number_input("Taille entraînement", min_value=500, max_value=1500, value=900, step=100)
        test_size = b2.number_input("Taille test", min_value=200, max_value=1000, value=600, step=100)
        retrain_every = b3.number_input("Réentraînement / N bars", min_value=5, max_value=60, value=12, step=1)
        threshold = b4.number_input("Seuil signal", min_value=0.001, max_value=0.02, value=DEFAULT_THRESHOLD, step=0.001, format="%.3f")
        fee = st.number_input("Frais simulés / changement de position", min_value=0.0, max_value=0.01, value=DEFAULT_FEE, step=0.0001, format="%.4f")

    st.markdown(
        '<div class="info-strip">⚠️ Simulation historique uniquement. '
        'Les résultats passés ne garantissent pas les performances futures.</div>',
        unsafe_allow_html=True
    )

    run = st.button("▶  Lancer le Walk-Forward", type="primary", use_container_width=True)

    if run:
        with st.spinner("Analyse en cours — entraînements ML successifs..."):
            feature_values = df[FEATURES].values.astype(float)
            close_values = df["Close"].values.astype(float)
            index_values = df.index.astype(str).tolist()
            data_hash = (len(df), str(df.index[0]), str(df.index[-1]), float(df["Close"].iloc[-1]))
            result = run_walk_forward(
                data_hash, close_values.tolist(), feature_values.tolist(), index_values,
                int(train_size), int(test_size), int(retrain_every),
                float(threshold), float(fee)
            )
        if result is None:
            st.error("Pas assez de données pour ces paramètres.")
        else:
            st.session_state["wf_result"] = result

    result = st.session_state.get("wf_result")
    if result is not None:
        initial_capital = 10000.0
        final_capital = initial_capital * result["final_multiple"]
        bh_final = initial_capital * result["bh_multiple"]

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("Capital final", f"${final_capital:,.2f}")
        r2.metric("Rendement", f"{result['final_multiple'] - 1:+.2%}")
        r3.metric("Buy & Hold", f"{result['bh_multiple'] - 1:+.2%}")
        r4.metric("Trades", f"{result['trades']}")
        r5.metric("Win rate", f"{result['win_rate']:.2%}")
        r6.metric("Max DD", f"{result['max_drawdown']:.2%}")

        st.plotly_chart(backtest_chart(result), use_container_width=True)
        st.plotly_chart(drawdown_chart(result), use_container_width=True)

        st.markdown('<div class="section-title">Comparaison</div>', unsafe_allow_html=True)
        comp = pd.DataFrame({
            "Indicateur": ["Capital final", "Rendement", "Nombre de trades", "Win rate", "Sharpe", "Max drawdown"],
            "Walk-Forward": [
                f"${final_capital:,.2f}", f"{result['final_multiple'] - 1:+.2%}",
                result["trades"], f"{result['win_rate']:.2%}",
                f"{result['sharpe']:.3f}", f"{result['max_drawdown']:.2%}"
            ],
            "Buy & Hold": [
                f"${bh_final:,.2f}", f"{result['bh_multiple'] - 1:+.2%}", "—", "—", "—", "—"
            ],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)

with tab3:
    st.markdown('<div class="section-title">🤖 ML Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Mesure chronologique des erreurs et de la capacité directionnelle à +1h.</div>',
        unsafe_allow_html=True
    )

    if "wf_result" not in st.session_state:
        st.info("Lance le Walk-Forward pour calculer les métriques ML.")
    else:
        result = st.session_state["wf_result"]
        e1, e2, e3 = st.columns(3)
        e1.metric("MAE +1h", f"${result['mae']:,.2f}")
        e2.metric("RMSE +1h", f"${result['rmse']:,.2f}")
        e3.metric("Direction accuracy", f"{result['direction_acc']:.2%}")

        st.markdown('<div class="section-title">Performance par horizon</div>', unsafe_allow_html=True)
        hm = result["horizon_metrics"].copy()
        hm["MAE"] = hm["MAE"].round(2)
        hm["RMSE"] = hm["RMSE"].round(2)
        hm["Direction accuracy"] = (hm["Direction accuracy"] * 100).round(2).astype(str) + "%"
        st.dataframe(hm, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Prix prédit vs prix réel à +1h</div>', unsafe_allow_html=True)
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=result["index"], y=result["actual"], mode="lines", name="Réel +1h",
            line=dict(width=2)
        ))
        fig_eval.add_trace(go.Scatter(
            x=result["index"], y=result["pred"], mode="lines", name="Prédit +1h",
            line=dict(width=2, dash="dot")
        ))
        fig_eval.update_layout(
            height=470, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,23,0.55)",
            xaxis_title="Temps", yaxis_title="Prix BTC",
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0),
        )
        fig_eval.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,.08)", zeroline=False)
        fig_eval.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.08)", zeroline=False)
        st.plotly_chart(fig_eval, use_container_width=True)

# Technical details moved to a compact footer
with st.expander("ℹ️ Détails techniques du modèle"):
    a, b, c = st.columns(3)
    a.write(f"**Type :** `{type(model).__name__}`")
    a.write(f"**Features :** `{getattr(model, 'n_features_in_', 'N/A')}`")
    b.write(f"**Sorties :** `{len(getattr(model, 'estimators_', []))}`")
    c.write(f"**Source :** `{MARKET_SOURCE} — {BINANCE_SYMBOL}`")
    if hasattr(model, "estimator"):
        c.write(
            f"**Random Forest :** {model.estimator.n_estimators} arbres · "
            f"depth={model.estimator.max_depth}"
        )
    st.write("**Features utilisées :**", FEATURES)

st.markdown(
    '<div class="footer">BTC Intelligence V3 · Market data Binance · '
    'Dashboard éducatif et simulation historique · Aucun ordre réel exécuté.</div>',
    unsafe_allow_html=True
)
