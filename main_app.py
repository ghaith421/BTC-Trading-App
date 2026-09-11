
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
# BTC ALGO TRADING - V2.6.1
# Educational / simulation dashboard - no order execution
# ============================================================

st.set_page_config(
    page_title="BTC Algo Trading V2.6.1",
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

# V2.6: Binance Spot public market data as the single reference source
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
                "User-Agent": "Mozilla/5.0 BTC-Algo-Trading-V2.6.1",
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
                "User-Agent": "Mozilla/5.0 BTC-Algo-Trading-V2.6.1",
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
        height=720,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=30, b=20),
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
        title="Courbe de performance normalisée",
        height=450,
        template="plotly_dark",
        yaxis_title="Capital / Capital initial",
    )

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
        height=350,
        template="plotly_dark",
        yaxis_title="Drawdown (%)",
    )

    return fig


# ============================================================
# APP
# ============================================================

st.title("₿ BTC Algo Trading — V2.6.1")
st.caption(
    "Dashboard éducatif et simulation historique. "
    "Aucun ordre réel n'est exécuté."
)

try:
    df = fetch_data()
except Exception as e:
    st.error("⚠️ Impossible de récupérer les données de marché.")
    st.warning(
        "Le problème vient de la connexion entre Streamlit Cloud et "
        "les endpoints Binance, pas du modèle ML. V2.6.1 essaie "
        "plusieurs endpoints publics officiels Binance."
    )
    with st.expander("Détails techniques"):
        st.code(str(e))
    st.stop()

if df is None or df.empty:
    st.error("Aucune donnée BTC disponible.")
    st.stop()

try:
    live_price = fetch_binance_ticker()
    data_status = "🟢 Binance connecté"
except Exception as e:
    live_price = float(df["Close"].iloc[-1])
    data_status = "🟠 Prix live indisponible — dernière clôture utilisée"

# Current state
preds, future_times, model_reference_price = predict_current(df)

# Important: prediction is relative to the completed candle used by the model.
# The live market price is displayed separately.
last_price = live_price
supports, resistances = detect_supports_resistances(df)

signal, strength, score, pct_change = calculate_signal(
    df, preds
)

# Live-vs-model-reference diagnostic
live_gap = (live_price - model_reference_price) / model_reference_price

last_time = df.index[-1]
now_tunis = pd.Timestamp.now(tz=TUNIS_TZ)

# ============================================================
# HEADER METRICS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Prix BTC", f"${last_price:,.2f}")
c2.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
c3.metric("Volatilité", f"{df['Volatility'].iloc[-1]:.4%}")
c4.metric("Variation ML 1h", f"{pct_change:.2%}")
c5.metric("Score", f"{score:+d}")

st.info(
    f"Signal: **{signal}** — Force: **{strength}** | "
    f"Source prix: **{MARKET_SOURCE} / {BINANCE_SYMBOL}** | "
    f"{data_status} | "
    f"Dernière bougie: **{last_time.strftime('%d/%m/%Y %H:%M:%S')}** "
    f"| Heure locale: **{now_tunis.strftime('%H:%M:%S')}**"
)

st.caption(
    "Source de référence: Binance public market data. "
    "V2.6.1 utilise data-api.binance.vision en priorité, puis plusieurs "
    "endpoints officiels Binance de secours. "
)

st.caption(
    f"Prix live Binance: **${live_price:,.2f}** · "
    f"Clôture 5 min utilisée par le modèle: **${model_reference_price:,.2f}** · "
    f"Écart live/clôture: **{live_gap:+.3%}**"
)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📈 Marché & prédiction",
    "🧪 Walk-Forward Backtest",
    "🤖 Évaluation ML",
])

# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.subheader("Prix BTC + prévision ML")

    fig = current_chart(
        df, preds, future_times, supports, resistances
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("12 horizons de prédiction")

    pred_table = pd.DataFrame({
        "Horizon": [f"{5 * (i + 1)} min" for i in range(HORIZONS)],
        "Heure": [
            t.strftime("%H:%M:%S") for t in future_times
        ],
        "Prix prédit": [
            round(float(p), 2) for p in preds
        ],
        "Variation vs actuel": [
            f"{((p - last_price) / last_price):.2%}"
            for p in preds
        ],
    })

    st.dataframe(
        pred_table,
        use_container_width=True,
        hide_index=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Supports")
        if supports:
            st.write([f"${x:,.2f}" for x in supports])
        else:
            st.write("Aucun support détecté.")

    with col_b:
        st.subheader("Résistances")
        if resistances:
            st.write([f"${x:,.2f}" for x in resistances])
        else:
            st.write("Aucune résistance détectée.")

# ============================================================
# TAB 2 - WALK FORWARD
# ============================================================

with tab2:
    st.subheader("Walk-Forward Backtest")

    st.warning(
        "V2.6 utilise Binance Spot BTC/USDT comme source de référence "
        "unique pour les bougies 5 minutes et le prix live. "
        "Le modèle reçoit uniquement une bougie 5 minutes clôturée. "
        "Le backtest reste une simulation historique et ne garantit "
        "aucune performance future."
    )

    st.markdown(
        "**Pourquoi Binance ?** Il n'existe pas un prix BTC mondial unique. "
        "Un prix affiché par Google, Yahoo ou une plateforme peut différer "
        "légèrement d'un exchange à l'autre. V2.6 choisit donc une source "
        "de marché précise et reproductible au lieu de mélanger plusieurs sources."
    )

    b1, b2, b3, b4 = st.columns(4)

    train_size = b1.number_input(
        "Taille entraînement",
        min_value=500,
        max_value=1500,
        value=900,
        step=100,
    )

    test_size = b2.number_input(
        "Taille test",
        min_value=200,
        max_value=1000,
        value=600,
        step=100,
    )

    retrain_every = b3.number_input(
        "Réentraînement tous les N bars",
        min_value=5,
        max_value=60,
        value=12,
        step=1,
    )

    threshold = b4.number_input(
        "Seuil signal",
        min_value=0.001,
        max_value=0.02,
        value=DEFAULT_THRESHOLD,
        step=0.001,
        format="%.3f",
    )

    fee = st.number_input(
        "Frais simulés par changement de position",
        min_value=0.0,
        max_value=0.01,
        value=DEFAULT_FEE,
        step=0.0001,
        format="%.4f",
    )

    run = st.button(
        "▶ Lancer le Walk-Forward Backtest",
        type="primary",
        use_container_width=True,
    )

    if run:
        with st.spinner(
            "Walk-forward en cours — plusieurs entraînements ML peuvent prendre du temps..."
        ):
            feature_values = df[FEATURES].values.astype(float)
            close_values = df["Close"].values.astype(float)
            index_values = df.index.astype(str).tolist()

            data_hash = (
                len(df),
                str(df.index[0]),
                str(df.index[-1]),
                float(df["Close"].iloc[-1]),
            )

            result = run_walk_forward(
                data_hash,
                close_values.tolist(),
                feature_values.tolist(),
                index_values,
                int(train_size),
                int(test_size),
                int(retrain_every),
                float(threshold),
                float(fee),
            )

        if result is None:
            st.error(
                "Pas assez de données pour ces paramètres. "
                "Réduis la taille d'entraînement ou du test."
            )
        else:
            st.session_state["wf_result"] = result

    result = st.session_state.get("wf_result")

    if result is not None:
        initial_capital = 10000.0
        final_capital = initial_capital * result["final_multiple"]
        bh_final = initial_capital * result["bh_multiple"]

        r1, r2, r3, r4, r5, r6 = st.columns(6)

        r1.metric(
            "Capital final",
            f"${final_capital:,.2f}",
        )
        r2.metric(
            "Rendement stratégie",
            f"{(result['final_multiple'] - 1):.2%}",
        )
        r3.metric(
            "Buy & Hold",
            f"{(result['bh_multiple'] - 1):.2%}",
        )
        r4.metric(
            "Trades",
            f"{result['trades']}",
        )
        r5.metric(
            "Win rate",
            f"{result['win_rate']:.2%}",
        )
        r6.metric(
            "Max Drawdown",
            f"{result['max_drawdown']:.2%}",
        )

        st.metric("Sharpe simulé", f"{result['sharpe']:.3f}")

        st.plotly_chart(
            backtest_chart(result),
            use_container_width=True,
        )

        st.plotly_chart(
            drawdown_chart(result),
            use_container_width=True,
        )

        st.subheader("Comparaison stratégie / Buy & Hold")

        comp = pd.DataFrame({
            "Indicateur": [
                "Capital final",
                "Rendement",
                "Nombre de trades",
                "Win rate",
                "Sharpe",
                "Max drawdown",
            ],
            "Walk-Forward": [
                f"${final_capital:,.2f}",
                f"{result['final_multiple'] - 1:.2%}",
                result["trades"],
                f"{result['win_rate']:.2%}",
                f"{result['sharpe']:.3f}",
                f"{result['max_drawdown']:.2%}",
            ],
            "Buy & Hold": [
                f"${bh_final:,.2f}",
                f"{result['bh_multiple'] - 1:.2%}",
                "—",
                "—",
                "—",
                "—",
            ],
        })

        st.dataframe(
            comp,
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# TAB 3 - ML EVALUATION
# ============================================================

with tab3:
    st.subheader("Évaluation chronologique du modèle")

    st.write(
        "Cette section mesure les erreurs sur une période de test "
        "chronologique et utilise le même principe de targets "
        "Close(t+1) ... Close(t+12)."
    )

    if "wf_result" not in st.session_state:
        st.info(
            "Lance d'abord le Walk-Forward Backtest pour obtenir "
            "les métriques d'évaluation."
        )
    else:
        result = st.session_state["wf_result"]

        e1, e2, e3 = st.columns(3)

        e1.metric(
            "MAE 1h",
            f"${result['mae']:,.2f}",
        )
        e2.metric(
            "RMSE 1h",
            f"${result['rmse']:,.2f}",
        )
        e3.metric(
            "Direction accuracy 1h",
            f"{result['direction_acc']:.2%}",
        )

        st.subheader("Performance par horizon")

        hm = result["horizon_metrics"].copy()
        hm["MAE"] = hm["MAE"].round(2)
        hm["RMSE"] = hm["RMSE"].round(2)
        hm["Direction accuracy"] = (
            hm["Direction accuracy"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            hm,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Prédiction 1h vs réalité")

        fig_eval = go.Figure()

        fig_eval.add_trace(
            go.Scatter(
                x=result["index"],
                y=result["actual"],
                mode="lines",
                name="Prix réel à +1h",
            )
        )

        fig_eval.add_trace(
            go.Scatter(
                x=result["index"],
                y=result["pred"],
                mode="lines",
                name="Prix prédit à +1h",
            )
        )

        fig_eval.update_layout(
            height=500,
            template="plotly_dark",
            xaxis_title="Temps",
            yaxis_title="Prix BTC",
        )

        st.plotly_chart(
            fig_eval,
            use_container_width=True,
        )

# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("ℹ️ Informations sur le modèle"):
    st.write(
        f"**Type :** `{type(model).__name__}`"
    )
    st.write(
        f"**Nombre de features :** `{getattr(model, 'n_features_in_', 'N/A')}`"
    )
    st.write(
        f"**Nombre de sorties :** `{len(getattr(model, 'estimators_', []))}`"
    )

    if hasattr(model, "estimator"):
        st.write(
            f"**Random Forest :** "
            f"{model.estimator.n_estimators} arbres, "
            f"max_depth={model.estimator.max_depth}, "
            f"random_state={model.estimator.random_state}"
        )

    st.write(
        "**Features utilisées :**",
        FEATURES,
    )
    st.write(
        f"**Source marché V2.6 :** {MARKET_SOURCE} — {BINANCE_SYMBOL}"
    )
    st.write(
        "**Bougies :** Spot 5 minutes ; prix live séparé du dernier close."
    )

st.caption(
    "V2.6.1 — données Yahoo Finance / modèle ML existant / "
    "backtest uniquement simulé."
)
