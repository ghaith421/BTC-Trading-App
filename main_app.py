
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
    page_title="BTC Intelligence — V3.3",
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
# V3.3 - OUT-OF-SAMPLE FORECAST RELIABILITY
# ============================================================
# The model's tree dispersion is useful as an uncertainty indicator,
# but it is NOT a calibrated prediction interval. V3.3 therefore
# estimates empirical forecast errors with a walk-forward procedure.
# These statistics are descriptive/educational and do not guarantee
# future accuracy.

@st.cache_data(ttl=900, show_spinner=False)
def compute_forecast_reliability(
    close_values,
    feature_values,
    index_values,
    train_size=650,
    test_size=260,
    retrain_every=12,
):
    close = np.asarray(close_values, dtype=float)
    X_all = np.asarray(feature_values, dtype=float)
    idx = pd.DatetimeIndex(index_values)

    n = len(close)
    if n <= train_size + test_size + HORIZONS + 10:
        return None

    usable_n = n - HORIZONS
    max_test_start = usable_n - test_size
    if max_test_start <= train_size:
        return None

    # Keep the calibration computation bounded for Streamlit Cloud.
    test_start = max(train_size, max_test_start)
    test_end = usable_n

    records = []
    cached_model = None

    for t in range(test_start, test_end):
        if cached_model is None or (t - test_start) % retrain_every == 0:
            model_fw = make_walk_forward_model()
            y_train = np.column_stack(
                [close[j + 1 : j + 1 + HORIZONS] for j in range(t - train_size, t)]
            )
            X_train = X_all[t - train_size : t]

            # Guard against NaN/inf in reconstructed features.
            mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train).all(axis=1)
            if mask.sum() < max(100, int(0.7 * len(mask))):
                continue

            X_train_clean = X_train[mask]
            y_train_clean = y_train[mask]

            # The saved scaler is part of the original live model.
            # For this out-of-sample diagnostic, fit a fresh scaler only
            # on past training data to avoid any future-data leakage.
            scaler_fw = StandardScaler()
            Xs_train = scaler_fw.fit_transform(X_train_clean)
            model_fw.fit(Xs_train, y_train_clean)
            cached_model = (model_fw, scaler_fw)

        if cached_model is None:
            continue

        model_fw, scaler_fw = cached_model
        X_test = X_all[t : t + 1]
        if not np.isfinite(X_test).all():
            continue

        pred = np.asarray(
            model_fw.predict(scaler_fw.transform(X_test))[0],
            dtype=float,
        )
        actual = close[t + 1 : t + 1 + HORIZONS]

        if len(actual) != HORIZONS:
            continue

        for h in range(HORIZONS):
            records.append({
                "t": t,
                "h": h + 1,
                "pred": float(pred[h]),
                "actual": float(actual[h]),
                "error": float(actual[h] - pred[h]),
                "abs_error_pct": float(abs(actual[h] - pred[h]) / max(abs(actual[h]), 1e-9) * 100),
                "direction_ok": int(
                    np.sign(pred[h] - close[t]) == np.sign(actual[h] - close[t])
                ),
            })

    if not records:
        return None

    res = pd.DataFrame(records)

    # Horizon-level empirical statistics.
    horizon_rows = []
    for h in range(1, HORIZONS + 1):
        g = res[res["h"] == h]
        if g.empty:
            continue
        err = g["error"].to_numpy()
        horizon_rows.append({
            "Horizon": h,
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE_pct": float(g["abs_error_pct"].mean()),
            "Direction": float(g["direction_ok"].mean() * 100),
            "Q10_error": float(np.quantile(err, 0.10)),
            "Q90_error": float(np.quantile(err, 0.90)),
            "N": int(len(g)),
        })

    horizon_df = pd.DataFrame(horizon_rows)

    if horizon_df.empty:
        return None

    overall = {
        "MAE_pct": float(res["abs_error_pct"].mean()),
        "Direction": float(res["direction_ok"].mean() * 100),
        "N": int(len(res)),
        "Observations": int(res["t"].nunique()),
    }

    return horizon_df, overall


def calibrated_interval_for_horizon(
    reference_price,
    central_prediction,
    horizon,
    reliability,
):
    """Build an empirical interval from past out-of-sample residuals."""
    if reliability is None:
        return central_prediction, central_prediction, None

    horizon_df, _ = reliability
    row = horizon_df[horizon_df["Horizon"] == int(horizon)]
    if row.empty:
        return central_prediction, central_prediction, None

    q10 = float(row["Q10_error"].iloc[0])
    q90 = float(row["Q90_error"].iloc[0])

    # Residual interval is empirical, not a probability guarantee.
    low = central_prediction + q10
    high = central_prediction + q90
    return float(low), float(high), float(row["MAE_pct"].iloc[0])


def reliability_label(direction_accuracy, mae_pct, sample_n):
    if sample_n < 100:
        return "ÉCHANTILLON LIMITÉ"
    if direction_accuracy >= 58 and mae_pct <= 0.60:
        return "BONNE"
    if direction_accuracy >= 52 and mae_pct <= 1.00:
        return "MOYENNE"
    return "FAIBLE"


def reliability_adjusted_quality(base_quality, reliability):
    """Blend current model coherence with observed out-of-sample behavior."""
    if reliability is None:
        return base_quality, "NON CALIBRÉE"

    _, overall = reliability
    direction = overall["Direction"]
    mae_pct = overall["MAE_pct"]

    # Direction component: 50% at random -> 100% at 75%+.
    direction_component = np.clip((direction - 50) / 25, 0, 1) * 100
    # Error component: <=0.25% is excellent, >=1.50% is weak.
    error_component = np.clip(1 - (mae_pct - 0.25) / 1.25, 0, 1) * 100

    empirical = 0.60 * direction_component + 0.40 * error_component
    final_quality = float(0.45 * base_quality + 0.55 * empirical)

    label = reliability_label(direction, mae_pct, overall["Observations"])
    return final_quality, label


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
# FORECAST QUALITY / UNCERTAINTY
# ============================================================
def get_ensemble_forecast(df):
    """
    Returns the model forecast plus an empirical dispersion estimate
    from the individual Random Forest trees.

    IMPORTANT:
    Tree dispersion is NOT a calibrated probability. It is used only
    as an uncertainty / model-agreement indicator.
    """
    X = df[FEATURES].iloc[-1:].values
    Xs = scaler.transform(X)

    means = []
    lows = []
    highs = []
    dispersions = []

    estimators = getattr(model, "estimators_", [])
    for est in estimators:
        if hasattr(est, "estimators_") and len(est.estimators_) > 1:
            tree_preds = np.array([
                float(tree.predict(Xs)[0]) for tree in est.estimators_
            ])
            means.append(float(np.mean(tree_preds)))
            lows.append(float(np.percentile(tree_preds, 10)))
            highs.append(float(np.percentile(tree_preds, 90)))
            dispersions.append(float(np.std(tree_preds)))
        else:
            value = float(est.predict(Xs)[0])
            means.append(value)
            lows.append(value)
            highs.append(value)
            dispersions.append(0.0)

    preds = np.asarray(means, dtype=float)
    p10 = np.asarray(lows, dtype=float)
    p90 = np.asarray(highs, dtype=float)
    dispersion = np.asarray(dispersions, dtype=float)

    reference = float(df["Close"].iloc[-1])
    return preds, p10, p90, dispersion, reference


def forecast_quality(df, preds, p10, p90, dispersion):
    """
    Educational forecast-quality score.

    It combines:
    - agreement of RF trees
    - agreement between short and long horizons
    - trend alignment
    - RSI context
    - volatility regime

    It is deliberately called QUALITY, not probability of profit.
    """
    last = df.iloc[-1]
    price = float(last["Close"])

    # 1) Model dispersion: lower relative dispersion = better agreement.
    rel_disp = float(np.mean(dispersion) / max(price, 1e-9))
    dispersion_quality = 100.0 * np.clip(1.0 - rel_disp / 0.004, 0, 1)

    # 2) Horizon consistency: compare the signs of consecutive forecast moves.
    moves = np.diff(np.r_[price, preds]) / price
    positive = np.sum(moves > 0)
    negative = np.sum(moves < 0)
    direction_consensus = max(positive, negative) / max(len(moves), 1)
    consensus_quality = direction_consensus * 100.0

    # 3) Trend alignment.
    sma5 = float(last["SMA_5"])
    sma20 = float(last["SMA_20"])
    trend_up = sma5 > sma20
    forecast_up = float(preds[-1]) > price
    trend_quality = 100.0 if trend_up == forecast_up else 35.0

    # 4) RSI is used as context, not as a buy/sell trigger.
    rsi = float(last["RSI"])
    rsi_quality = 100.0 if 35 <= rsi <= 65 else 65.0

    # 5) Volatility regime: extreme volatility lowers reliability.
    vol = float(last["Volatility"])
    volatility_quality = 100.0 * np.clip(1.0 - max(vol - 0.004, 0) / 0.012, 0, 1)

    quality = (
        0.35 * dispersion_quality
        + 0.25 * consensus_quality
        + 0.20 * trend_quality
        + 0.10 * rsi_quality
        + 0.10 * volatility_quality
    )
    quality = float(np.clip(quality, 0, 100))

    if quality >= 75:
        label = "ÉLEVÉE"
    elif quality >= 55:
        label = "MOYENNE"
    else:
        label = "FAIBLE"

    # Direction is only considered meaningful when the move exceeds
    # both the model uncertainty and a small practical noise floor.
    horizon_move = (float(preds[-1]) - price) / price
    uncertainty = (float(p90[-1]) - float(p10[-1])) / (2 * price)
    noise_floor = max(0.0015, uncertainty)

    if abs(horizon_move) < noise_floor:
        direction = "INCERTAINE"
    elif horizon_move > 0:
        direction = "HAUSSIÈRE"
    else:
        direction = "BAISSIÈRE"

    return {
        "quality": quality,
        "label": label,
        "direction": direction,
        "uncertainty": uncertainty,
        "dispersion_rel": rel_disp,
        "consensus": direction_consensus,
    }


def decision_framework(df, preds, p10, p90, quality_info):
    """
    Produces an educational decision state:
    FAVORABLE / ATTENDRE / PRUDENCE.

    This is deliberately not a financial recommendation.
    """
    price = float(df["Close"].iloc[-1])
    move = (float(preds[-1]) - price) / price
    band = quality_info["uncertainty"]

    # Require the forecast to clear the uncertainty band.
    signal_strength = abs(move) / max(band, 0.001)

    if quality_info["quality"] < 55:
        state = "PRUDENCE"
        reason = "Qualité du modèle insuffisante"
    elif signal_strength < 1.25:
        state = "ATTENDRE"
        reason = "Projection trop proche de l'incertitude"
    elif quality_info["direction"] == "HAUSSIÈRE":
        state = "SCÉNARIO HAUSSIER"
        reason = "Projection au-dessus de la zone d'incertitude"
    elif quality_info["direction"] == "BAISSIÈRE":
        state = "SCÉNARIO BAISSIER"
        reason = "Projection sous la zone d'incertitude"
    else:
        state = "ATTENDRE"
        reason = "Direction non confirmée"

    return state, reason, signal_strength


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
# ============================================================
# V3.3 — PREMIUM TERMINAL UI
# ============================================================

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root { --bg:#060a10; --panel:#0b111a; --panel2:#0f1722; --line:rgba(148,163,184,.12); --muted:#8290a5; --text:#edf2f7; --btc:#f7931a; --green:#35d07f; --red:#ff5c6c; --blue:#5b8cff; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp { background: radial-gradient(circle at 15% -10%,rgba(247,147,26,.13),transparent 25%), radial-gradient(circle at 100% 5%,rgba(91,140,255,.12),transparent 28%), linear-gradient(180deg,#060a10 0%,#080d14 100%); color:var(--text); }
.block-container { max-width:1540px; padding:1.0rem 2rem 3rem; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"] { visibility:hidden; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#080d14,#070b11); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] > div { padding:1.2rem 1rem; }

.brand { display:flex; align-items:center; gap:.75rem; }
.brand-icon { width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;background:linear-gradient(145deg,#f7931a,#ffb04a);color:#111;font-size:1.45rem;font-weight:900;box-shadow:0 10px 30px rgba(247,147,26,.18); }
.brand-name { font-size:1.05rem;font-weight:800;letter-spacing:-.02em; }
.brand-sub { color:var(--muted);font-size:.68rem;margin-top:.12rem; }

.hero { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:26px; padding:1.55rem 1.65rem; margin-bottom:1rem; background:linear-gradient(135deg,rgba(15,23,34,.96),rgba(8,13,20,.88)); box-shadow:0 24px 80px rgba(0,0,0,.28); }
.hero:after { content:""; position:absolute; width:320px;height:320px;right:-110px;top:-180px;border-radius:50%;background:rgba(247,147,26,.12);filter:blur(12px); }
.hero-grid { display:flex;justify-content:space-between;align-items:center;gap:1rem;position:relative;z-index:2; }
.hero-title { font-size:2.15rem;font-weight:800;letter-spacing:-.055em;margin:.45rem 0 .2rem; }
.hero-sub { color:#8e9bae;font-size:.82rem; }
.status { display:inline-flex;align-items:center;gap:.45rem;border:1px solid rgba(53,208,127,.25);background:rgba(53,208,127,.08);color:#7df0ac;padding:.38rem .68rem;border-radius:999px;font-size:.68rem;font-weight:800;letter-spacing:.06em; }
.status-dot { width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px rgba(53,208,127,.75); }
.hero-right { text-align:right;min-width:170px; }
.hero-price-label { color:var(--muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.09em; }
.hero-price { font-size:1.8rem;font-weight:800;margin-top:.15rem; }
.hero-gap { font-size:.72rem;color:#8e9bae;margin-top:.15rem; }

.kpi-grid { display:grid;grid-template-columns:repeat(5,1fr);gap:.75rem;margin:.8rem 0 1rem; }
.kpi { min-height:112px;padding:1rem 1.05rem;border-radius:18px;border:1px solid var(--line);background:linear-gradient(145deg,rgba(15,23,34,.92),rgba(9,14,22,.96));box-shadow:0 12px 35px rgba(0,0,0,.16); }
.kpi-top { display:flex;justify-content:space-between;align-items:center;gap:.5rem; }
.kpi-label { color:#7f8da2;font-size:.64rem;font-weight:800;text-transform:uppercase;letter-spacing:.09em; }
.kpi-icon { width:27px;height:27px;border-radius:9px;background:rgba(148,163,184,.08);display:flex;align-items:center;justify-content:center;font-size:.8rem; }
.kpi-value { font-size:1.42rem;font-weight:800;margin-top:.42rem;letter-spacing:-.03em; }
.kpi-note { color:#637084;font-size:.67rem;margin-top:.22rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.btc { color:var(--btc); }.green { color:var(--green); }.red { color:var(--red); }.blue { color:#8eaeff; }.yellow { color:#f5cc5d; }

.panel { border:1px solid var(--line);border-radius:20px;background:rgba(11,17,26,.82);box-shadow:0 14px 45px rgba(0,0,0,.14);padding:1rem 1.05rem;margin-bottom:.85rem; }
.panel-head { display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;margin-bottom:.75rem; }
.panel-title { font-size:.96rem;font-weight:800;letter-spacing:-.02em; }
.panel-sub { color:#657287;font-size:.68rem;margin-top:.18rem; }
.badge { display:inline-flex;padding:.3rem .55rem;border-radius:8px;background:rgba(91,140,255,.09);border:1px solid rgba(91,140,255,.15);color:#9bb5ff;font-size:.62rem;font-weight:800; }

.signal { display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:1rem;border:1px solid var(--line);border-radius:18px;padding:.85rem 1rem;background:linear-gradient(90deg,rgba(247,147,26,.055),rgba(91,140,255,.035));margin-bottom:.85rem; }
.signal-icon { width:44px;height:44px;border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:1.35rem;font-weight:900;background:rgba(247,147,26,.10);border:1px solid rgba(247,147,26,.16); }
.signal-name { font-size:1.05rem;font-weight:800; }.signal-desc { color:#7e8ba0;font-size:.69rem;margin-top:.15rem; }
.score-box { text-align:right;min-width:120px; }.score-num { font-size:1.2rem;font-weight:800; }.score-track { height:6px;background:#1a2431;border-radius:99px;overflow:hidden;margin-top:.35rem; }.score-fill { height:100%;border-radius:99px;background:linear-gradient(90deg,#f7931a,#ffd166); }

.info-strip { display:flex;flex-wrap:wrap;gap:.55rem .9rem;padding:.62rem .78rem;border-radius:12px;background:rgba(15,23,34,.72);border:1px solid var(--line);color:#7f8da2;font-size:.66rem;margin-bottom:.85rem; }
.info-item b { color:#dbe3ed; }

.section-title { font-size:1rem;font-weight:800;margin:.3rem 0 .2rem; }.section-caption { color:#68768b;font-size:.69rem;margin-bottom:.65rem; }

div[data-baseweb="tab-list"] { gap:4px;border-bottom:1px solid var(--line); }
button[data-baseweb="tab"] { border-radius:10px 10px 0 0;padding:.7rem .9rem;color:#7f8da2;font-weight:700; }
button[data-baseweb="tab"][aria-selected="true"] { color:#f5f7fa; }

.stButton > button { border-radius:11px;font-weight:800;min-height:2.6rem; }
div[data-testid="stDataFrame"] { border:1px solid var(--line);border-radius:14px;overflow:hidden; }
div[data-testid="stMetric"] { background:rgba(15,23,34,.78);border:1px solid var(--line);border-radius:14px;padding:.7rem .8rem; }
.stAlert { border-radius:14px; }
hr { border-color:var(--line); }
.footer { text-align:center;color:#3f4d61;font-size:.64rem;padding-top:1.4rem; }

@media (max-width: 1100px) { .kpi-grid{grid-template-columns:repeat(3,1fr);} .hero-grid{align-items:flex-start;} }
@media (max-width: 700px) { .block-container{padding:1rem .75rem 2rem;} .kpi-grid{grid-template-columns:repeat(2,1fr);} .hero-grid{flex-direction:column;align-items:flex-start;} .hero-right{text-align:left;} .hero-title{font-size:1.7rem;} .signal{grid-template-columns:auto 1fr;} .score-box{grid-column:2;text-align:left;} }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">₿</div><div><div class="brand-name">BTC Intelligence</div><div class="brand-sub">V3.3 • FORECAST LAB</div></div></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**⚙️ CONTRÔLE DU DASHBOARD**")
    refresh_info = st.select_slider("Actualisation", options=["5s","15s","30s","60s"], value="30s")
    auto_refresh = st.toggle("Actualisation automatique", value=False)
    st.divider()
    st.markdown("**📡 MARCHÉ**")
    st.caption(f"{MARKET_SOURCE} · {BINANCE_SYMBOL}")
    st.caption(f"Bougies {BAR_MINUTES} min · prix live séparé")
    st.divider()
    st.markdown("**🧠 MODÈLE**")
    st.caption("Random Forest · 12 horizons")
    st.caption("Features : RSI, SMA, volatilité, lags")
    st.divider()
    st.caption("Mode éducatif / simulation historique. Aucun ordre réel n'est exécuté.")

if auto_refresh:
    refresh_seconds = {"5s":5,"15s":15,"30s":30,"60s":60}[refresh_info]
    st.markdown(f'<meta http-equiv="refresh" content="{refresh_seconds}">', unsafe_allow_html=True)

# -----------------------------
# Data
# -----------------------------
try:
    df = fetch_data()
except Exception as e:
    st.error("⚠️ Impossible de récupérer les données de marché.")
    st.warning("La connexion aux endpoints publics Binance a échoué. Le modèle ML n'est pas en cause.")
    with st.expander("Détails techniques"):
        st.code(str(e))
    st.stop()

if df is None or df.empty:
    st.error("Aucune donnée BTC disponible.")
    st.stop()

try:
    live_price = fetch_binance_ticker()
    data_status = "Connecté"
except Exception:
    live_price = float(df["Close"].iloc[-1])
    data_status = "Dernière clôture"

preds, p10_tree, p90_tree, tree_dispersion, model_reference_price = get_ensemble_forecast(df)
future_times = [df.index[-1] + timedelta(minutes=BAR_MINUTES * (i + 1)) for i in range(HORIZONS)]
supports, resistances = detect_supports_resistances(df)

# Empirical out-of-sample reliability/calibration.
reliability = compute_forecast_reliability(
    df["Close"].values,
    df[FEATURES].values,
    df.index.values,
    train_size=650,
    test_size=260,
    retrain_every=12,
)

# Replace the raw tree spread with empirical residual intervals where available.
p10 = np.zeros(HORIZONS, dtype=float)
p90 = np.zeros(HORIZONS, dtype=float)
empirical_mae_pct = []
for h in range(1, HORIZONS + 1):
    lo, hi, mae_pct = calibrated_interval_for_horizon(
        model_reference_price,
        float(preds[h - 1]),
        h,
        reliability,
    )
    if reliability is None:
        lo = float(p10_tree[h - 1])
        hi = float(p90_tree[h - 1])
    p10[h - 1] = lo
    p90[h - 1] = hi
    if mae_pct is not None:
        empirical_mae_pct.append(mae_pct)

base_quality_info = forecast_quality(df, preds, p10_tree, p90_tree, tree_dispersion)
quality_score, reliability_label_text = reliability_adjusted_quality(
    base_quality_info["quality"], reliability
)
quality_info = dict(base_quality_info)
quality_info["quality"] = quality_score
quality_info["reliability"] = reliability_label_text

decision_state, decision_reason, signal_strength = decision_framework(
    df, preds, p10, p90, quality_info
)
signal, strength, score, pct_change = calculate_signal(df, preds)
live_gap = (live_price - model_reference_price) / model_reference_price
rsi = float(df["RSI"].iloc[-1])
volatility = float(df["Volatility"].iloc[-1])
last_time = df.index[-1]
now_tunis = pd.Timestamp.now(tz=TUNIS_TZ)

signal_class = "green" if quality_info["direction"] == "HAUSSIÈRE" else "red" if quality_info["direction"] == "BAISSIÈRE" else "yellow"
signal_icon = "↑" if quality_info["direction"] == "HAUSSIÈRE" else "↓" if quality_info["direction"] == "BAISSIÈRE" else "→"
score_pct = int(np.clip(quality_info["quality"], 0, 100))

# -----------------------------
# Hero
# -----------------------------
st.markdown(f"""
<div class="hero">
  <div class="hero-grid">
    <div>
      <div class="status"><span class="status-dot"></span> {data_status.upper()} · {MARKET_SOURCE.upper()}</div>
      <div class="hero-title">₿ BTC Intelligence</div>
      <div class="hero-sub">Real-time market view · Machine Learning · Walk-Forward Analytics</div>
    </div>
    <div class="hero-right">
      <div class="hero-price-label">BTC / USDT · LIVE</div>
      <div class="hero-price">${live_price:,.2f}</div>
      <div class="hero-gap">Live / model close&nbsp; <b>{live_gap:+.3%}</b></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# -----------------------------
# KPI cards
# -----------------------------
kpis = [
    ("BTC / USDT","₿",f"${live_price:,.2f}",f"{data_status} · live","btc"),
    ("Prévision +1H","◈",f"${preds[-1]:,.2f}",f"{pct_change:+.2%} vs clôture modèle","green" if pct_change>=0 else "red"),
    ("Scénario",signal_icon,f"{decision_state}",decision_reason,signal_class),
    ("Qualité prévision","◉",f"{quality_info["quality"]:.0f}/100",f"Confiance modèle : {quality_info["label"]}","blue"),
    ("Incertitude +1H","≈",f"±{quality_info["uncertainty"]:.2%}",f"Dispersion ensemble","yellow"),
]
html='<div class="kpi-grid">'
for label,icon,value,note,cls in kpis:
    html += f'<div class="kpi"><div class="kpi-top"><div class="kpi-label">{label}</div><div class="kpi-icon">{icon}</div></div><div class="kpi-value {cls}">{value}</div><div class="kpi-note">{note}</div></div>'
html+='</div>'
st.markdown(html, unsafe_allow_html=True)

# Signal / confidence-style visual (score, not probability)
st.markdown(f"""
<div class="signal">
  <div class="signal-icon {signal_class}">{signal_icon}</div>
  <div><div class="signal-name">{decision_state} <span style="color:#718096;font-size:.72rem">· qualité {quality_info["label"]}</span></div>
  <div class="signal-desc">{decision_reason} · projection +1h : {pct_change:+.2%} · zone d'incertitude : ±{quality_info["uncertainty"]:.2%}</div></div>
  <div class="score-box"><div class="score-num {signal_class}">{quality_info["quality"]:.0f}</div>
  <div class="score-track"><div class="score-fill" style="width:{score_pct}%"></div></div></div>
</div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="info-strip">
  <span class="info-item">📡 <b>{MARKET_SOURCE}</b> · {BINANCE_SYMBOL}</span>
  <span class="info-item">🕐 Bougie clôturée <b>{last_time.strftime("%d/%m %H:%M")}</b></span>
  <span class="info-item">⌚ Tunis <b>{now_tunis.strftime("%H:%M:%S")}</b></span>
  <span class="info-item">↔ Écart live/modèle <b>{live_gap:+.3%}</b></span>
  <span class="info-item">🧪 Mode <b>Simulation</b></span>
</div>""", unsafe_allow_html=True)

# -----------------------------
# Main tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊  MARKET OVERVIEW", "🧪  WALK-FORWARD", "🤖  ML ANALYTICS"])

with tab1:
    st.markdown('<div class="panel"><div class="panel-head"><div><div class="panel-title">Prix BTC & projection ML</div><div class="panel-sub">80 dernières bougies · tendance · zones techniques · projection 12 horizons</div></div><div class="badge">5 MIN DATA</div></div>', unsafe_allow_html=True)
    st.plotly_chart(current_chart(df, preds, future_times, supports, resistances), use_container_width=True, config={"displaylogo":False,"modeBarButtonsToRemove":["lasso2d","select2d"]})

    st.markdown('<div class="panel"><div class="panel-head"><div><div class="panel-title">🛡️ Zone de prévision & incertitude</div><div class="panel-sub">Dispersion des arbres Random Forest — indicateur de fiabilité, pas une probabilité de gain</div></div><div class="badge">UNCERTAINTY</div></div>', unsafe_allow_html=True)
    band_fig = go.Figure()
    band_fig.add_trace(go.Scatter(
        x=future_times + future_times[::-1],
        y=list(p90) + list(p10[::-1]),
        fill="toself",
        fillcolor="rgba(100,116,139,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="Zone d'incertitude"
    ))
    band_fig.add_trace(go.Scatter(
        x=future_times, y=preds, mode="lines+markers",
        name="Prévision centrale",
        line=dict(width=3)
    ))
    band_fig.update_layout(
        height=360, template="plotly_dark", margin=dict(l=10,r=10,t=20,b=10),
        xaxis_title="Horizon", yaxis_title="BTC / USDT",
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(band_fig, use_container_width=True, config={"displaylogo":False})
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    a,b=st.columns([1.75,1])
    with a:
        st.markdown('<div class="panel"><div class="panel-head"><div><div class="panel-title">🔮 Projection multi-horizon</div><div class="panel-sub">Prix estimés à partir de la dernière bougie clôturée</div></div><div class="badge">12 HORIZONS</div></div>', unsafe_allow_html=True)
        pred_table=pd.DataFrame({
            "Horizon":[f"{5*(i+1)} min" for i in range(HORIZONS)],
            "Heure":[t.strftime("%H:%M:%S") for t in future_times],
            "Prévision":[round(float(p),2) for p in preds],
            "Borne basse":[round(float(p),2) for p in p10],
            "Borne haute":[round(float(p),2) for p in p90],
            "Variation":[f"{((p-model_reference_price)/model_reference_price):+.2%}" for p in preds],
        })
        st.dataframe(pred_table,use_container_width=True,hide_index=True,column_config={"Prévision":st.column_config.NumberColumn(format="$%.2f"),"Borne basse":st.column_config.NumberColumn(format="$%.2f"),"Borne haute":st.column_config.NumberColumn(format="$%.2f")})
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-head"><div><div class="panel-title">🎯 Zones techniques</div><div class="panel-sub">Niveaux détectés sur l’historique récent</div></div></div>', unsafe_allow_html=True)
        if supports:
            st.markdown("**Support**")
            for x in reversed(supports[-3:]): st.markdown(f"`$ {x:,.2f}`")
        else: st.caption("Aucun support détecté.")
        st.markdown("**Résistance**")
        if resistances:
            for x in reversed(resistances[-3:]): st.markdown(f"`$ {x:,.2f}`")
        else: st.caption("Aucune résistance détectée.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-strip"><span>ℹ️ Le modèle utilise la dernière bougie 5 minutes <b>clôturée</b>.</span><span>Le prix live est affiché séparément et n’est pas injecté dans les features de cette prédiction.</span></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-strip">
      <span>🧠 <b>Qualité modèle :</b> {quality_info["quality"]:.0f}/100 ({quality_info["label"]})</span>
      <span>📐 <b>Incertitude :</b> ±{quality_info["uncertainty"]:.2%}</span>
      <span>🧭 <b>Direction :</b> {quality_info["direction"]}</span>
      <span>🔎 <b>Consensus :</b> {quality_info["consensus"]:.0%}</span>
    </div>
    <div class="panel" style="margin-top:12px">
      <div class="panel-head">
        <div><div class="panel-title">🧭 Aide à l'interprétation</div>
        <div class="panel-sub">Règle pédagogique : une projection n'est retenue comme scénario que si elle dépasse raisonnablement sa zone d'incertitude.</div></div>
        <div class="badge">{decision_state}</div>
      </div>
      <div style="color:#9aa8ba;font-size:.84rem;line-height:1.7">
        <b>Lecture :</b> {decision_reason}. Le score de qualité mesure l'accord interne du modèle et le contexte technique ; 
        <b>ce n'est pas une probabilité de réussite</b>. Plus la zone d'incertitude est large, plus la projection doit être considérée avec prudence.
      </div>
    </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="panel"><div class="panel-head"><div><div class="panel-title">Walk-Forward Backtest</div><div class="panel-sub">Réentraînement chronologique · simulation historique · comparaison Buy & Hold</div></div><div class="badge">NO LIVE ORDERS</div></div>', unsafe_allow_html=True)
    with st.expander("⚙️ Paramètres du test", expanded=True):
        b1,b2,b3,b4=st.columns(4)
        train_size=b1.number_input("Taille entraînement",min_value=500,max_value=1500,value=900,step=100)
        test_size=b2.number_input("Taille test",min_value=200,max_value=1000,value=600,step=100)
        retrain_every=b3.number_input("Réentraînement / N bars",min_value=5,max_value=60,value=12,step=1)
        threshold=b4.number_input("Seuil signal",min_value=0.001,max_value=0.02,value=DEFAULT_THRESHOLD,step=0.001,format="%.3f")
        fee=st.number_input("Frais simulés / changement de position",min_value=0.0,max_value=0.01,value=DEFAULT_FEE,step=0.0001,format="%.4f")
    st.markdown('<div class="info-strip"><span>⚠️ Résultats historiques uniquement.</span><span>Ils ne constituent pas une garantie de performance future.</span></div>', unsafe_allow_html=True)
    run=st.button("▶  Lancer le Walk-Forward",type="primary",use_container_width=True)
    if run:
        with st.spinner("Analyse en cours — entraînements ML successifs..."):
            feature_values=df[FEATURES].values.astype(float); close_values=df["Close"].values.astype(float); index_values=df.index.astype(str).tolist()
            data_hash=(len(df),str(df.index[0]),str(df.index[-1]),float(df["Close"].iloc[-1]))
            result=run_walk_forward(data_hash,close_values.tolist(),feature_values.tolist(),index_values,int(train_size),int(test_size),int(retrain_every),float(threshold),float(fee))
        if result is None: st.error("Pas assez de données pour ces paramètres.")
        else: st.session_state["wf_result"]=result
    st.markdown('</div>', unsafe_allow_html=True)

    result=st.session_state.get("wf_result")
    if result is not None:
        initial_capital=10000.0; final_capital=initial_capital*result["final_multiple"]; bh_final=initial_capital*result["bh_multiple"]
        st.markdown('<div class="section-title">Résultats de simulation</div>',unsafe_allow_html=True)
        r1,r2,r3,r4,r5,r6=st.columns(6)
        r1.metric("Capital final",f"${final_capital:,.2f}"); r2.metric("Rendement",f"{result['final_multiple']-1:+.2%}"); r3.metric("Buy & Hold",f"{result['bh_multiple']-1:+.2%}"); r4.metric("Trades",f"{result['trades']}"); r5.metric("Win rate",f"{result['win_rate']:.2%}"); r6.metric("Max DD",f"{result['max_drawdown']:.2%}")
        c1,c2=st.columns([1.55,1])
        with c1: st.plotly_chart(backtest_chart(result),use_container_width=True)
        with c2: st.plotly_chart(drawdown_chart(result),use_container_width=True)
        st.markdown('<div class="panel"><div class="panel-title">Comparaison stratégie / Buy & Hold</div><div class="panel-sub">Lecture synthétique des métriques simulées</div>',unsafe_allow_html=True)
        comp=pd.DataFrame({"Indicateur":["Capital final","Rendement","Nombre de trades","Win rate","Sharpe","Max drawdown"],"Walk-Forward":[f"${final_capital:,.2f}",f"{result['final_multiple']-1:+.2%}",result["trades"],f"{result['win_rate']:.2%}",f"{result['sharpe']:.3f}",f"{result['max_drawdown']:.2%}"],"Buy & Hold":[f"${bh_final:,.2f}",f"{result['bh_multiple']-1:+.2%}","—","—","—","—"]})
        st.dataframe(comp,use_container_width=True,hide_index=True)
        st.markdown('</div>',unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="panel"><div class="panel-head"><div><div class="panel-title">🤖 ML Analytics</div><div class="panel-sub">Erreurs chronologiques · précision directionnelle · comportement par horizon</div></div><div class="badge">MODEL EVALUATION</div></div>',unsafe_allow_html=True)
    if "wf_result" not in st.session_state:
        st.info("Lance le Walk-Forward pour calculer les métriques ML.")
    else:
        result=st.session_state["wf_result"]
        e1,e2,e3=st.columns(3); e1.metric("MAE +1h",f"${result['mae']:,.2f}"); e2.metric("RMSE +1h",f"${result['rmse']:,.2f}"); e3.metric("Direction accuracy",f"{result['direction_acc']:.2%}")
        hm=result["horizon_metrics"].copy(); hm["MAE"]=hm["MAE"].round(2); hm["RMSE"]=hm["RMSE"].round(2); hm["Direction accuracy"]=(hm["Direction accuracy"]*100).round(2).astype(str)+"%"
        st.markdown('<div class="section-title">Performance par horizon</div>',unsafe_allow_html=True)
        st.dataframe(hm,use_container_width=True,hide_index=True)
        st.markdown('<div class="section-title">Prix prédit vs prix réel à +1h</div>',unsafe_allow_html=True)
        fig_eval=go.Figure(); fig_eval.add_trace(go.Scatter(x=result["index"],y=result["actual"],mode="lines",name="Réel +1h",line=dict(width=2))); fig_eval.add_trace(go.Scatter(x=result["index"],y=result["pred"],mode="lines",name="Prédit +1h",line=dict(width=2,dash="dot")))
        fig_eval.update_layout(height=470,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(13,17,23,.55)",margin=dict(l=10,r=10,t=25,b=10),hovermode="x unified",legend=dict(orientation="h",y=1.02,x=0))
        fig_eval.update_xaxes(showgrid=True,gridcolor="rgba(148,163,184,.08)",zeroline=False); fig_eval.update_yaxes(showgrid=True,gridcolor="rgba(148,163,184,.08)",zeroline=False)
        st.plotly_chart(fig_eval,use_container_width=True)
    st.markdown('</div>',unsafe_allow_html=True)

with st.expander("ℹ️ Détails techniques du modèle"):
    a,b,c=st.columns(3); a.write(f"**Type :** `{type(model).__name__}`"); a.write(f"**Features :** `{getattr(model,'n_features_in_','N/A')}`"); b.write(f"**Sorties :** `{len(getattr(model,'estimators_',[]))}`"); c.write(f"**Source :** `{MARKET_SOURCE} — {BINANCE_SYMBOL}`")
    if hasattr(model,"estimator"): c.write(f"**Random Forest :** {model.estimator.n_estimators} arbres · depth={model.estimator.max_depth}")
    st.write("**Features utilisées :**",FEATURES)

st.markdown('<div class="footer">BTC Intelligence V3.1 · Binance public market data · Dashboard éducatif et simulation historique · Aucun ordre réel exécuté.</div>',unsafe_allow_html=True)


# ============================================================
# FORECAST RELIABILITY PANEL
# ============================================================
st.markdown("### 📏 Fiabilité historique de la prévision")

if reliability is None:
    st.info(
        "Calibration historique indisponible sur cet échantillon. "
        "Les bornes affichées reposent alors sur la dispersion interne du modèle."
    )
else:
    rel_df, rel_overall = reliability
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Précision directionnelle", f"{rel_overall['Direction']:.1f}%")
    r2.metric("Erreur absolue moyenne", f"{rel_overall['MAE_pct']:.2f}%")
    r3.metric("Observations testées", f"{rel_overall['Observations']:,}")
    r4.metric("Fiabilité", reliability_label_text)

    st.caption(
        "Mesure calculée hors échantillon par validation walk-forward. "
        "Elle décrit le comportement historique du modèle et ne constitue pas une probabilité de gain."
    )

    display_rel = rel_df.copy()
    display_rel["Horizon"] = display_rel["Horizon"].map(lambda x: f"+{int(x)*5} min")
    display_rel["MAE_pct"] = display_rel["MAE_pct"].map(lambda x: f"{x:.2f}%")
    display_rel["Direction"] = display_rel["Direction"].map(lambda x: f"{x:.1f}%")
    display_rel["MAE"] = display_rel["MAE"].map(lambda x: f"${x:,.2f}")
    display_rel["RMSE"] = display_rel["RMSE"].map(lambda x: f"${x:,.2f}")
    display_rel = display_rel[["Horizon", "MAE", "RMSE", "MAE_pct", "Direction", "N"]]
    display_rel.columns = [
        "Horizon", "MAE", "RMSE", "Erreur %", "Direction correcte", "N"
    ]
    st.dataframe(display_rel, use_container_width=True, hide_index=True)

    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(
        x=rel_df["Horizon"] * 5,
        y=rel_df["Direction"],
        mode="lines+markers",
        name="Direction correcte (%)",
    ))
    fig_rel.add_hline(y=50, line_dash="dash", annotation_text="Hasard 50%")
    fig_rel.update_layout(
        height=320,
        template="plotly_dark",
        xaxis_title="Horizon (minutes)",
        yaxis_title="Direction correcte (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_rel, use_container_width=True)

    st.warning(
        "⚠️ Les intervalles V3.3 sont des intervalles empiriques basés sur les erreurs "
        "hors échantillon. Ils ne sont pas des intervalles de confiance garantis."
    )

