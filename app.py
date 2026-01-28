import pickle
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from model import MLPRegressor

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Saudi Weather Dashboard",
    page_icon="📊",
    layout="wide",
)

# -----------------------
# Load data (bundled, no upload)
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/saudi_weather_sample.csv")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df

# -----------------------
# Load model assets
# -----------------------
@st.cache_resource
def load_model_assets():
    with open("assets/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("assets/ohe_columns.pkl", "rb") as f:
        ohe_cols = pickle.load(f)

    model = MLPRegressor(in_features=len(ohe_cols))
    state = torch.load("weights/mlp_weather_state_dict.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, scaler, ohe_cols

def preprocess_row_for_model(df_row: pd.DataFrame, ohe_cols: list, scaler):
    row_ohe = pd.get_dummies(
        df_row,
        columns=["station_name", "city", "Season_name"],
        drop_first=False
    )
    for col in ohe_cols:
        if col not in row_ohe.columns:
            row_ohe[col] = 0
    row_ohe = row_ohe[ohe_cols]
    X = scaler.transform(row_ohe).astype(np.float32)
    return X

def predict_temp(model, X_np: np.ndarray) -> float:
    x_t = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x_t).cpu().numpy().ravel()[0]
    return float(pred)

# -----------------------
# App
# -----------------------
st.title("📊 Saudi Weather Dashboard + Temperature Prediction")
st.caption("Simple dashboard (from bundled data) with a PyTorch MLP temperature predictor below.")

# Load data
try:
    df = load_data()
except Exception:
    st.error("Could not load dataset. Make sure data/saudi_weather_sample.csv exists in the repo.")
    st.stop()

# ==============
# DASHBOARD (simple)
# ==============
st.subheader("Dashboard")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]}")
c3.metric("Cities", f"{df['city'].nunique():,}" if "city" in df.columns else "—")
c4.metric("Stations", f"{df['station_name'].nunique():,}" if "station_name" in df.columns else "—")

# Filters (light)
filt = df.copy()
colA, colB, colC, colD = st.columns([2, 2, 2, 2])

if "city" in filt.columns:
    city_options = sorted(filt["city"].dropna().unique().tolist())
    sel_city = colA.selectbox("City", ["All"] + city_options)
    if sel_city != "All":
        filt = filt[filt["city"] == sel_city]

if "Season_name" in filt.columns:
    season_options = sorted(filt["Season_name"].dropna().unique().tolist())
    sel_season = colB.selectbox("Season", ["All"] + season_options)
    if sel_season != "All":
        filt = filt[filt["Season_name"] == sel_season]

if "month" in filt.columns:
    mmin = int(pd.to_numeric(filt["month"], errors="coerce").min()) if filt["month"].notna().any() else 1
    mmax = int(pd.to_numeric(filt["month"], errors="coerce").max()) if filt["month"].notna().any() else 12
    sel_month = colC.slider("Month", 1, 12, min(max(mmin, 1), 12))
    # If month column exists, filter by selected month
    filt["month"] = pd.to_numeric(filt["month"], errors="coerce")
    filt = filt[filt["month"] == sel_month]

# Optional datetime range
if "datetime" in filt.columns and pd.api.types.is_datetime64_any_dtype(filt["datetime"]):
    dtmin = filt["datetime"].min()
    dtmax = filt["datetime"].max()
    if pd.notna(dtmin) and pd.notna(dtmax):
        start, end = colD.date_input("Date range", value=(dtmin.date(), dtmax.date()))
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filt = filt[(filt["datetime"] >= start_dt) & (filt["datetime"] <= end_dt)]
else:
    colD.write("")

# Charts (simple)
ch1, ch2 = st.columns(2)

with ch1:
    st.write("Air temperature distribution")
    if "air_temperature" in filt.columns:
        vals = pd.to_numeric(filt["air_temperature"], errors="coerce").dropna().values
        if len(vals) > 0:
            fig, ax = plt.subplots()
            ax.hist(vals, bins=30)
            ax.set_xlabel("Air temperature (°C)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("No valid air_temperature values after filtering.")
    else:
        st.info("Column not found: air_temperature")

with ch2:
    st.write("Average temperature by hour")
    if "air_temperature" in filt.columns and "hour" in filt.columns:
        tmp = filt.copy()
        tmp["air_temperature"] = pd.to_numeric(tmp["air_temperature"], errors="coerce")
        tmp["hour"] = pd.to_numeric(tmp["hour"], errors="coerce")
        tmp = tmp.dropna(subset=["air_temperature", "hour"])
        if len(tmp) > 0:
            grp = tmp.groupby("hour")["air_temperature"].mean().sort_index()
            fig, ax = plt.subplots()
            ax.plot(grp.index.values, grp.values)
            ax.set_xlabel("Hour")
            ax.set_ylabel("Avg air temperature (°C)")
            st.pyplot(fig)
        else:
            st.info("Not enough valid data to plot.")
    else:
        st.info("Required columns not found: air_temperature and/or hour")

with st.expander("Preview data"):
    st.dataframe(filt.head(200), use_container_width=True)

st.divider()

# ==============
# MODEL (below dashboard)
# ==============
st.subheader("Temperature Prediction (PyTorch MLP)")

try:
    model, scaler, ohe_cols = load_model_assets()
except Exception:
    st.error("Could not load model assets. Check assets/ and weights/ files.")
    st.stop()

# Use dataset values to reduce user typing
col1, col2, col3 = st.columns(3)

if "station_name" in df.columns:
    station_options = sorted(df["station_name"].dropna().unique().tolist())
    station_name = col1.selectbox("Station name", station_options[:300] if station_options else ["ABHA"])
else:
    station_name = col1.text_input("Station name", value="ABHA")

if "city" in df.columns:
    city_options = sorted(df["city"].dropna().unique().tolist())
    city = col2.selectbox("City", city_options[:300] if city_options else ["ABHA"])
else:
    city = col2.text_input("City", value="ABHA")

if "Season_name" in df.columns:
    season_options = sorted(df["Season_name"].dropna().unique().tolist())
    season = col3.selectbox("Season", season_options if season_options else ["Winter", "Spring", "Summer", "Autumn"])
else:
    season = col3.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"], index=2)

colA, colB, colC, colD = st.columns(4)
month = colA.slider("Month", 1, 12, 8)
hour = colB.slider("Hour", 0, 23, 21)
dew = colC.number_input("Dew point (°C)", value=16.0, step=0.5)
wind = colD.number_input("Wind speed", value=1.2, step=0.1)

visibility = st.number_input("Visibility distance", value=10000.0, step=100.0)

with st.expander("Advanced (optional)"):
    dayofweek = st.slider("Day of week (0=Mon ... 6=Sun)", 0, 6, 0)
    day = st.slider("Day of month", 1, 31, 1)
    is_weekend = st.selectbox("Is weekend?", [0, 1], index=0)

if st.button("Predict"):
    row = pd.DataFrame([{
        "station_name": station_name,
        "city": city,
        "Season_name": season,
        "month": int(month),
        "hour": int(hour),
        "air_temperature_dew_point": float(dew),
        "wind_speed_rate": float(wind),
        "visibility_distance": float(visibility),
        "dayofweek": int(dayofweek),
        "day": int(day),
        "is_weekend": int(is_weekend),
    }])

    try:
        X = preprocess_row_for_model(row, ohe_cols=ohe_cols, scaler=scaler)
        pred = predict_temp(model, X)
        st.success(f"Predicted air temperature: **{pred:.2f} °C**")
    except Exception:
        st.error("Prediction failed. Ensure scaler.pkl and ohe_columns.pkl match the training pipeline.")
