import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Setup
st.set_page_config(page_title="Delhi AQI Dashboard", layout="wide")
st.title("🌫️ Delhi AQI Dashboard (PM2.5 + Meteorological Analysis)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed.csv", parse_dates=["Timestamp"])
    df.set_index("Timestamp", inplace=True)
    loc_df = pd.read_csv("location.csv")
    loc_df.columns = loc_df.columns.str.lower()
    return df, loc_df

df, location_df = load_data()

stations = sorted({col.split('_')[0] for col in df.columns if '_' in col})
meteo_suffixes = ['AT (°C)', 'RH (%)', 'WS (m/s)', 'WD (deg)', 'BP (mmHg)', 'RF (mm)', 'SR (W/mt2)']

# Sidebar
view = st.sidebar.radio("📌 Select View", ["📈 Time Series", "📊 Station Clusters", "🗺️ Station Map"])

# =============================
# View 1: Time Series Analysis
# =============================
if view == "📈 Time Series":
    st.subheader("📈 Daily or Monthly Average of Selected Parameter")

    params = sorted({col.split('_', 1)[1] for col in df.columns if '_' in col and not col.endswith(('.1', 'hour_weight', 'month_weight', 'of_week'))})
    param = st.selectbox("Select Parameter", params)
    freq = st.radio("Resample By", ['D', 'M'], horizontal=True)

    resampled = df.resample(freq).mean()
    param_cols = [col for col in resampled.columns if col.endswith(param)]
    param_avg = resampled[param_cols].mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(param_avg.index, param_avg, color='blue', label='Avg')
    ax.axvspan(pd.to_datetime('2018-01-01'), pd.to_datetime('2019-12-31'), color='blue', alpha=0.1, label='Pre-COVID')
    ax.axvspan(pd.to_datetime('2020-01-01'), pd.to_datetime('2021-12-31'), color='orange', alpha=0.1, label='During COVID')
    ax.axvspan(pd.to_datetime('2022-01-01'), param_avg.index.max(), color='green', alpha=0.1, label='Post-COVID')
    ax.set_title(f"{freq}-Average of {param}")
    ax.legend()
    st.pyplot(fig)

# =============================
# View 2: Station Clustering
# =============================
elif view == "📊 Station Clusters":
    st.subheader("🔗 Station Clustering by PM2.5-Meteorology Correlation")

    corr_data = {}
    for station in stations:
        pm_col = f"{station}_PM2.5 (µg/m³)"
        if pm_col not in df.columns:
            continue
        vec = []
        for suffix in meteo_suffixes:
            met_col = f"{station}_{suffix}"
            if met_col in df.columns:
                temp = df[[pm_col, met_col]].dropna()
                corr = temp[pm_col].corr(temp[met_col]) if len(temp) > 1 else 0
                vec.append(corr)
            else:
                vec.append(0)
        corr_data[station] = vec

    corr_df = pd.DataFrame.from_dict(corr_data, orient='index', columns=meteo_suffixes).fillna(0)
    corr_df = corr_df.loc[:, corr_df.std() > 0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(corr_df)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    corr_df['Cluster'] = clusters
    corr_df['Station'] = corr_df.index

    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=corr_df['Cluster'].astype(str),
        hover_name=corr_df['Station'],
        labels={'x': 'PCA 1', 'y': 'PCA 2'},
        title="PCA + KMeans Clustering of Stations"
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================
# View 3: Map of Clusters
# =============================
elif view == "🗺️ Station Map":
    st.subheader("🗺️ Station Cluster Map of Delhi")

    # Generate fake cluster assignments if not from above
    station_df = pd.DataFrame({'main_station': stations})
    station_df['cluster'] = station_df.index % 3  # placeholder if real clusters not passed

    merged = pd.merge(station_df, location_df, how='left', left_on='main_station', right_on='location')
    merged = merged.dropna(subset=['latitudes', 'longitudes'])

    fig = px.scatter_mapbox(
        merged,
        lat="latitudes",
        lon="longitudes",
        color="cluster",
        text="main_station",
        zoom=9,
        height=700,
        mapbox_style="open-street-map"
    )
    fig.update_layout(title="Delhi AQI Station Clusters")
    st.plotly_chart(fig, use_container_width=True)
