import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Set up page config
st.set_page_config(page_title="Delhi AQI Dashboard", layout="wide")
st.title("🌫️ Delhi AQI Dashboard (PM2.5 + Meteorological Analysis)")

# Load data once, cache for performance
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed.csv", parse_dates=["Timestamp"])
    df.set_index("Timestamp", inplace=True)
    loc_df = pd.read_csv("location.csv")
    loc_df.columns = loc_df.columns.str.lower()
    return df, loc_df

df, location_df = load_data()

# Get list of stations from df columns (assume format: station_parameter)
stations = sorted({col.split('_')[0] for col in df.columns if '_' in col and "PM2.5" in col})

# Sidebar: Select view
view = st.sidebar.radio("📌 Select View", ["📈 Time Series", "🗺️ Station Map", "⚙️ Modeling"])

# Time Series View
if view == "📈 Time Series":
    st.subheader("📈 Daily or Monthly Average of Selected Parameter")
    params = sorted({col.split('_', 1)[1] for col in df.columns if '_' in col})
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

# Map View
elif view == "🗺️ Station Map":
    st.subheader("🗺️ Station Map of Delhi AQI Stations")
    # Merge stations with locations
    station_df = pd.DataFrame({'main_station': stations})
    merged = pd.merge(station_df, location_df, how='left', left_on='main_station', right_on='location')
    merged = merged.dropna(subset=['latitudes', 'longitudes'])

    fig = px.scatter_mapbox(
        merged,
        lat="latitudes",
        lon="longitudes",
        text="main_station",
        zoom=9,
        height=700,
        mapbox_style="open-street-map"
    )
    fig.update_layout(title="Delhi AQI Stations")
    st.plotly_chart(fig, use_container_width=True)

# Modeling View
elif view == "⚙️ Modeling":
    st.subheader("⚙️ Station-wise PM2.5 Prediction Using Random Forest")

    selected_station = st.selectbox("Select Station for Modeling", stations)

    if selected_station:
        st.markdown(f"### Modeling results for **{selected_station}**")

        target_col = f"{selected_station}_PM2.5 (µg/m³)"
        if target_col not in df.columns:
            st.warning(f"No PM2.5 data available for station '{selected_station}'.")
        else:
            # Prepare dataset for modeling
            df_station = df.reset_index()

            # Select features from this station except PM10 or target itself
            station_features = [col for col in df_station.columns if col.startswith(selected_station + "_") 
                                and col != target_col and "PM10" not in col]

            # Temporal features if available
            temporal_features = ['month', 'day_of_week', 'hour', 'PM2.5_month_weight', 'PM2.5_hour_weight']
            available_temporal = [feat for feat in temporal_features if feat in df_station.columns]

            selected_features = station_features + available_temporal

            modeling_cols = ['Timestamp', target_col] + selected_features
            modeling_df = df_station[modeling_cols].dropna().sort_values('Timestamp')

            if modeling_df.empty:
                st.warning("Not enough data after filtering to train model.")
            else:
                X = modeling_df.drop(columns=['Timestamp', target_col])
                y = modeling_df[target_col]

                # Train-test split by date
                train_mask = modeling_df['Timestamp'] < '2024-01-01'
                test_mask = modeling_df['Timestamp'] >= '2024-01-01'

                X_train, y_train = X[train_mask], y[train_mask]
                X_test, y_test = X[test_mask], y[test_mask]

                if X_train.empty or X_test.empty:
                    st.warning("Insufficient train or test data after splitting.")
                else:
                    # Initial RF training
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X_train, y_train)

                    y_pred_train = rf.predict(X_train)
                    y_pred_test = rf.predict(X_test)

                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)

                    st.markdown(f"**Initial Train R² Score:** {train_r2:.3f}")
                    st.markdown(f"**Initial Test R² Score:** {test_r2:.3f}")

                    # Feature importance
                    importances = rf.feature_importances_
                    feat_imp_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=feat_imp_df.head(20), x='Importance', y='Feature', ax=ax)
                    ax.set_title(f"Top 20 Feature Importances (Initial) for {selected_station}")
                    st.pyplot(fig)

                    # Filter features with importance > 0
                    selected_feats = feat_imp_df[feat_imp_df['Importance'] > 0]['Feature'].tolist()

                    st.markdown(f"Selected {len(selected_feats)} features with importance > 0.")

                    if len(selected_feats) == 0:
                        st.warning("No features have importance > 0 after initial training.")
                    else:
                        # Retrain RF with selected features
                        X_train_sel = X_train[selected_feats]
                        X_test_sel = X_test[selected_feats]

                        rf_sel = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_sel.fit(X_train_sel, y_train)

                        y_pred_train_sel = rf_sel.predict(X_train_sel)
                        y_pred_test_sel = rf_sel.predict(X_test_sel)

                        train_r2_sel = r2_score(y_train, y_pred_train_sel)
                        test_r2_sel = r2_score(y_test, y_pred_test_sel)

                        st.markdown(f"**Retrained Train R² Score:** {train_r2_sel:.3f}")
                        st.markdown(f"**Retrained Test R² Score:** {test_r2_sel:.3f}")

                        # New feature importance plot
                        importances_sel = rf_sel.feature_importances_
                        feat_imp_df_sel = pd.DataFrame({
                            'Feature': selected_feats,
                            'Importance': importances_sel
                        }).sort_values(by='Importance', ascending=False)

                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=feat_imp_df_sel.head(20), x='Importance', y='Feature', ax=ax2)
                        ax2.set_title(f"Top 20 Feature Importances (After Selection) for {selected_station}")
                        st.pyplot(fig2)

                        # Plot Actual vs Predicted for Train
                        fig3, ax3 = plt.subplots(figsize=(12, 5))
                        ax3.plot(y_train.values, label='Actual Train', alpha=0.7)
                        ax3.plot(y_pred_train_sel, label='Predicted Train', alpha=0.7)
                        ax3.set_title(f"Actual vs Predicted PM2.5 on Train Data ({selected_station})")
                        ax3.legend()
                        st.pyplot(fig3)

                        # Plot Actual vs Predicted for Test
                        fig4, ax4 = plt.subplots(figsize=(12, 5))
                        ax4.plot(y_test.values, label='Actual Test', alpha=0.7)
                        ax4.plot(y_pred_test_sel, label='Predicted Test', alpha=0.7)
                        ax4.set_title(f"Actual vs Predicted PM2.5 on Test Data ({selected_station})")
                        ax4.legend()
                        st.pyplot(fig4)
