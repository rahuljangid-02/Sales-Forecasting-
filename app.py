"""
End-to-End Sales Forecasting & Demand Intelligence System
Streamlit Dashboard (Task 7)

Run locally with:  streamlit run app.py
Deploy on Streamlit Community Cloud by pointing it at this file + requirements.txt + train.csv
in the same repo.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "train.csv"


# --------------------------------------------------------------------------
# Data loading & shared feature engineering
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter

    def get_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Spring"
        elif m in [6, 7, 8]:
            return "Summer"
        return "Fall"

    df["Season"] = df["Month"].apply(get_season)
    return df


@st.cache_data
def get_monthly_series(_df, category=None, region=None):
    """Monthly sales series, optionally filtered to one category or region,
    reindexed to a continuous monthly range (missing months filled with 0)."""
    mask = pd.Series(True, index=_df.index)
    if category:
        mask &= _df["Category"] == category
    if region:
        mask &= _df["Region"] == region
    s = _df[mask].set_index("Order Date").resample("MS")["Sales"].sum()
    full_range = pd.date_range(_df["Order Date"].min().replace(day=1),
                                _df["Order Date"].max().replace(day=1), freq="MS")
    return s.reindex(full_range, fill_value=0)


def build_features(series):
    d = series.reset_index()
    d.columns = ["Date", "Sales"]
    d["Month"] = d["Date"].dt.month
    d["Quarter"] = d["Date"].dt.quarter
    d["Season"] = d["Month"].apply(
        lambda m: "Winter" if m in [12, 1, 2] else "Spring" if m in [3, 4, 5]
        else "Summer" if m in [6, 7, 8] else "Fall"
    )
    d["SeasonCode"] = d["Season"].astype("category").cat.codes
    for lag in [1, 2, 3]:
        d[f"lag_{lag}"] = d["Sales"].shift(lag)
    d["rolling_mean_3"] = d["Sales"].shift(1).rolling(3).mean()
    return d.dropna().reset_index(drop=True)


FEATURE_COLS = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "Month", "Quarter", "SeasonCode"]


@st.cache_data
def evaluate_and_forecast(series, horizon):
    """Fit XGBoost (the best model from Task 3), evaluate on the last 3 held-out
    months, then produce a recursive forecast `horizon` months into the future
    using all available data."""
    d = build_features(series)

    # --- held-out evaluation (always on the last 3 months, for a stable metric) ---
    train, test = d.iloc[:-3], d.iloc[-3:]
    eval_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    eval_model.fit(train[FEATURE_COLS], train["Sales"])
    test_pred = eval_model.predict(test[FEATURE_COLS])
    mae = mean_absolute_error(test["Sales"], test_pred)
    rmse = np.sqrt(mean_squared_error(test["Sales"], test_pred))

    # --- refit on full history, forecast forward ---
    full_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    full_model.fit(d[FEATURE_COLS], d["Sales"])

    hist = d[["Date", "Sales"]].copy()
    season_lookup = d[["Month", "SeasonCode"]].drop_duplicates().set_index("Month")["SeasonCode"]
    future_dates = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    preds = []
    for fd in future_dates:
        l1, l2, l3 = hist["Sales"].iloc[-1], hist["Sales"].iloc[-2], hist["Sales"].iloc[-3]
        rm3 = hist["Sales"].iloc[-3:].mean()
        scode = season_lookup.get(fd.month, 0)
        Xf = pd.DataFrame([[l1, l2, l3, rm3, fd.month, fd.quarter, scode]], columns=FEATURE_COLS)
        p = full_model.predict(Xf)[0]
        preds.append(p)
        hist = pd.concat([hist, pd.DataFrame({"Date": [fd], "Sales": [p]})], ignore_index=True)

    forecast = pd.Series(preds, index=future_dates)
    return forecast, mae, rmse, test["Sales"], pd.Series(test_pred, index=test["Date"])


@st.cache_data
def detect_anomalies(_df):
    weekly = _df.set_index("Order Date").resample("W")["Sales"].sum().reset_index()
    weekly.columns = ["Week", "Sales"]

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly["is_anomaly_iso"] = iso.fit_predict(weekly[["Sales"]]) == -1

    roll_mean = weekly["Sales"].rolling(window=8, min_periods=4).mean()
    roll_std = weekly["Sales"].rolling(window=8, min_periods=4).std()
    weekly["zscore"] = (weekly["Sales"] - roll_mean) / roll_std
    weekly["is_anomaly_zscore"] = weekly["zscore"].abs() > 2

    weekly["is_anomaly"] = weekly["is_anomaly_iso"] | weekly["is_anomaly_zscore"]
    return weekly


@st.cache_data
def cluster_products(_df):
    d = _df.copy()
    d["YearMonth"] = d["Order Date"].dt.to_period("M")

    total_sales = d.groupby("Sub-Category")["Sales"].sum()
    avg_order_value = d.groupby("Sub-Category")["Sales"].mean()
    monthly_by_sub = d.groupby(["Sub-Category", "YearMonth"])["Sales"].sum().unstack(fill_value=0)
    volatility = monthly_by_sub.std(axis=1)
    yearly_by_sub = d.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack(fill_value=0)
    y_min, y_max = yearly_by_sub.columns.min(), yearly_by_sub.columns.max()
    growth_rate = (yearly_by_sub[y_max] - yearly_by_sub[y_min]) / yearly_by_sub[y_min].replace(0, np.nan) * 100

    features = pd.DataFrame({
        "TotalSales": total_sales,
        "GrowthRate": growth_rate,
        "Volatility": volatility,
        "AvgOrderValue": avg_order_value,
    }).dropna()

    X_scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    features["Cluster"] = kmeans.fit_predict(X_scaled)

    summary = features.groupby("Cluster")[["TotalSales", "GrowthRate", "Volatility"]].mean()
    medians = summary.median()

    def label(row):
        vol_high = row["Volatility"] > medians["Volatility"]
        sales_high = row["TotalSales"] > medians["TotalSales"]
        growth_high = row["GrowthRate"] > medians["GrowthRate"]
        if sales_high and not vol_high:
            return "High Volume, Stable Demand"
        elif not sales_high and vol_high:
            return "Low Volume, High Volatility"
        elif growth_high:
            return "Growing Demand"
        return "Declining Demand"

    summary["Label"] = summary.apply(label, axis=1)
    features = features.merge(summary[["Label"]], left_on="Cluster", right_index=True)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    features["PC1"], features["PC2"] = coords[:, 0], coords[:, 1]
    return features, pca.explained_variance_ratio_


# --------------------------------------------------------------------------
# App layout
# --------------------------------------------------------------------------
df = load_data()

st.sidebar.title("Sales Forecasting & Demand Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

# ---------------------------------------------------------------- Page 1 --
if page == "Sales Overview":
    st.title("📊 Sales Overview Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.multiselect("Filter by Region", sorted(df["Region"].unique()),
                                        default=sorted(df["Region"].unique()))
    with col2:
        category_filter = st.multiselect("Filter by Category", sorted(df["Category"].unique()),
                                          default=sorted(df["Category"].unique()))

    filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)]

    total_sales = filtered["Sales"].sum()
    total_orders = filtered["Order ID"].nunique()
    avg_order_value = filtered.groupby("Order ID")["Sales"].sum().mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sales", f"${total_sales:,.0f}")
    m2.metric("Total Orders", f"{total_orders:,}")
    m3.metric("Avg Order Value", f"${avg_order_value:,.2f}")

    st.subheader("Total Sales by Year")
    yearly = filtered.groupby("Year")["Sales"].sum().reset_index()
    fig_year = px.bar(yearly, x="Year", y="Sales", text_auto=".2s", color="Year")
    fig_year.update_layout(showlegend=False)
    st.plotly_chart(fig_year, use_container_width=True)

    st.subheader("Monthly Sales Trend")
    monthly = filtered.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
    fig_month = px.line(monthly, x="Order Date", y="Sales", markers=True)
    st.plotly_chart(fig_month, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Sales by Region")
        region_sales = filtered.groupby("Region")["Sales"].sum().reset_index()
        fig_region = px.pie(region_sales, names="Region", values="Sales", hole=0.4)
        st.plotly_chart(fig_region, use_container_width=True)
    with col4:
        st.subheader("Sales by Category")
        cat_sales = filtered.groupby("Category")["Sales"].sum().reset_index()
        fig_cat = px.pie(cat_sales, names="Category", values="Sales", hole=0.4)
        st.plotly_chart(fig_cat, use_container_width=True)

# ---------------------------------------------------------------- Page 2 --
elif page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.caption("Forecasts are generated with the XGBoost model, the top performer from the "
               "model comparison in the notebook (Task 3).")

    dimension = st.selectbox("Forecast by", ["Category", "Region"])
    if dimension == "Category":
        options = sorted(df["Category"].unique())
    else:
        options = sorted(df["Region"].unique())
    selection = st.selectbox(f"Select {dimension}", options)

    horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)

    if dimension == "Category":
        series = get_monthly_series(df, category=selection)
    else:
        series = get_monthly_series(df, region=selection)

    forecast, mae, rmse, test_actual, test_pred = evaluate_and_forecast(series, horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Actual",
                              mode="lines", line=dict(color="#4C72B0")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast",
                              mode="lines+markers", line=dict(color="#C44E52", dash="dash")))
    fig.update_layout(title=f"{selection} — {horizon}-Month Forecast", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast values")
    forecast_table = forecast.reset_index()
    forecast_table.columns = ["Month", "Forecasted Sales"]
    forecast_table["Forecasted Sales"] = forecast_table["Forecasted Sales"].map(lambda x: f"${x:,.2f}")
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    st.subheader("Model accuracy (last 3 actual months, held out)")
    c1, c2 = st.columns(2)
    c1.metric("MAE", f"${mae:,.2f}")
    c2.metric("RMSE", f"${rmse:,.2f}")

# ---------------------------------------------------------------- Page 3 --
elif page == "Anomaly Report":
    st.title("🚨 Anomaly Report")
    st.caption("Weekly sales flagged by Isolation Forest and/or a rolling Z-score (>2 std dev "
               "from an 8-week rolling mean).")

    weekly = detect_anomalies(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["Week"], y=weekly["Sales"], name="Weekly Sales",
                              mode="lines", line=dict(color="#4C72B0")))
    iso_pts = weekly[weekly["is_anomaly_iso"]]
    fig.add_trace(go.Scatter(x=iso_pts["Week"], y=iso_pts["Sales"], name="Isolation Forest anomaly",
                              mode="markers", marker=dict(color="red", symbol="x", size=10)))
    z_pts = weekly[weekly["is_anomaly_zscore"]]
    fig.add_trace(go.Scatter(x=z_pts["Week"], y=z_pts["Sales"], name="Z-score anomaly",
                              mode="markers", marker=dict(color="green", symbol="diamond", size=9)))
    fig.update_layout(title="Weekly Sales with Detected Anomalies", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected anomaly weeks")
    anomaly_table = weekly[weekly["is_anomaly"]][["Week", "Sales", "is_anomaly_iso", "is_anomaly_zscore"]].copy()
    anomaly_table.columns = ["Week", "Sales", "Flagged by Isolation Forest", "Flagged by Z-score"]
    anomaly_table["Sales"] = anomaly_table["Sales"].map(lambda x: f"${x:,.2f}")
    st.dataframe(anomaly_table.sort_values("Week"), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Page 4 --
elif page == "Product Demand Segments":
    st.title("🧩 Product Demand Segments")
    st.caption("K-Means clustering (k=4) on sub-category total sales, YoY growth rate, "
               "sales volatility, and average order value.")

    features, var_ratio = cluster_products(df)

    fig = px.scatter(
        features.reset_index(), x="PC1", y="PC2", color="Label", text="Sub-Category",
        hover_data=["TotalSales", "GrowthRate", "Volatility", "AvgOrderValue"],
        labels={"PC1": f"PC1 ({var_ratio[0]*100:.1f}% variance)",
                "PC2": f"PC2 ({var_ratio[1]*100:.1f}% variance)"},
    )
    fig.update_traces(textposition="top center", marker=dict(size=14))
    fig.update_layout(title="Product Sub-Category Demand Clusters (PCA-reduced)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-category → cluster assignment")
    table = features.reset_index()[["Sub-Category", "Label", "TotalSales", "GrowthRate",
                                     "Volatility", "AvgOrderValue"]].sort_values("Label")
    table["TotalSales"] = table["TotalSales"].map(lambda x: f"${x:,.0f}")
    table["GrowthRate"] = table["GrowthRate"].map(lambda x: f"{x:,.1f}%")
    table["Volatility"] = table["Volatility"].map(lambda x: f"${x:,.0f}")
    table["AvgOrderValue"] = table["AvgOrderValue"].map(lambda x: f"${x:,.2f}")
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Recommended stocking strategy")
    st.markdown("""
- **High Volume, Stable Demand** — standard reorder-point inventory, moderate safety stock.
- **Low Volume, High Volatility** — higher safety-stock buffer relative to sales volume; avoid large fixed purchase commitments.
- **Growing Demand** — increase reorder quantities ahead of trend; prioritize supplier capacity planning.
- **Declining Demand** — reduce standing inventory; consider phasing out persistently low-growth, low-volume SKUs.
""")
