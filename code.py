# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ======================================================
# 0. PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Enchanto – AI Inventory Dashboard",
    layout="wide",
    page_icon="📦"
)

# White–purple style via simple CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f3ff;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .kpi-box {
        border-radius: 10px;
        background-color: #faf5ff;
        border: 1px solid #e5e7eb;
        padding: 0.6rem 0.8rem;
        font-size: 0.8rem;
    }
    .kpi-label {
        color: #6b7280;
        font-size: 0.75rem;
    }
    .kpi-value {
        color: #4c1d95;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .kpi-note {
        color: #6b7280;
        font-size: 0.7rem;
    }
    .stock-banner {
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
    }
    .stock-high {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }
    .stock-medium {
        background: #fffbeb;
        border: 1px solid #facc15;
        color: #92400e;
    }
    .stock-low {
        background: #ecfdf3;
        border: 1px solid #bbf7d0;
        color: #166534;
    }
    .tag-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 9px;
        border-radius: 999px;
        background: #ede9fe;
        color: #4c1d95;
        font-size: 0.7rem;
        margin-bottom: 4px;
    }
    .tag-pill-icon {
        width: 16px;
        height: 16px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #6d28d9;
        color: white;
        font-size: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# 1. DATA LOADING
# ======================================================

# Adjust these paths for your GitHub/Streamlit project structure
# For Streamlit Cloud, put the Excel files in a "data" folder next to app.py
SALES_FILE = Path("enchanto_sales_simulated_2024.xlsx")
INVENTORY_TEMPLATE_FILE = Path("data/enchanto_inventory_solver_template.xlsx")  # optional, loaded if you want later


@st.cache_data(show_spinner=True)
def load_sales_data(sales_path: Path):
    """
    Load the multi-sheet Enchanto sales Excel and build daily demand table.
    """
    xls = pd.ExcelFile(sales_path)
    all_months = [pd.read_excel(sales_path, sheet_name=s) for s in xls.sheet_names]
    sales = pd.concat(all_months, ignore_index=True)

    sales["Order DateTime"] = pd.to_datetime(sales["Order DateTime"])
    sales["Order Date"] = pd.to_datetime(sales["Order Date"])

    # Clean festival flag
    if "Festival Season" in sales.columns:
        sales["Festival Season"] = sales["Festival Season"].fillna("No")
    else:
        sales["Festival Season"] = "No"

    # Daily demand per SKU/Region
    daily_demand = (
        sales
        .groupby(["Order Date", "SKU", "Region", "Category", "Product Name", "Festival Season"])
        ["Quantity Sold"]
        .sum()
        .reset_index()
        .rename(columns={
            "Order Date": "date",
            "Quantity Sold": "daily_demand"
        })
    )

    # time columns
    daily_demand["date"] = pd.to_datetime(daily_demand["date"])
    daily_demand["Year"] = daily_demand["date"].dt.year
    daily_demand["Month"] = daily_demand["date"].dt.month
    daily_demand["Month_Name"] = daily_demand["date"].dt.strftime("%b")

    # Smooth demand: 3-day moving average within each SKU-Region
    daily_demand = daily_demand.sort_values(["SKU", "Region", "date"]).reset_index(drop=True)
    daily_demand["demand_smooth"] = (
        daily_demand
        .groupby(["SKU", "Region"])["daily_demand"]
        .transform(lambda x: x.rolling(window=3, min_periods=1, center=True).mean())
    )

    return sales, daily_demand


sales, daily_demand = load_sales_data(SALES_FILE)

# ======================================================
# 2. HELPER FUNCTIONS
# ======================================================

def build_features_for_sku_region(daily_demand, sku, region, min_days=40):
    df = daily_demand[(daily_demand["SKU"] == sku) & (daily_demand["Region"] == region)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    history_len = len(df)

    if history_len < min_days:
        return None, None, None, None, history_len

    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["festival_flag"] = (df["Festival Season"] == "Yes").astype(int)

    df["lag_1"] = df["demand_smooth"].shift(1)
    df["lag_7"] = df["demand_smooth"].shift(7)
    df["rolling_7"] = df["demand_smooth"].shift(1).rolling(window=7).mean()

    df_model = df.dropna().reset_index(drop=True)

    feature_cols = [
        "day_of_week",
        "month",
        "is_weekend",
        "festival_flag",
        "lag_1",
        "lag_7",
        "rolling_7",
    ]

    X = df_model[feature_cols]
    y = df_model["demand_smooth"]
    return df_model, X, y, feature_cols, history_len


def forecast_next_n_days(model, df_history, n_days=30):
    """
    Roll-forward forecast using previous predictions as lags.
    df_history must have: date, daily_demand, demand_smooth, Festival Season.
    """
    history = df_history.copy().sort_values("date").reset_index(drop=True)
    last_date = history["date"].iloc[-1]

    preds = []
    for i in range(1, n_days + 1):
        next_date = last_date + pd.Timedelta(days=i)
        dow = next_date.weekday()
        month = next_date.month
        is_weekend = int(dow in [5, 6])
        # Simple festive flag: Oct–Dec
        festival_flag = 1 if month in [10, 11, 12] else 0

        lag_1 = history["demand_smooth"].iloc[-1]
        if len(history) >= 7:
            lag_7 = history["demand_smooth"].iloc[-7]
            rolling_7 = history["demand_smooth"].iloc[-7:].mean()
        else:
            lag_7 = lag_1
            rolling_7 = history["demand_smooth"].mean()

        x_future = pd.DataFrame([{
            "day_of_week": dow,
            "month": month,
            "is_weekend": is_weekend,
            "festival_flag": festival_flag,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_7": rolling_7
        }])

        y_future = model.predict(x_future)[0]
        preds.append({"date": next_date, "forecast_demand": y_future})

        new_row = {
            "date": next_date,
            "daily_demand": y_future,
            "demand_smooth": y_future,
            "Festival Season": "Yes" if festival_flag == 1 else "No"
        }
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(preds)


def get_current_stock(sales, sku, region):
    """
    Robust: latest non-NaN Stock After Sale; if missing, return 0.
    """
    df = (
        sales[(sales["SKU"] == sku) & (sales["Region"] == region)]
        .sort_values("Order DateTime")
    )
    if df.empty or "Stock After Sale" not in df.columns:
        return 0
    stock_series = df["Stock After Sale"].dropna()
    if stock_series.empty:
        return 0
    return int(round(stock_series.iloc[-1]))


def approx_z(service_level):
    table = {
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.97: 1.88,
        0.98: 2.05,
        0.99: 2.33,
    }
    # pick closest
    keys = list(table.keys())
    closest = min(keys, key=lambda k: abs(k - service_level))
    return table[closest]


def compute_lead_time_stats_from_forecast(forecast_df, lead_time_days=7):
    lt = forecast_df.head(lead_time_days)["forecast_demand"]
    mean_daily = lt.mean()
    total_mean = lt.sum()
    daily_std = lt.std(ddof=0)
    total_std = daily_std * np.sqrt(max(lead_time_days, 1))
    return mean_daily, total_mean, daily_std, total_std


def optimize_reorder_policy(
    forecast_df,
    current_stock,
    holding_cost_per_unit,
    stockout_cost_per_unit,
    lead_time_days=7,
):
    mean_daily, total_mean, daily_std, total_std = compute_lead_time_stats_from_forecast(
        forecast_df, lead_time_days
    )
    service_levels = [0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99]

    rows = []
    best_row = None
    for sl in service_levels:
        z = approx_z(sl)
        safety_stock = z * total_std
        reorder_level = total_mean + safety_stock
        holding_cost = holding_cost_per_unit * reorder_level
        expected_stockout_units = (1 - sl) * total_mean
        stockout_cost = stockout_cost_per_unit * expected_stockout_units
        total_cost = holding_cost + stockout_cost
        reorder_qty = max(0, reorder_level - current_stock)

        row = {
            "Service_Level": sl,
            "Safety_Stock": safety_stock,
            "Reorder_Level": reorder_level,
            "Holding_Cost": holding_cost,
            "Stockout_Cost": stockout_cost,
            "Total_Cost": total_cost,
            "Reorder_Qty": reorder_qty,
        }
        rows.append(row)
        if best_row is None or total_cost < best_row["Total_Cost"]:
            best_row = row

    result_df = pd.DataFrame(rows).sort_values("Total_Cost")
    return result_df, best_row, mean_daily, daily_std, total_mean


def optimize_logistics_distribution(
    sales,
    daily_demand,
    sku,
    inbound_units,
    lead_time_days=7,
):
    """
    Simple heuristic: estimate current stock & LT demand per region, then greedily
    allocate inbound units to regions with highest shortage.
    """
    regions = (
        daily_demand[daily_demand["SKU"] == sku]["Region"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    rows = []
    remaining = inbound_units

    for region in regions:
        rec_hist = daily_demand[(daily_demand["SKU"] == sku) & (daily_demand["Region"] == region)]
        if rec_hist.empty:
            continue
        # current stock: from sales table
        current_stock = get_current_stock(sales, sku, region)

        # approximate forecast LT demand using historical last 30 days average
        # (simpler than per-region model, but consistent with "AI-style" approach)
        rec_hist = rec_hist.sort_values("date")
        last_30 = rec_hist.tail(30)
        mean_daily = last_30["daily_demand"].mean()
        lt_total = mean_daily * lead_time_days

        shortage = max(0.0, lt_total - current_stock)
        rows.append({
            "Region": region,
            "Current_Stock": current_stock,
            "LT_Demand_7d": lt_total,
            "Shortage_Before": shortage,
            "Inbound_Allocated": 0.0,
        })

    # allocate inbound to largest shortage first
    rows = sorted(rows, key=lambda r: r["Shortage_Before"], reverse=True)
    for row in rows:
        if remaining <= 0:
            break
        need = row["Shortage_Before"]
        give = min(remaining, need)
        row["Inbound_Allocated"] = give
        remaining -= give

    # compute shortage/excess after allocation
    for row in rows:
        final_stock = row["Current_Stock"] + row["Inbound_Allocated"]
        shortage_after = max(0.0, row["LT_Demand_7d"] - final_stock)
        excess_after = max(0.0, final_stock - row["LT_Demand_7d"])
        row["Shortage_After"] = shortage_after
        row["Excess_After"] = excess_after

    # back to DF and region order
    df_dist = pd.DataFrame(rows)
    return df_dist


# ======================================================
# 3. UI – HEADER
# ======================================================
st.markdown(
    """
    <div class="tag-pill">
      <span class="tag-pill-icon">AI</span>
      Enchanto – Demand Forecasting & Inventory Optimization
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Enchanto Inventory Dashboard")

col_head1, col_head2 = st.columns([2, 1.8])
with col_head1:
    st.write(
        "56 SKUs • AI-driven demand forecasting • Reorder optimization • "
        "Logistics allocation across regions"
    )
with col_head2:
    st.markdown(
        """
        **Managerial Insight**  
        This dashboard demonstrates how **AI-based demand forecasting** can:
        - Reduce **stockouts** by anticipating demand at SKU–Region level  
        - Lower **working capital** via optimized safety stock and reorder levels  
        - Improve **delivery speed** by allocating inbound stock to the right regions  
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ======================================================
# 4. SIDEBAR / CONTROLS
# ======================================================
with st.sidebar:
    st.header("Scenario Selection")

    sku_options = (
        daily_demand[["SKU", "Product Name"]]
        .drop_duplicates()
        .sort_values("SKU")
    )
    sku_list = sku_options["SKU"].tolist()
    sku = st.selectbox(
        "Choose SKU",
        options=sku_list,
        format_func=lambda s: f"{s} – {sku_options.set_index('SKU').loc[s, 'Product Name']}",
    )

    regions_for_sku = (
        daily_demand[daily_demand["SKU"] == sku]["Region"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    region = st.selectbox("Region", options=regions_for_sku)

   # Current stock is read-only: taken from the latest "Stock After Sale" in the Excel
    computed_stock = get_current_stock(sales, sku, region)

    st.metric(
        "Current stock in this region (units)",
        value=int(computed_stock),
        help="Pulled from the latest 'Stock After Sale' in the transaction data."
    )

    # Use this fixed value in all downstream calculations
    current_stock = float(computed_stock)

    inbound_units = st.number_input(
        "Inbound stock for this SKU (total units across regions)",
        min_value=0.0,
        value=250.0,
        step=10.0,
        help="These units will be allocated to regions with the highest forecasted shortage."
    )

    st.markdown("**Cost assumptions** (driven by logistics):")
    transport_cost_per_unit = st.number_input(
        "Logistics cost per unit (₹)",
        min_value=1.0,
        value=50.0,
        step=1.0,
        help="Used only to derive holding cost. Default: ₹50 per unit."
    )
    holding_cost_per_unit = transport_cost_per_unit * 0.02
    stockout_cost_per_unit = holding_cost_per_unit * 8

    st.write(f"- Holding cost per unit = 2% of logistics cost → **₹{holding_cost_per_unit:.2f}**")
    st.write(f"- Stockout cost per unit = 8× holding cost → **₹{stockout_cost_per_unit:.2f}**")

    st.markdown("---")
    run_btn = st.button("▶ Run analysis")


if (not run_btn) and "first_run_done" not in st.session_state:
    st.session_state["first_run_done"] = True
    run_btn = True  # run once at start

if run_btn:
    # Filter data for selected SKU & region
    df_model, X, y, feature_cols, history_len = build_features_for_sku_region(
        daily_demand, sku, region, min_days=40
    )

    if df_model is None:
        st.warning(
            f"Not enough history for {sku} in {region} "
            f"(have {history_len} days, need at least 40). Try another region or SKU."
        )
    else:
        # ======================================================
        # 5. MODEL TRAINING & FORECASTING
        # ======================================================
        split_index = int(len(df_model) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # forecast 30 days
        future_forecast = forecast_next_n_days(
            model,
            df_model[["date", "daily_demand", "demand_smooth", "Festival Season"]],
            n_days=30,
        )

        # ======================================================
        # 6. DEMAND FORECASTING SECTION
        # ======================================================
        st.subheader("2. Demand Forecasting (SKU × Region)")

        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Model MAE (smoothed demand)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{mae:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-note">Lower = closer fit on hold-out period.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_kpi2:
            st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Model RMSE (smoothed demand)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{rmse:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-note">Penalizes larger errors more heavily.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_kpi3:
            hist_days = df_model["date"].nunique()
            st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">History length (days)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{hist_days}</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-note">Daily sales used to train the model.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        col_forecast, col_fit = st.columns(2)

        # Smoothed demand + forecast chart
        with col_forecast:
            hist = df_model.copy()
            hist["type"] = "Smoothed history"
            fc = future_forecast.copy()
            fc["type"] = "Forecast"
            fc = fc.rename(columns={"forecast_demand": "demand_smooth"})

            combined = pd.concat([
                hist[["date", "demand_smooth", "type"]],
                fc[["date", "demand_smooth", "type"]],
            ])

            fig_forecast = px.line(
                combined,
                x="date",
                y="demand_smooth",
                color="type",
                title=f"Smoothed daily demand & 30-day forecast – {sku} / {region}",
                labels={"demand_smooth": "Smoothed demand (units)", "date": "Date"},
                color_discrete_map={
                    "Smoothed history": "#6366f1",
                    "Forecast": "#a855f7",
                },
            )
            fig_forecast.update_layout(
                legend_title_text="",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        # Actual vs Predicted chart (test period)
        with col_fit:
            df_fit = pd.DataFrame({
                "date": df_model["date"].iloc[split_index:],
                "Actual": y_test.values,
                "Predicted": y_test_pred,
            })
            df_fit_melt = df_fit.melt("date", var_name="Series", value_name="Demand")

            fig_fit = px.line(
                df_fit_melt,
                x="date",
                y="Demand",
                color="Series",
                title="Model fit – Actual vs Predicted (hold-out period)",
                color_discrete_map={
                    "Actual": "#9ca3af",
                    "Predicted": "#6366f1",
                },
            )
            fig_fit.update_layout(
                legend_title_text="",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_fit, use_container_width=True)

        # Demand KPIs (from forecast)
        mean_daily, total_mean, daily_std, total_std = compute_lead_time_stats_from_forecast(
            future_forecast, lead_time_days=7
        )

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Avg daily demand (next 7 days)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{mean_daily:.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_d2:
            st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Volatility σ (per day)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{daily_std:.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_d3:
            st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Lead-time demand (7 days)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{total_mean:.1f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            **Interpretation – Demand Forecasting**  
            For **{sku}** in **{region}**, the smoothed historical demand curve shows the underlying trend
            without day-to-day noise. The forecast (purple line) extends this pattern over the next 30 days.
            On the right, the model’s predicted series tracks the actual smoothed demand quite closely
            (MAE ≈ {mae:.2f}, RMSE ≈ {rmse:.2f}), indicating that the Random Forest captures weekend
            effects and seasonal changes reasonably well.  

            For the next 7 days of lead time, the model expects an average of about **{mean_daily:.2f} units/day**
            with a volatility (σ) of **{daily_std:.2f} units/day**. Higher volatility implies more uncertainty and
            normally requires slightly higher safety stock to avoid stockouts.
            """
        )

        st.markdown("---")

        # ======================================================
        # 7. DEMAND BY REGION & MONTH (for this SKU)
        # ======================================================
        st.subheader("3. Demand by Region & Month (selected SKU)")

        sku_hist = daily_demand[daily_demand["SKU"] == sku].copy()

        region_totals = (
            sku_hist.groupby("Region")["daily_demand"]
            .sum()
            .reset_index()
            .sort_values("daily_demand", ascending=False)
        )

        month_totals = (
            sku_hist.groupby("Month_Name")["daily_demand"]
            .sum()
            .reindex(index=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
            .reset_index()
            .rename(columns={"Month_Name": "Month"})
        )

        col_reg, col_mon = st.columns(2)
        with col_reg:
            fig_reg = px.bar(
                region_totals,
                x="Region",
                y="daily_demand",
                title=f"Total demand by region – {sku}",
                labels={"daily_demand": "Units sold"},
                color="Region",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_reg.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_reg, use_container_width=True)
        with col_mon:
            fig_mon = px.bar(
                month_totals,
                x="Month",
                y="daily_demand",
                title=f"Total demand by month – {sku} (all regions)",
                labels={"daily_demand": "Units sold"},
                color="Month",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_mon.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_mon, use_container_width=True)

        peak_region = region_totals.iloc[0]["Region"] if not region_totals.empty else "-"
        peak_month_row = month_totals.sort_values("daily_demand", ascending=False).iloc[0]
        peak_month = peak_month_row["Month"] if not month_totals.empty else "-"

        st.markdown(
            f"""
            **Interpretation – Demand by Region & Month**  
            - The left chart shows that **{sku}** is strongest in **{peak_region}**, suggesting that this region
              should get priority when allocating limited inbound stock.  
            - The right chart reveals that demand peaks around **{peak_month}**, which can be associated with
              festive or promotional periods.  

            Taken together, these views help the manager decide **where** to hold more inventory (key regions)
            and **when** to ramp up stock (peak months), instead of spreading stock evenly and risking
            stockouts in high-demand locations.
            """
        )

        st.markdown("---")

        # ======================================================
        # 8. REORDER OPTIMIZATION
        # ======================================================
        st.subheader("4. Reorder Level Optimization")

        reorder_df, best_policy, mean_daily_lt, daily_std_lt, total_mean_lt = optimize_reorder_policy(
            future_forecast,
            current_stock=current_stock,
            holding_cost_per_unit=holding_cost_per_unit,
            stockout_cost_per_unit=stockout_cost_per_unit,
            lead_time_days=7,
        )

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Optimal service level</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="kpi-value">{best_policy["Service_Level"]*100:.0f}%</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col_r2:
            st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Reorder level (units)</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="kpi-value">{best_policy["Reorder_Level"]:.1f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col_r3:
            st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Recommended reorder qty</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="kpi-value">{best_policy["Reorder_Qty"]:.1f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col_r4:
            st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Minimum total cost (₹)</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="kpi-value">{best_policy["Total_Cost"]:.1f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # nice table
        df_display = reorder_df.copy()
        df_display["Service_Level"] = (df_display["Service_Level"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(
            df_display.style.highlight_min(subset=["Total_Cost"], color="#bbf7d0"),
            use_container_width=True,
        )

        st.markdown(
            f"""
            **Interpretation – Reorder Optimization**  
            The AI forecast implies a 7-day lead-time demand of about **{total_mean_lt:.1f} units**.
            For each candidate service level (80–99%), the dashboard:  

            1. Computes **safety stock** = z × σ<sub>LT</sub>  
            2. Sets **reorder level** = mean lead-time demand + safety stock  
            3. Calculates **holding cost** (inventory × ₹{holding_cost_per_unit:.2f} per unit)  
            4. Estimates **expected stockout cost** (probability of stockout × demand × ₹{stockout_cost_per_unit:.2f})  

            The policy highlighted in green achieves the **lowest total cost**.  
            For **{sku} / {region}**, the recommended service level is around
            **{best_policy["Service_Level"]*100:.0f}%**, with a reorder level of
            **{best_policy["Reorder_Level"]:.1f} units**.  
            Given your current stock of **{current_stock:.1f} units**, the dashboard suggests reordering
            approximately **{best_policy["Reorder_Qty"]:.1f} units** to reach this optimal policy.
            """
        )

        st.markdown("---")

        # ======================================================
        # 9. LOGISTICS DISTRIBUTION & STOCKOUT RISK
        # ======================================================
        st.subheader("5. Logistics Distribution Across Regions & Stockout Risk")

        dist_df = optimize_logistics_distribution(
            sales,
            daily_demand,
            sku,
            inbound_units=inbound_units,
            lead_time_days=7,
        )

        if not dist_df.empty:
            st.dataframe(dist_df, use_container_width=True)

            # stockout banner for selected region
            row_sel = dist_df[dist_df["Region"] == region].iloc[0]
            final_stock = row_sel["Current_Stock"] + row_sel["Inbound_Allocated"]
            coverage_ratio = final_stock / (row_sel["LT_Demand_7d"] or 1)
            shortage_after = row_sel["Shortage_After"]

            if shortage_after > 0:
                # suggest extra reorder
                extra_reorder = shortage_after * 1.2
                banner_class = "stock-banner stock-high"
                banner_text = (
                    f"🛑 <strong>HIGH STOCKOUT RISK</strong> for <strong>{region}</strong>. "
                    f"Forecast 7-day demand ≈ <strong>{row_sel['LT_Demand_7d']:.1f} units</strong>, "
                    f"but stock after allocation is only <strong>{final_stock:.1f} units</strong>. "
                    f"Expected shortfall ≈ <strong>{shortage_after:.1f} units</strong>.<br>"
                    f"→ Suggested **additional reorder** ≈ <strong>{extra_reorder:.0f} units</strong> "
                    f"for this region to cover the gap with a buffer."
                )
            elif coverage_ratio < 1.2:
                extra_reorder = row_sel["LT_Demand_7d"] * 0.2
                banner_class = "stock-banner stock-medium"
                banner_text = (
                    f"⚠️ <strong>WATCH LEVELS</strong> in <strong>{region}</strong>. "
                    f"Stock after allocation just covers the 7-day demand "
                    f"(coverage ratio ≈ <strong>{coverage_ratio:.2f}</strong>). "
                    f"A modest spike could cause a stockout.<br>"
                    f"→ Optionally, reorder ~<strong>{extra_reorder:.0f} units</strong> as a precaution."
                )
            else:
                banner_class = "stock-banner stock-low"
                banner_text = (
                    f"✅ <strong>LOW STOCKOUT RISK</strong> in <strong>{region}</strong>. "
                    f"Stock after allocation comfortably covers the 7-day forecasted demand "
                    f"(coverage ratio ≈ <strong>{coverage_ratio:.2f}</strong>). "
                    f"No immediate extra reorder is required for this region."
                )

            st.markdown(
                f'<div class="{banner_class}">{banner_text}</div>',
                unsafe_allow_html=True,
            )

            worst_row = dist_df.sort_values("Shortage_After", ascending=False).iloc[0]
            best_row_log = dist_df.sort_values("Excess_After", ascending=False).iloc[0]

            st.markdown(
                f"""
                **Interpretation – Logistics & Risk**  
                The inbound batch of **{inbound_units:.0f} units** is distributed starting with regions that have
                the largest **forecasted shortfall**. After allocation:

                - The most constrained region is **{worst_row['Region']}** with an expected shortage of
                  **{worst_row['Shortage_After']:.1f} units**.  
                - The most comfortable region is **{best_row_log['Region']}** with an excess of
                  **{best_row_log['Excess_After']:.1f} units** above its 7-day demand.  

                The stockout banner above translates this into a clear action message for the selected region
                (**{region}**): whether you are safe, should watch levels, or need to **raise an additional
                reorder**. This is how AI-driven forecasting, reorder policy and logistics distribution
                come together to minimize stockouts and support faster deliveries.
                """
            )
        else:
            st.info("No demand history found for this SKU across regions – cannot run logistics optimization.")
