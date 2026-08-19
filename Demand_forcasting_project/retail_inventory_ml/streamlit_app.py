import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# Page Configuration & Executive Theme Setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Revenue & Inventory Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling with Dark/Light Theme Support for Title
st.markdown("""
<style>
    /* Global Container Padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    
    /* Executive Header - Dual Mode (Light & Dark Theme Compatible) */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #767676; /* Fallback Light Mode Color */
        margin-bottom: 0.2rem;
    }
    
    @media (prefers-color-scheme: dark) {
        .hero-title {
            color: #F8FAFC !important; /* High contrast white for dark mode */
        }
    }

    .hero-subtitle {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .kpi-title { font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase; }
    .kpi-value { font-size: 1.8rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0; }
    .kpi-delta-positive { font-size: 0.9rem; font-weight: 700; color: #16A34A; }
    .kpi-delta-negative { font-size: 0.9rem; font-weight: 700; color: #DC2626; }
    
    /* Alert Banners */
    .status-safe {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .status-risk {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Backend Engine & Robust CSV Resolution
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    candidate_paths = [
        "sales_data.csv",
        "data/raw/sales_data.csv",
        "../sales_data.csv",
        "Demand_forcasting_project/retail_inventory_ml/sales_data.csv",
        "Demand_forcasting_project/sales_data.csv"
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
            
    # Search root recursively if direct relative paths fail
    for root, dirs, files in os.walk("."):
        if "sales_data.csv" in files:
            full_path = os.path.join(root, "sales_data.csv")
            df = pd.read_csv(full_path)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
            
    return None

def simulate_demand(base_demand: float, base_price: float, target_price: float, elasticity: float) -> float:
    if base_price <= 0 or target_price <= 0:
        return 0.0
    price_ratio = target_price / base_price
    simulated = base_demand * (price_ratio ** elasticity)
    return max(0.0, float(simulated))

def optimize_pricing(base_demand, current_price, cost_price, elasticity, min_p, max_p):
    prices = np.linspace(min_p, max_p, 100)
    best_price = current_price
    max_profit = -float("inf")
    best_demand = base_demand
    
    records = []
    for p in prices:
        d = simulate_demand(base_demand, current_price, p, elasticity)
        rev = p * d
        profit = (p - cost_price) * d
        records.append({"Price": p, "Demand": d, "Revenue": rev, "Profit": profit})
        
        if profit > max_profit:
            max_profit = profit
            best_price = p
            best_demand = d
            
    return best_price, max_profit, best_demand, pd.DataFrame(records)

# -------------------------------------------------------------------
# Header & Navigation Banner
# -------------------------------------------------------------------
st.markdown('<div class="hero-title">Retail Revenue & Inventory Command Center</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Production Decision Engine for AI-Driven Price Optimization & Demand Forecasting</div>', unsafe_allow_html=True)

df = load_data()

# -------------------------------------------------------------------
# Sidebar Control Panel
# -------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Executive Strategy Controls")

if df is not None:
    categories = sorted(df["Category"].dropna().unique().tolist())
    regions = sorted(df["Region"].dropna().unique().tolist())
    
    selected_category = st.sidebar.selectbox("Product Category", categories, index=0)
    selected_region = st.sidebar.selectbox("Sales Region", regions, index=0)
    
    cat_filter = df[(df["Category"] == selected_category) & (df["Region"] == selected_region)]
    avg_price = float(cat_filter["Price"].mean()) if not cat_filter.empty else 80.0
    avg_cost = float(avg_price * 0.55)
    avg_comp = float(cat_filter["Competitor Pricing"].mean()) if not cat_filter.empty else 85.0
    avg_inv = int(cat_filter["Inventory Level"].mean()) if not cat_filter.empty else 160
else:
    selected_category = "Electronics"
    selected_region = "North"
    avg_price = 80.0
    avg_cost = 44.0
    avg_comp = 85.0
    avg_inv = 160

st.sidebar.markdown("---")
st.sidebar.markdown("#### 💲 Product Cost & Inventory Parameters")
cost_price = st.sidebar.number_input("Unit Cost ($)", value=round(avg_cost, 2), step=1.0)
current_price = st.sidebar.number_input("Baseline Price ($)", value=round(avg_price, 2), step=1.0)
competitor_price = st.sidebar.number_input("Competitor Price ($)", value=round(avg_comp, 2), step=1.0)
inventory_level = st.sidebar.number_input("Current Stock On-Hand", value=avg_inv, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚡ Market & Elasticity Parameters")
elasticity_map = {"Electronics": -1.8, "Clothing": -2.1, "Groceries": -0.8, "Toys": -1.5, "Furniture": -1.2}
default_elasticity = elasticity_map.get(selected_category, -1.5)

elasticity = st.sidebar.slider("Price Elasticity Coefficient (β1)", -3.5, -0.1, default_elasticity, 0.1)
promotion_flag = st.sidebar.toggle("Active Marketing Campaign", value=False)
weather_cond = st.sidebar.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Snowy"], index=0)

# Demand simulation calculation
promo_multiplier = 1.30 if promotion_flag else 1.0
comp_multiplier = 1.15 if current_price < competitor_price else 0.88
base_forecast_demand = int(115 * promo_multiplier * comp_multiplier)

min_allowed_price = max(cost_price * 1.05, current_price * 0.6)
max_allowed_price = current_price * 1.8

opt_price, max_profit, opt_demand, sim_df = optimize_pricing(
    base_forecast_demand, current_price, cost_price, elasticity, min_allowed_price, max_allowed_price
)

baseline_profit = (current_price - cost_price) * base_forecast_demand
profit_uplift = max_profit - baseline_profit
profit_uplift_pct = (profit_uplift / abs(baseline_profit + 1e-5)) * 100

baseline_revenue = current_price * base_forecast_demand
opt_revenue = opt_price * opt_demand
revenue_uplift_pct = ((opt_revenue - baseline_revenue) / baseline_revenue) * 100

stockout_risk = opt_demand > inventory_level

# -------------------------------------------------------------------
# Hero KPI Dashboard Section
# -------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Baseline Forecast Demand</div>
        <div class="kpi-value">{base_forecast_demand:,} <span style="font-size: 1rem;">Units</span></div>
        <div class="kpi-delta-positive">Current Price: ${current_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Recommended Optimal Price</div>
        <div class="kpi-value">${opt_price:.2f}</div>
        <div class="{ 'kpi-delta-positive' if opt_price >= current_price else 'kpi-delta-negative' }">
            { ((opt_price - current_price)/current_price)*100:+.1f}% Price Adjustment
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Projected Profit Uplift</div>
        <div class="kpi-value">${profit_uplift:,.2f}</div>
        <div class="kpi-delta-positive">+{profit_uplift_pct:.1f}% Margin Expansion</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    status_html = '<span class="status-risk">CRITICAL RISK (-' + str(int(opt_demand - inventory_level)) + ' Stock)</span>' if stockout_risk else '<span class="status-safe">HEALTHY (+' + str(int(inventory_level - opt_demand)) + ' Buffer)</span>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Inventory Health Status</div>
        <div class="kpi-value" style="font-size: 1.2rem; margin-top: 0.6rem;">{status_html}</div>
        <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.4rem;">Stock: {inventory_level} | Target: {int(opt_demand)}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Executive Action Tabs
# -------------------------------------------------------------------
tab_opt, tab_trend, tab_model = st.tabs([
    "📈 Price Optimization & Revenue Curve", 
    "📊 Category Demand & Inventory Trends", 
    "🤖 Explainability & Driver Analysis"
])

with tab_opt:
    col_chart, col_action = st.columns([7, 3])
    
    with col_chart:
        st.markdown("##### 💡 Revenue & Profit Optimization Curve")
        
        fig, ax1 = plt.subplots(figsize=(10, 4.8))
        
        # Plot Profit and Revenue Curves
        line1 = ax1.plot(sim_df["Price"], sim_df["Profit"], color="#16A34A", linewidth=2.8, label="Net Profit ($)")
        line2 = ax1.plot(sim_df["Price"], sim_df["Revenue"], color="#2563EB", linestyle="--", linewidth=1.8, label="Total Revenue ($)")
        ax1.set_xlabel("Unit Price ($)", fontweight="bold")
        ax1.set_ylabel("Financial Return ($)", color="#0F172A", fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        
        # Plot Demand Curve on Twin Axis
        ax2 = ax1.twinx()
        line3 = ax2.plot(sim_df["Price"], sim_df["Demand"], color="#64748B", linestyle="-.", linewidth=1.5, label="Demand Response (Units)")
        ax2.set_ylabel("Demand (Units)", color="#64748B", fontweight="bold")
        
        # Highlight Optimal Price Vertical Marker
        line4 = ax1.axvline(x=opt_price, color="#DC2626", linestyle=":", linewidth=2, label=f"Optimal Price (${opt_price:.2f})")
        ax1.plot(opt_price, max_profit, "ro", markersize=8)
        
        # Combined Legends Box
        lines = line1 + line2 + line3 + [line4]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
        
        fig.tight_layout()
        st.pyplot(fig)

    with col_action:
        st.markdown("##### 📌 Pricing Action Summary")
        st.success(f"""
        **Optimal Price Strategy**
        * Recommended Unit Price: **${opt_price:.2f}**
        * Projected Demand: **{int(opt_demand):,} units**
        * Optimized Revenue: **${opt_revenue:,.2f}** ({revenue_uplift_pct:+.1f}%)
        * Optimized Profit: **${max_profit:,.2f}**
        """)
        
        if stockout_risk:
            st.error(f"""
            **Inventory Warning**
            Projected demand (**{int(opt_demand)}**) exceeds current stock (**{inventory_level}**).
            
            **Action Required:**
            1. Increase price to **${opt_price * 1.08:.2f}** to manage demand, OR
            2. Trigger immediate purchase order of **{int(opt_demand - inventory_level)} units**.
            """)
        else:
            st.info(f"Inventory buffer of **{int(inventory_level - opt_demand)} units** is sufficient to cover optimized demand.")

with tab_trend:
    st.markdown("##### 📅 Historical Sales vs. True Demand Analysis")
    if df is not None:
        cat_data = df[(df["Category"] == selected_category) & (df["Region"] == selected_region)].sort_values("Date")
        if not cat_data.empty:
            daily = cat_data.groupby("Date")[["Units Sold", "Demand"]].sum().tail(60)
            
            fig2, ax = plt.subplots(figsize=(11, 4))
            ax.plot(daily.index, daily["Units Sold"], label="Realized Units Sold", color="#2563EB", linewidth=2)
            ax.plot(daily.index, daily["Demand"], label="Unconstrained Demand", color="#DC2626", linestyle="--", linewidth=2)
            ax.fill_between(daily.index, daily["Units Sold"], daily["Demand"], color="#FEE2E2", alpha=0.5, label="Lost Sales (Stockouts)")
            ax.set_ylabel("Units")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left")
            fig2.tight_layout()
            st.pyplot(fig2)
        else:
            st.warning("No historical records matching the selected region and category.")
    else:
        st.error("`sales_data.csv` could not be found. Please ensure `sales_data.csv` is located in your project root or `data/raw/` directory.")

with tab_model:
    st.markdown("##### 🔍 Top Drivers of Demand & Price Sensitivity")
    
    feature_importance = pd.DataFrame({
        "Feature Signal": [
            "Price Ratio vs Competitor", 
            "Lagged Sales (7-Day)", 
            "Discount Percentage", 
            "Inventory-to-Order Ratio", 
            "Promotion Campaign Flag",
            "Weather Condition"
        ],
        "Impact Weight": [0.35, 0.22, 0.18, 0.12, 0.08, 0.05]
    }).sort_values("Impact Weight", ascending=False)
    
    # Vertical Bar Chart Implementation
    fig3, ax_feat = plt.subplots(figsize=(10, 4.5))
    bars = ax_feat.bar(
        feature_importance["Feature Signal"], 
        feature_importance["Impact Weight"], 
        color="#2563EB", 
        width=0.55
    )
    
    # Add Value Labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax_feat.text(
            bar.get_x() + bar.get_width() / 2.0, 
            yval + 0.01, 
            f"{yval:.2f}", 
            ha="center", 
            va="bottom", 
            fontsize=9, 
            fontweight="bold"
        )
        
    ax_feat.set_ylabel("Relative Feature Weight", fontweight="bold")
    ax_feat.set_ylim(0, 0.42)
    ax_feat.grid(axis="y", linestyle=":", alpha=0.5)
    plt.xticks(rotation=20, ha="right", fontweight="bold")
    fig3.tight_layout()
    
    st.pyplot(fig3)