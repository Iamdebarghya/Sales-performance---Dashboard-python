import streamlit as st
import pandas as pd
import plotly.express as px

from analytics import validate_dataset, calculate_kpis, aggregate_by, monthly_trend

st.set_page_config(page_title="Sales Performance Dashboard", page_icon="📊", layout="wide")
st.title("Sales Performance Dashboard")
st.caption("A reusable CSV analytics workflow for sales reporting.")

SAMPLE = pd.DataFrame({
    "Date": ["2024-01-01","2024-01-02","2024-02-03","2024-02-08","2024-03-01","2024-03-12"],
    "Region": ["North","South","East","West","North","South"],
    "Product": ["Laptop","Desk","Phone","Laptop","Desk","Phone"],
    "Sales": [1800,960,2100,900,1200,1450],
    "Quantity": [2,4,3,1,5,2],
})

uploaded = st.file_uploader("Upload sales CSV", type=["csv"])
if uploaded:
    try:
        raw = validate_dataset(pd.read_csv(uploaded))
        source_label = f"Uploaded file: {uploaded.name}"
    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")
        st.stop()
else:
    raw = SAMPLE.copy()
    source_label = "Illustrative sample data (not actual business results)"

st.info(source_label)
st.sidebar.header("Analysis settings")
columns = list(raw.columns)
numeric_columns = [c for c in columns if pd.to_numeric(raw[c], errors="coerce").notna().any()]
if not numeric_columns:
    st.error("No numeric measure found. Add a numeric sales or revenue column.")
    st.stop()

amount_col = st.sidebar.selectbox("Measure", numeric_columns, index=numeric_columns.index("Sales") if "Sales" in numeric_columns else 0)
region_col = st.sidebar.selectbox("Region field (optional)", ["None"] + columns, index=(columns.index("Region")+1) if "Region" in columns else 0)
product_col = st.sidebar.selectbox("Product field (optional)", ["None"] + columns, index=(columns.index("Product")+1) if "Product" in columns else 0)
date_col = st.sidebar.selectbox("Date field (optional)", ["None"] + columns, index=(columns.index("Date")+1) if "Date" in columns else 0)

df = raw.copy()
if region_col != "None":
    choices = sorted(df[region_col].dropna().astype(str).unique())
    selected = st.sidebar.multiselect("Filter regions", choices, default=choices)
    df = df[df[region_col].astype(str).isin(selected)]
if product_col != "None":
    choices = sorted(df[product_col].dropna().astype(str).unique())
    selected = st.sidebar.multiselect("Filter products", choices, default=choices)
    df = df[df[product_col].astype(str).isin(selected)]

kpis = calculate_kpis(df, amount_col)
m1, m2, m3 = st.columns(3)
m1.metric(f"Total {amount_col}", f"{kpis['total']:,.2f}")
m2.metric(f"Average {amount_col}", f"{kpis['average']:,.2f}")
m3.metric("Valid measure rows", f"{kpis['valid_rows']:,}")

if date_col != "None":
    trend = monthly_trend(df, date_col, amount_col)
    if not trend.empty:
        st.subheader("Monthly trend")
        st.plotly_chart(px.line(trend, x="month", y="value", markers=True), use_container_width=True)
    else:
        st.info("No valid date and measure pairs were found for the monthly trend.")

left, right = st.columns(2)
if region_col != "None" and not df.empty:
    regional = aggregate_by(df, region_col, amount_col)
    left.subheader("Performance by region")
    left.plotly_chart(px.bar(regional, x="category", y="value"), use_container_width=True)
if product_col != "None" and not df.empty:
    products = aggregate_by(df, product_col, amount_col)
    right.subheader("Performance by product")
    right.plotly_chart(px.bar(products, x="category", y="value"), use_container_width=True)

st.subheader("Filtered records")
st.dataframe(df, use_container_width=True)
st.download_button("Download filtered CSV", df.to_csv(index=False).encode("utf-8-sig"), "sales_filtered.csv", "text/csv")
st.caption("Definitions: totals and averages use valid numeric values in the selected measure. Revenue, profit and margin are not inferred from one another.")
