import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Performance Dashboard", page_icon="📊", layout="wide")
st.title("Sales Performance Dashboard")
st.caption("Explore sales performance from your CSV. Sample data is clearly labelled when no file is uploaded.")

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
        raw = pd.read_csv(uploaded)
        source_label = f"Uploaded file: {uploaded.name}"
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()
else:
    raw = SAMPLE.copy()
    source_label = "Illustrative sample data (not actual business results)"

if raw.empty:
    st.warning("The uploaded file has no rows.")
    st.stop()

st.info(source_label)
st.sidebar.header("Analysis settings")
columns = list(raw.columns)
numeric_columns = [c for c in columns if pd.api.types.is_numeric_dtype(raw[c])]
if not numeric_columns:
    st.error("No numeric columns found. Add a numeric sales or revenue field to the CSV.")
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

values = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
total = float(values.sum())
average = float(values.mean()) if len(values) else 0.0
m1, m2, m3 = st.columns(3)
m1.metric(f"Total {amount_col}", f"{total:,.2f}")
m2.metric(f"Average {amount_col}", f"{average:,.2f}")
m3.metric("Filtered rows", f"{len(df):,}")

if date_col != "None":
    trend = df[[date_col]].copy()
    trend[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    trend[date_col] = pd.to_datetime(trend[date_col], errors="coerce")
    trend = trend.dropna(subset=[date_col, amount_col])
    if not trend.empty:
        trend["Month"] = trend[date_col].dt.to_period("M").astype(str)
        monthly = trend.groupby("Month", as_index=False)[amount_col].sum()
        st.subheader("Monthly trend")
        st.plotly_chart(px.line(monthly, x="Month", y=amount_col, markers=True), use_container_width=True)

left, right = st.columns(2)
if region_col != "None" and not df.empty:
    regional = df.groupby(region_col, as_index=False)[amount_col].sum().sort_values(amount_col, ascending=False)
    left.subheader("Performance by region")
    left.plotly_chart(px.bar(regional, x=region_col, y=amount_col), use_container_width=True)
if product_col != "None" and not df.empty:
    products = df.groupby(product_col, as_index=False)[amount_col].sum().sort_values(amount_col, ascending=False)
    right.subheader("Performance by product")
    right.plotly_chart(px.bar(products, x=product_col, y=amount_col), use_container_width=True)

st.subheader("Filtered records")
st.dataframe(df, use_container_width=True)
st.download_button("Download filtered CSV", df.to_csv(index=False).encode("utf-8-sig"), "sales_filtered.csv", "text/csv")
st.caption("Definitions: the selected numeric field is summed as the displayed measure. Revenue, profit and margin are not inferred from one another.")
