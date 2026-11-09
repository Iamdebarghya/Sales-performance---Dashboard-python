import streamlit as st
import pandas as pd
import plotly.express as px

def validate_dataset(data):
    if data is None or data.empty:
        raise ValueError("The dataset is empty.")
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")
    return data.copy()

def calculate_kpis(data, measure):
    if measure not in data.columns:
        raise KeyError(f"Unknown measure column: {measure}")
    values=pd.to_numeric(data[measure],errors="coerce").dropna()
    return {"total":float(values.sum()),"average":float(values.mean()) if not values.empty else 0.0,"valid_rows":int(values.count())}

def aggregate_by(data, category, measure):
    if category not in data.columns or measure not in data.columns:
        raise KeyError("Selected category or measure column does not exist.")
    work=pd.DataFrame({"category":data[category].astype("string"),"value":pd.to_numeric(data[measure],errors="coerce")}).dropna()
    return work.groupby("category",as_index=False)["value"].sum().sort_values("value",ascending=False)

def monthly_trend(data, date_column, measure):
    work=pd.DataFrame({"date":pd.to_datetime(data[date_column],errors="coerce"),"value":pd.to_numeric(data[measure],errors="coerce")}).dropna()
    if work.empty: return pd.DataFrame(columns=["month","value"])
    work["month"]=work["date"].dt.to_period("M").astype(str)
    return work.groupby("month",as_index=False)["value"].sum().sort_values("month")

st.set_page_config(page_title="Sales Performance Dashboard",page_icon="📊",layout="wide")
st.title("Sales Performance Dashboard")
st.caption("A reusable CSV analytics workflow for sales reporting.")
SAMPLE=pd.DataFrame({"Date":["2024-01-01","2024-01-02","2024-02-03","2024-02-08","2024-03-01","2024-03-12"],"Region":["North","South","East","West","North","South"],"Product":["Laptop","Desk","Phone","Laptop","Desk","Phone"],"Sales":[1800,960,2100,900,1200,1450],"Quantity":[2,4,3,1,5,2]})
uploaded=st.file_uploader("Upload sales CSV",type=["csv"])
if uploaded:
    try:
        raw=validate_dataset(pd.read_csv(uploaded)); source=f"Uploaded file: {uploaded.name}"
    except Exception as exc:
        st.error(f"Could not process CSV: {exc}"); st.stop()
else:
    raw=SAMPLE.copy(); source="Illustrative sample data (not actual business results)"
st.info(source)
columns=list(raw.columns)
numeric=[c for c in columns if pd.to_numeric(raw[c],errors="coerce").notna().any()]
if not numeric:
    st.error("No numeric measure found. Add a numeric sales or revenue column."); st.stop()
st.sidebar.header("Analysis settings")
measure=st.sidebar.selectbox("Measure",numeric,index=numeric.index("Sales") if "Sales" in numeric else 0)
region=st.sidebar.selectbox("Region field",["None"]+columns,index=columns.index("Region")+1 if "Region" in columns else 0)
product=st.sidebar.selectbox("Product field",["None"]+columns,index=columns.index("Product")+1 if "Product" in columns else 0)
date=st.sidebar.selectbox("Date field",["None"]+columns,index=columns.index("Date")+1 if "Date" in columns else 0)
df=raw.copy()
for field,label in [(region,"Regions"),(product,"Products")]:
    if field!="None":
        choices=sorted(df[field].dropna().astype(str).unique())
        selected=st.sidebar.multiselect(f"Filter {label.lower()}",choices,default=choices)
        df=df[df[field].astype(str).isin(selected)]
k=calculate_kpis(df,measure)
c1,c2,c3=st.columns(3)
c1.metric(f"Total {measure}",f"{k['total']:,.2f}")
c2.metric(f"Average {measure}",f"{k['average']:,.2f}")
c3.metric("Valid measure rows",f"{k['valid_rows']:,}")
if date!="None":
    trend=monthly_trend(df,date,measure); st.subheader("Monthly trend")
    if trend.empty: st.info("No valid date and measure pairs found.")
    else: st.plotly_chart(px.line(trend,x="month",y="value",markers=True),use_container_width=True)
left,right=st.columns(2)
if region!="None" and not df.empty:
    left.subheader("Performance by region"); left.plotly_chart(px.bar(aggregate_by(df,region,measure),x="category",y="value"),use_container_width=True)
if product!="None" and not df.empty:
    right.subheader("Performance by product"); right.plotly_chart(px.bar(aggregate_by(df,product,measure),x="category",y="value"),use_container_width=True)
st.subheader("Filtered records")
st.dataframe(df,use_container_width=True)
st.download_button("Download filtered CSV",df.to_csv(index=False).encode("utf-8-sig"),"sales_filtered.csv","text/csv")
st.caption("The selected numeric field is summed as-is. Revenue, profit and margin are not inferred from column names.")
