import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(page_title="Sales Performance",page_icon="📊",layout="wide")
st.title("Sales Performance Dashboard")
st.caption("Interactive sales analysis. Upload your own CSV or use the included sample.")
up=st.file_uploader("Upload sales CSV",type=["csv"])
if up is not None:
    df=pd.read_csv(up)
else:
    df=pd.DataFrame({"Date":["2023-01-01","2023-01-02","2023-01-03","2023-01-04","2023-01-05","2023-01-06","2023-01-07","2023-01-08"],"Region":["North","South","East","West","North","South","East","West"],"Product":["Product A","Product B","Product C","Product D","Product B","Product A","Product A","Product B"],"Sales":[1000,1200,800,900,1100,1300,1500,1400]})
st.sidebar.header("Filters")
regions=st.sidebar.multiselect("Region",sorted(df["Region"].dropna().astype(str).unique()),default=sorted(df["Region"].dropna().astype(str).unique())) if "Region" in df else []
if "Region" in df: df=df[df.Region.astype(str).isin(regions)]
amount=st.sidebar.selectbox("Sales column",[c for c in df if pd.api.types.is_numeric_dtype(df[c])])
a,b=st.columns(2); a.metric("Total sales",f"{df[amount].sum():,.0f}"); b.metric("Rows",len(df))
if "Date" in df:
 df["Date"]=pd.to_datetime(df.Date,errors="coerce"); m=df.dropna(subset=["Date"]).groupby(df["Date"].dt.to_period("M").astype(str))[amount].sum().reset_index(name="Sales")
 if not m.empty: st.plotly_chart(px.line(m,x="Date",y="Sales",markers=True),use_container_width=True)
if "Region" in df: st.plotly_chart(px.bar(df.groupby("Region",as_index=False)[amount].sum(),x="Region",y=amount),use_container_width=True)
if "Product" in df: st.plotly_chart(px.pie(df,names="Product",values=amount),use_container_width=True)
st.dataframe(df,use_container_width=True)
st.download_button("Download filtered CSV",df.to_csv(index=False),"sales_filtered.csv")
