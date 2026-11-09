import pandas as pd

def validate_dataset(data):
    if data is None or data.empty: raise ValueError("The dataset is empty.")
    if data.columns.duplicated().any(): raise ValueError("Duplicate column names are not supported.")
    return data.copy()

def calculate_kpis(data, measure):
    if measure not in data.columns: raise KeyError(f"Unknown measure column: {measure}")
    values=pd.to_numeric(data[measure],errors="coerce").dropna()
    return {"total":float(values.sum()),"average":float(values.mean()) if not values.empty else 0.0,"valid_rows":int(values.count())}

def aggregate_by(data, category, measure):
    if category not in data.columns or measure not in data.columns: raise KeyError("Unknown category or measure column.")
    work=pd.DataFrame({"category":data[category].astype("string"),"value":pd.to_numeric(data[measure],errors="coerce")}).dropna()
    return work.groupby("category",as_index=False)["value"].sum().sort_values("value",ascending=False).reset_index(drop=True)

def monthly_trend(data, date_column, measure):
    if date_column not in data.columns or measure not in data.columns: raise KeyError("Unknown date or measure column.")
    work=pd.DataFrame({"date":pd.to_datetime(data[date_column],errors="coerce"),"value":pd.to_numeric(data[measure],errors="coerce")}).dropna()
    if work.empty: return pd.DataFrame(columns=["month","value"])
    work["month"]=work["date"].dt.to_period("M").astype(str)
    return work.groupby("month",as_index=False)["value"].sum().sort_values("month").reset_index(drop=True)
