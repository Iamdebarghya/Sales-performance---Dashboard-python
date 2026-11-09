# Sales Performance Dashboard

An interactive sales reporting application built with Python, Streamlit, Pandas and Plotly. Upload a CSV, choose a numeric measure and dimensions, filter the data, inspect trends and export filtered records.

## Business questions
- What is the total and average value of the selected measure?
- Which regions and products contribute most to that measure?
- How does the selected measure change month by month?
- How do results change when region or product filters are applied?

## Features
- CSV upload and basic data validation
- Configurable measure, date, region and product fields
- KPI cards, monthly trend and grouped comparisons
- Filtered data preview and CSV export
- Reusable calculations separated into `analytics.py`
- Unit tests for validation, KPIs, grouping and date handling

## Architecture
```text
CSV -> validation -> analytics functions -> Streamlit + Plotly -> filtered CSV
```

## Run locally
Python 3.10+:
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Input contract
At least one numeric column is required. Date, region and product fields are optional. Choose the relevant columns in the sidebar. The selected numeric measure is summed as-is. Revenue, profit and margin are not inferred from column names.

## Tests
```bash
python -m unittest discover -s tests -v
```

## Data quality and limitations
Invalid numeric cells and dates are excluded from the corresponding calculations. Missing category labels are excluded from grouped summaries. Sample data is illustrative only and must not be represented as real business performance. This is a CSV analytics app, not an ERP integration or forecasting system.

## Author
Debarghya Hajra | Python, Pandas, Streamlit, Plotly
