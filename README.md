# Sales Performance Dashboard

An interactive sales reporting application built with Python, Streamlit, Pandas and Plotly. Upload a CSV, select the measure and dimensions, filter the dataset, inspect trends and export the filtered records.

## Business questions
- What is the total and average value of the selected sales measure?
- Which regions and products contribute the most to the selected measure?
- How does the selected measure change month by month?
- How do results change when users filter the region or product?

## Features
- CSV upload with basic dataset validation
- Selectable numeric measure, date field, region and product dimensions
- KPI cards for total, average and valid observation count
- Monthly trend with invalid dates excluded
- Region and product aggregations
- Filtered table and CSV export
- Reusable analytics functions separated from the Streamlit interface
- Unit tests for aggregation, KPI calculations, date handling and invalid inputs

## Architecture
```text
CSV upload
   |
   v
Dataset validation (analytics.py)
   |
   +--> KPI calculations
   +--> Region / product aggregations
   +--> Monthly trend
   |
   v
Streamlit UI + Plotly charts
   |
   v
Filtered table / CSV export
```

## Run locally
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Without an uploaded file, the app uses a small illustrative dataset labelled as sample data.

## Input format
The CSV should contain at least one numeric column. Date, region and product columns are optional and can be selected in the sidebar. The app does not assume that a column named "Sales" is profit or revenue unless the source owner defines it that way.

## Tests
```bash
python -m unittest discover -s tests -v
```

## Data quality and interpretation
- Non-numeric measure cells are excluded from KPI and aggregation calculations.
- Invalid dates are excluded from monthly trend calculations.
- Missing category labels are excluded from grouped summaries.
- Results depend on the selected measure and uploaded source data.
- The sample data is synthetic and must not be presented as company performance.

## Limitations
This is a CSV analytics application, not a live ERP integration or a production forecasting system. It does not calculate profit or margin without explicit source columns and definitions.

## Author
Debarghya Hajra | Python, Pandas, Streamlit, Plotly
