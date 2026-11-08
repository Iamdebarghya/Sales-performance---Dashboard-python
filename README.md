# Sales Performance Dashboard

An interactive Streamlit application for exploring sales performance from a CSV file. The dashboard provides KPI summaries, a monthly trend, regional comparisons, product mix, filters and a filtered-data export.

## Business questions
- What is total sales for the selected dataset and filters?
- How does sales performance change over time?
- Which regions and products contribute most to sales?
- How does the filtered dataset differ from the full dataset?

## Features
- Upload a user-provided CSV
- Region filter and numeric sales-column selection
- Sales total and row-count KPIs
- Monthly trend when a date field is present
- Region and product charts when those fields are present
- Filtered CSV download

## Expected input
At minimum, provide one numeric sales/revenue column. For the full experience, use columns named `Date`, `Region`, `Product` and a numeric `Sales` field. Column names can be adjusted in the uploaded source; the app uses the selected numeric column for value calculations.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
The live Streamlit service is hosted on Render and tracks the `live-dashboard` branch. Free instances may sleep when idle and take a short time to wake.

## Data methodology
The dashboard aggregates the selected numeric field. It does not assume that sales equals profit. Use a separate profit column and explicitly extend the analysis before presenting profitability conclusions. Uploaded files are processed by the running app; do not upload confidential customer data.

## Validation
Run the test suite with:
```bash
python -m unittest discover -s tests -v
```

## Author
Debarghya Hajra | Python, Pandas, Streamlit, Plotly
