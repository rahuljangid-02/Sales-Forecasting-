# Sales Forecasting & Demand Intelligence System

An end-to-end data science project that forecasts monthly product demand, detects anomalous sales weeks, segments products by demand behavior, and surfaces it all through an interactive dashboard.

Built on the [Superstore Sales dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) (9,800 orders, Jan 2015 – Dec 2018), with the [Video Game Sales dataset](https://www.kaggle.com/datasets/gregorut/videogamesales) used for a multi-source merge exercise.

## What's in this repo

```
sales-forecasting-project/
└── Sales_Forecasting/
    ├── app.py                 # Streamlit dashboard
    ├── analysis.ipynb         # Complete analysis notebook
    ├── requirements.txt       # Project dependencies
    ├── summary.docx           # Executive summary
    ├── train.csv              # Superstore Sales dataset
    ├── vgsales.csv            # Video Game Sales dataset
    └── charts/                # Exported visualizations
```

## Key results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **XGBoost (recommended)** | $18,900 | $20,794 | **19.4%** |
| SARIMA | $19,244 | $19,950 | 20.5% |
| Prophet | $20,296 | $22,487 | 21.9% |

- **Top revenue category:** Technology ($827K), ahead of Furniture ($729K) and Office Supplies ($705K)
- **Most consistent growth:** East region — grew every single year, 2015–2018
- **Seasonality:** Nov/Dec/Sep consistently spike; Jan/Feb are consistently the slowest months
- **Anomalies:** 16 of 209 weeks flagged (Isolation Forest + rolling Z-score), mostly aligning with holiday peaks and post-holiday troughs
- **Product segments:** 17 sub-categories grouped into 4 demand clusters (High Volume/Stable, Growing, Low Volume/High Volatility, Declining), each with a different recommended stocking strategy

Full detail, charts, and reasoning are in `analysis.ipynb`; the plain-language version is in `summary.docx`.

## Getting started

### 1. Clone the repo and enter the project folder

```bash
git clone https://github.com/rahuljangid-02/Sales-Forecasting-.git
cd Sales-Forecasting-
```

All commands below assume you're inside `Sales-Forecasting-/`, where `app.py`, `analysis.ipynb`, and `requirements.txt` actually live.

### 2. Set up the environment

**Windows**

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Explore the analysis

```bash
jupyter notebook analysis.ipynb
```
Runs top to bottom with no manual steps — `train.csv` and `vgsales.csv` are loaded directly from this folder.

### 5. Run the dashboard locally

```bash
streamlit run app.py
```
Opens at `http://localhost:8501` with four pages: **Sales Overview**, **Forecast Explorer**, **Anomaly Report**, and **Product Demand Segments**. If port 8501 is already in use, Streamlit will automatically pick the next free port and print the actual URL in the terminal.

## Live deployment

- **Streamlit app:** [Live demo](https://qxdxbbuie9d5xfsvqfybw9.streamlit.app/)

To deploy your own copy: push this repo to GitHub, go to [share.streamlit.io](https://share.streamlit.io), point a new app at your repo with `Sales-Forecasting-/app.py` as the entry point, and deploy — it installs everything from `requirements.txt` automatically.

## Tech stack

Python · Pandas/NumPy · Statsmodels (SARIMA) · Prophet · XGBoost · Scikit-learn (Isolation Forest, K-Means, PCA) · Matplotlib/Seaborn · Plotly · Streamlit

## Notes & limitations

- Forecasts are based on 48 months of internal order data only — no external signals (competitor activity, marketing spend, macro trends) are included.
- The XGBoost future forecast is recursive (each predicted month feeds the next), so error can compound the further out it projects. Recommended to re-run monthly as new actuals arrive rather than treating any single forecast as fixed.
- See `summary.docx` for the full write-up of risks and business recommendations.