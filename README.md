# Baku House Price Prediction — Linear Regression

A full linear regression workflow predicting apartment prices in Baku,
Azerbaijan, from listing features (size, floor, location, condition, etc.).
Built to demonstrate a complete, leakage-aware regression pipeline: EDA,
feature engineering, assumption checks, multicollinearity handling,
encoding, scaling, evaluation, and feature-importance analysis.

## Results

| Metric | Train | Test |
|---|---|---|
| R² | 0.756 | 0.759 |
| Adjusted R² | 0.756 | 0.758 |
| MAPE | 1.61% | 1.60% |
| RMSE (log price) | 0.268 | 0.268 |

The final model uses just **5 features** — `square`, `which_floor`,
`new_building`, and two location indicators — and explains about **76%**
of the variance in log(price) on unseen data, with train and test scores
nearly identical (no meaningful overfitting).

## Process

1. **Data cleaning** — dropped rows with missing values; dropped `ID` and
   `price_per_sqm` (the latter is derived from the target and would cause
   data leakage).
2. **Feature engineering** — added location-level aggregates
   (`Avg_square_by_location`, `Max_floors_by_location`).
3. **Linearity check** — log-transformed `price` after seeing a right-skewed
   scatter plot against `square`/`rooms`.
4. **Train/test split** — 80/20, done *before* any step that could leak
   test information (outlier bounds, scaling, encoding).
5. **Outlier treatment** — IQR bounds computed from the training set only,
   then applied to both train and test.
6. **Distribution check** — Kolmogorov–Smirnov test showed all numeric
   features are non-normal, so Spearman (rank-based) correlation was used
   instead of Pearson.
7. **Multicollinearity check** — VIF flagged the location-aggregate
   features (VIF ≈ 65) and `total_floors`/`rooms` as redundant with other
   predictors; these were dropped.
8. **High-cardinality handling** — `location` had 111 categories; any
   location with fewer than 100 training examples was grouped into
   `"Other"` before one-hot encoding, reducing it to 44 categories.
9. **Encoding & scaling** — one-hot encoding (`drop='first'`) for
   categoricals, `StandardScaler` fit on training data only.
10. **Modeling** — fit a full `LinearRegression` model, then ran a
    per-feature univariate R² analysis to identify the strongest individual
    predictors, and refit a simpler 5-feature model using those.
11. **Evaluation** — MAE, MSE, RMSE, MAPE, R², and adjusted R², compared
    between train and test to check for overfitting.

## Key takeaway

The 5-feature reduced model performs about as well as the full ~49-column
model, with a smaller train/test gap — a good example of how a simpler
model can generalize better than throwing every engineered feature at the
problem.

## Repo structure

```
├── notebooks/
│   └── Baku_House_Price_LR.ipynb   # full analysis
├── data/
│   └── README.md                   # notes on the dataset (not committed — see below)
├── requirements.txt
└── README.md
```

## Data

The dataset (`Baku_House_Price_Prediction.xlsx`) is **not included in this
repo** — see [`data/README.md`](data/README.md) for details on obtaining it
and the column descriptions. Place it in `data/` before running the
notebook.

## Setup

```bash
git clone https://github.com/<your-username>/baku-house-price-regression.git
cd baku-house-price-regression
pip install -r requirements.txt
jupyter notebook notebooks/Baku_House_Price_LR.ipynb
```

## Tech stack

Python · pandas · NumPy · scikit-learn · statsmodels · SciPy · Matplotlib · Seaborn

## Possible next steps

- Try Ridge/Lasso regression to regularize the one-hot location coefficients
- Use k-fold cross-validation instead of a single train/test split
- Try target or frequency encoding for `location` as an alternative to one-hot
