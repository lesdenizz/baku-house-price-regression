# Baku House Price Prediction — Linear Regression

A linear regression project that predicts apartment/house prices in Baku, Azerbaijan from listing features such as size, floor, location, and building condition. Built as a practice project to demonstrate an end-to-end regression workflow: EDA, feature engineering, assumption checks, and model evaluation.

## Dataset

The dataset (`Baku_House_Price_Prediction.xlsx`, ~39,300 listings) contains the following fields:

| Column | Description |
|---|---|
| `location` | District / metro station area |
| `rooms` | Number of rooms |
| `square` | Total area (m²) |
| `price_per_sqm` | Price per square meter |
| `which_floor` | Floor the unit is on |
| `total_floors` | Total floors in the building |
| `is_top_floor` | Whether the unit is on the top floor |
| `new_building` | Whether it's a new building |
| `has_repair` | Whether the unit has been renovated |
| `has_bill_of_sale` | Whether a bill of sale is available |
| `has_mortgage` | Whether the unit is mortgage-eligible |
| `price` | Target variable — sale price |

> The raw data file is not included in this repo (see `.gitignore`). Place `Baku_House_Price_Prediction.xlsx` in the project root to reproduce the notebook.

## Workflow

The notebook (`Baku_House_Price_Prediction.ipynb`) follows these steps:

1. **Import libraries** — numpy, pandas, matplotlib, seaborn, scikit-learn
2. **Load the data** and inspect its structure
3. **Descriptive statistics** on all features
4. **Missing value check**
5. **Feature engineering** — added `Avg_square_by_location` and `Max_floors_by_location`
6. **Drop unnecessary columns** (`ID`, `location`, `is_top_floor`)
7. **Check linearity assumptions** between features and price
8. **Define inputs (X) and target (y)**
9. **Train/test split** (70/30, `random_state=1`)
10. **Outlier detection** using the IQR method, computed on the training set only
11. **Distribution check** with the Kolmogorov–Smirnov test
12. **Correlation analysis** between independent features
13. **Multicollinearity check** with Variance Inflation Factor (VIF)
14. **Encode categorical variables** with `OneHotEncoder` (drop-first)
15. **Feature scaling** with `StandardScaler` (fit on train, applied to test)
16. **Fit the linear regression model** on all selected features
17. **Evaluate the model** on train and test sets
18. **Univariate analysis** — R² of each individual feature, to see which variables drive the most predictive power
19. **Reduced model** — refit using only the strongest predictors (`square`, `price_per_sqm`)

## Results

**Full-feature model** (square, which_floor, price_per_sqm, new_building, has_repair, has_bill_of_sale, has_mortgage):

| Metric | Train | Test |
|---|---|---|
| MAE | 37,341.65 | 36,581.83 |
| RMSE | 113,935.61 | 91,850.92 |
| MAPE | 19.53% | 20.04% |
| R² | 0.630 | 0.713 |
| Adjusted R² | 0.630 | 0.713 |

Per-feature univariate R² showed `square` (0.53) and `price_per_sqm` (0.18) as the two strongest single predictors, with the remaining features contributing marginally.

**Reduced model** (`square` + `price_per_sqm` only):

| Metric | Train | Test |
|---|---|---|
| MAE | 37,879.57 | 36,992.13 |
| RMSE | 114,250.85 | 92,068.43 |
| MAPE | 19.69% | 20.07% |
| R² | 0.628 | 0.712 |
| Adjusted R² | 0.628 | 0.712 |

**Takeaway:** the two-feature reduced model performs almost identically to the full model, showing that unit size and price-per-square-meter carry nearly all the predictive signal in this dataset.

## Tech stack

- Python 3
- pandas, numpy
- scikit-learn (`LinearRegression`, `StandardScaler`, `OneHotEncoder`, `train_test_split`, `metrics`)
- statsmodels (VIF)
- scipy (Kolmogorov–Smirnov test)
- matplotlib, seaborn

## Repository structure

```
baku-house-price-regression/
├── Baku_House_Price_Prediction.ipynb   # Main analysis notebook
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run

```bash
git clone https://github.com/lesdenizz/baku-house-price-regression.git
cd baku-house-price-regression
pip install -r requirements.txt
jupyter notebook Baku_House_Price_Prediction.ipynb
```

## Author

**lesdenizz** — [GitHub](https://github.com/lesdenizz)
