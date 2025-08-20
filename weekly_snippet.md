# Weekly Snippet

**Author:** Zijun Qiu  
**Created Date:** July 8th, 2025  
**Last Updated Date:** July 10th, 2025  

---

# week 1
### Database Service
- Used `yfinance` to download OHLCV data for 10 tech stocks (e.g., "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "AMZN", "TSLA", "CRM", "INTU")  
- Collected basic fundamental data (P/E, EPS, Revenue) using `yfinance` and financial statement parsing (in progress)  
- Stored both price and fundamentals as local CSV files in `/data/raw/`  


---

## Progress and Issues Resolved
- Reviewed all onboarding materials and internship documents  
- Set up Python development environment using github mainly,jupiter notebook partially 
- Installed required libraries: `yfinance`, `pandas`, `numpy`, `matplotlib`  
- Initialized GitHub repo for version control and created folder structure
Stock Price & Fundamentals Data Extraction  
- Project Environment Setup
Used `yfinance` to download OHLCV data for 10 tech stocks (e.g., "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "AMZN", "TSLA", "CRM", "INTU")  
- Collected basic fundamental data (P/E, EPS, Revenue) using `yfinance` and financial statement parsing  
- Stored both price and fundamentals as local CSV files in `week1/data/`  

## Recommendation Service
not yet

---

## Next Week Plan (Week 2)

- Handle missing values and inconsistent date indices  
- Standardize and normalize numeric features  
- Generate data quality report (null %, stats, fixes applied)  
- Finalize a unified structured dataset for feature engineering  

---

# week 2  
### Data Cleaning & Preparation
- Loaded raw stock CSV files from `/data/raw/`
- Cleaned each file using pandas:
  - Parsed `Date` column as datetime
  - Sorted values by date
  - Forward filled missing values
  - Dropped any remaining NaNs
- Saved cleaned data to `/cleaned/` folder as `{TICKER}_price.csv`

---

## Progress and Issues Resolved
- Implemented consistent cleaning function for all 10 stock CSVs
- Verified cleaned outputs and file integrity
- Encountered path error due to incorrect `input_dir`; fixed by updating path references
- Saved cleaned data for all stocks in `/cleaned/` successfully for next stage

---

# week 3  
### Feature Engineering
- Loaded cleaned data from `/cleaned/`
- Engineered the following features for each stock:
  - SMA (20-day, 50-day)
  - EMA (20-day, 50-day)
  - RSI (14-day)
  - Bollinger Bands (20-day window with ±2 std)
  - MACD and Signal Line (12,26 EMA)
  - Rolling Volatility (20-day std)
  - Lag features (1-day, 5-day)
  - Rolling mean and std for `Close` column
- Saved updated enriched datasets into `week3/` folder
- Generated correlation matrix for each stock’s features and saved as `feature_corr_{stock}.csv`

---

## Progress and Issues Resolved
- Finalized full feature engineering script using pandas
- All indicators computed and saved for 10 stocks
- Created and stored correlation matrices for exploratory data analysis
- Initially faced `FileNotFoundError` (missing `week2/cleaned` path); resolved by checking correct folder mapping
- Confirmed output files saved in `week3/` and verified feature columns

# week 4

### Exploratory Data Analysis (EDA)
- Loaded data from `week3/featured`
- Created price trend charts for each stock with:
  - Simple Moving Average (14-day)
  - Exponential Moving Average (14-day)
  - Relative Strength Index (14-day)
  - Moving Average Convergence Divergence and Signal Line
  - Bollinger Bands (20-day window with ±2 standard deviations)
- Made correlation heatmaps to show how features are related
- Used seasonal decomposition to find patterns like trend and seasonality in closing prices
- Added two target columns for modeling:
  - `NextDay_Return`: shows the percent change in closing price the next day (for regression)
  - `Return_Label`: 1 if next day’s return is positive, 0 if negative (for classification)
- Saved everything (charts, heatmaps, updated data) in the folder `week4/eda_outputs/`

---

## Progress and Issues Resolved
- Finished EDA for all 10 stocks without errors
- Charts and indicators looked correct when checked by eye
- Some stocks didn’t have enough data for decomposition, but the rest worked well
- Found that features like rolling standard deviation and Moving Average Convergence Divergence are related to return direction
- Made sure all files were saved in the right folder with the right names

# week 5
API key is: 61a08f8e1e6c4f91b7af08392130ccbb

https://newsapi.org/

### Exploratory Data Analysis (EDA)  
- Loaded processed feature dataset from `week4/eda_outputs/`  
- Generated advanced correlation heatmaps for 10 tech stocks (using seaborn) to highlight:  
  - Strong positive correlations among mega-cap tech firms (e.g., AAPL–MSFT, META–GOOGL)  
  - Weak or negative correlations between growth and cyclical stocks (e.g., NVDA–ABNB)  
- Created rolling volatility measures:  
  - 14-day and 30-day rolling standard deviation of returns  
  - Compared short-term vs. long-term volatility patterns  
- Investigated stationarity using Augmented Dickey-Fuller (ADF) test for closing prices  
- Added engineered features:  
  - Lagged returns (1-day, 3-day, 5-day)  
  - Rolling mean & variance windows (7-day, 30-day)  
- Saved visualizations (heatmaps, rolling vol plots) and updated dataset in `week5/features/`  

---

## Progress and Issues Resolved  
- Confirmed data pipeline reuses cleaned files from Week 2 (avoided duplication)  
- Correlation heatmaps revealed redundancy among certain features (e.g., EMA highly overlaps with SMA)  
- Identified multicollinearity risk in regression models due to correlated indicators  
- Some ADF tests failed (non-stationary series); flagged for future differencing or log returns  
- All outputs saved under `week5/eda_outputs/`  

---

# week 6  

### Exploratory Data Analysis (EDA)  
- Loaded extended features dataset from `week5/features/`  
- Applied Principal Component Analysis (PCA) for dimensionality reduction:  
  - First 3 components explained ~85% of variance across all indicators  
  - Observed that PCA grouped momentum vs. volatility features into separate clusters  
- Built feature importance ranking using correlation with target (`NextDay_Return`)  
- Visualized heatmaps for feature-target correlations across all 10 stocks  
- Tested classification baseline using Logistic Regression on `Return_Label`:  
  - Train-test split (80/20)  
  - Accuracy baseline ~55–60% (above random, but still modest)  
- Saved reduced PCA datasets and baseline model outputs in `week6/models/`  

---

## Progress and Issues Resolved  
- PCA confirmed redundancy in moving average features, supporting earlier correlation findings  
- Logistic regression highlighted which features had predictive signal (e.g., RSI, short-term volatility)  
- Noted imbalance in `Return_Label` distribution (more 0s than 1s), requiring resampling in later weeks  
- Confirmed all outputs and plots exported correctly to `week6/eda_outputs/`  
