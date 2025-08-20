# Innovation AI – 12-Week Quantitative Finance & Machine Learning Internship

**Author:** Zijun Qiu  
**Internship Duration:** July 8 – September 27, 2025  
**Organization:** Innovation AI  
**Role:** Quantitative Finance & Machine Learning Intern  



print(r"""
                         _ooOoo_
                        o8888888o
                        88" . "88
                        (| -_- |)
                        O\  =  /O
                     ____/`---'\____
                   .'  \\|     |//  `.
                  /  \\|||  :  |||//  \
                 /  _||||| -:- |||||-  \
                 |   | \\\  -  /// |   |
                 | \_|  ''\---/''  |   |
                 \  .-\__  `-`  ___/-. /
               ___`. .'  /--.--\  `. . __
            ."" '<  `.___\_<|>_/___.'  >'"".
           | | :  `- \`.;`\ _ /`;.`/ - ` : | |
           \  \ `-.   \_ __\ /__ _/   .-` /  /
      ======`-.____`-.___\_____/___.-`____.-'======
                         `=---='

      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ALL BLESSINGS             NEVER BUGGING
""")


---

## Project Overview

This 12-week internship focuses on applying machine learning, data science, and natural language processing (NLP) techniques to financial markets. The primary objective is to build an AI-driven investment bot that predicts stock prices, integrates financial news sentiment, and generates real-time Buy/Sell/Hold recommendations for technology sector equities.

---

## Project Components

### 1. **Data Collection**
- Extracted historical OHLCV data for selected tech stocks using `yfinance`
- Collected fundamental financial indicators (EPS, P/E ratio, revenue, cash flow)
- Stored raw data in structured CSV format under version control

### 2. **Data Preprocessing**
- Cleaned and normalized raw data
- Handled missing values and aligned time series indices
- Generated quality reports on data integrity

### 3. **Feature Engineering & EDA**
- Created financial indicators (RSI, MACD, Bollinger Bands, moving averages)
- Performed exploratory data analysis with correlation plots and trend charts

### 4. **Sentiment Analysis**
- Scraped and parsed financial news headlines
- Applied FinBERT/VADER to quantify sentiment signals
- Integrated sentiment scores with price-based features

### 5. **Modeling**
- Trained predictive models: Linear Regression, Random Forest, XGBoost, Neural Networks
- Evaluated models using time-series-aware validation and financial performance metrics

### 6. **Backtesting**
- Simulated trading strategies based on model outputs and sentiment
- Compared against baseline strategies (Buy-and-Hold, Random Walk)
- Calculated Sharpe Ratio, Max Drawdown, and CAGR

### 7. **AI Recommendation Bot**
- Developed a decision engine to output real-time investment actions
- Implemented a basic CLI (and optionally Streamlit UI)
- Provided rationale and logs for every recommendation

---

## 📁 Repository Structure

