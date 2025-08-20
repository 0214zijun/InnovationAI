# week 1
import yfinance as yf
import pandas as pd
import os

### folders
price_dir = "week1/Prices_Data"
fundamental_dir = "week1/fundamentals_Data"

### Create directories if they don't exist
os.makedirs(price_dir, exist_ok=True)
os.makedirs(fundamental_dir, exist_ok=True)

### 10 tech tickers
tickers = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "AMZN", "TSLA", "CRM", "INTU"]

### Loop over each ticker
for ticker in tickers:
    print(f"Processing {ticker}...")

    # Download price data
    price_data = yf.download(ticker, period="5y")
    price_data.reset_index(inplace=True)
    price_path = os.path.join(price_dir, f"{ticker}_price.csv")
    price_data.to_csv(price_path, index=False)
    print(f"{ticker}: Price data saved in {price_path}")

    # Get fundamentals
    stock = yf.Ticker(ticker)
    info = stock.info

    fundamentals = {
        "Ticker": ticker,
        "EPS (TTM)": info.get("trailingEps"),
        "P/E Ratio (TTM)": info.get("trailingPE"),
        "Total Revenue (TTM)": info.get("totalRevenue"),
        "Operating Cash Flow (TTM)": info.get("operatingCashflow")
    }

    fundamentals_df = pd.DataFrame([fundamentals])
    fundamentals_path = os.path.join(fundamental_dir, f"{ticker}_fundamentals.csv")
    fundamentals_df.to_csv(fundamentals_path, index=False)
    print(f"{ticker}: Fundamentals saved in {fundamentals_path}")

print("\n All price and fundamental data saved in two separate folders inweek 1.")

# week 2
import os
import pandas as pd
import numpy as np

#input/output/report paths
input_path = 'week1/Prices_Data'
output_path = 'week2/cleaned'
report_path = 'week2/cleaning_report.csv'

os.makedirs(output_path, exist_ok=True)

#flatten multi-index columns if needed
def flatten_columns(df):
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    return df

### preprocess each CSV file
def preprocess_file(file):
    path = os.path.join(input_path, file)
    try:
        df = pd.read_csv(path)
        df = flatten_columns(df)

        # Rename 'Price_Date' to 'Date' if needed
        if 'Price_Date' in df.columns:
            df.rename(columns={'Price_Date': 'Date'}, inplace=True)

        # Check if 'Date' exists
        if 'Date' not in df.columns:
            raise ValueError("Missing 'Date' column")

        # Parse and clean dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.sort_values('Date', inplace=True)
        df.drop_duplicates(inplace=True)

        # Count nulls before fill
        na_before = df.isna().sum().sum()

        # Detect and replace outliers (IQR method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()
            df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan  # Set outliers to NaN

        # Fill missing values (including outliers set to NaN)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Count nulls after fill
        na_after = df.isna().sum().sum()
        filled_na = na_before - na_after

        # Normalize numeric columns (Z-score)
        df[numeric_cols] = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
        df[numeric_cols] = df[numeric_cols].round(4)

        # Save cleaned CSV
        cleaned_file = os.path.join(output_path, file)
        df.to_csv(cleaned_file, index=False)

        # Prepare report
        missing_ratio = df.isnull().mean().mean() * 100
        return {
            'File': file,
            'Missing Ratio Avg (%)': round(missing_ratio, 2),
            'Rows': len(df),
            'Filled NA': filled_na,
            'Status': 'Cleaned'}

    except Exception as e:
        return {
            'File': file,
            'Missing Ratio Avg (%)': 'N/A',
            'Rows': 'N/A',
            'Filled NA': 'N/A',
            'Status': f'Error: {str(e)}'}


#all CSVs in input folder
report = []
for file in sorted(os.listdir(input_path)):
    if file.endswith('.csv'):
        report.append(preprocess_file(file))

### Save cleaning report
pd.DataFrame(report).to_csv(report_path, index=False)
print(" Week 2 complete: Cleaned files saved and report generated.")

# week 3
import os
import pandas as pd
import numpy as np
import random

### Set directories
input_dir = 'week2/cleaned'
output_dir = 'week3/featured'
os.makedirs(output_dir, exist_ok=True)

### Feature engineering function
def compute_indicators(df):
    df = df.copy()
    df = df.sort_values('Date')

    # SMA & EMA
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Volatility + Lag features
    df['Rolling_Std_14'] = df['Close'].rolling(window=14).std()
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Rolling_Mean_7'] = df['Close'].rolling(window=7).mean()

    return df

### Nulls + Outlier injection
def inject_nulls_and_outliers(df, seed=42):
    df = df.copy()
    np.random.seed(seed)

    # Inject NULLs into 1% of rows in each numeric column
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        null_indices = np.random.choice(df.index, size=max(1, len(df) // 100), replace=False)
        df.loc[null_indices, col] = np.nan

    # Inject 3 random outliers into 'Close'
    outlier_indices = random.sample(list(df.index), 3)
    df.loc[outlier_indices, 'Close'] *= np.random.uniform(3, 6)  # 3x to 6x spike

    return df

### Process all CSVs
for file in sorted(os.listdir(input_dir)):
    if file.endswith('.csv'):
        filepath = os.path.join(input_dir, file)
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df = compute_indicators(df)
        df = inject_nulls_and_outliers(df)
        df.dropna(inplace=True)  # Optional: keep or remove this line depending on modeling use

        # Save enriched feature file
        out_path = os.path.join(output_dir, file)
        df.to_csv(out_path, index=False)
        print(f" Features + nulls/outliers saved for {file}")

        # Save correlation matrix
        corr_matrix = df.corr(numeric_only=True)
        stock = file.replace('_price.csv', '')
        corr_path = f'week3/feature_corr_{stock}.csv'
        corr_matrix.to_csv(corr_path)
        print(f"Correlation matrix saved for {stock}")

print("\n Week 3 complete: All 10 stocks processed with technical features, nulls, and outliers.")

# week 4 
### Week 4: Exploratory Data Analysis
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

### Set directories
input_dir = 'week3/featured'
output_dir = 'week4/eda_outputs'
os.makedirs(output_dir, exist_ok=True)

### Target variable: Next-day return
def compute_targets(df):
    df['NextDay_Return'] = df['Close'].pct_change().shift(-1)
    df['Return_Label'] = (df['NextDay_Return'] > 0).astype(int)
    return df

### Load and process files
for file in sorted(os.listdir(input_dir)):
    if file.endswith('.csv'):
        stock = file.replace('_price.csv', '')
        df = pd.read_csv(os.path.join(input_dir, file), parse_dates=['Date'])
        df = compute_targets(df)

        # Plot price + indicators
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Close'], label='Close Price', linewidth=1)
        ax.plot(df['Date'], df['SMA_14'], label='SMA 14', linestyle='--')
        ax.plot(df['Date'], df['EMA_14'], label='EMA 14', linestyle=':')
        ax.set_title(f"{stock}: Price with Moving Averages")
        ax.legend()
        fig.savefig(f"{output_dir}/{stock}_price_indicators.png")
        plt.close(fig)

        # Correlation heatmap
        numeric = df.select_dtypes(include=np.number)
        corr = numeric.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
        plt.title(f"{stock} - Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{stock}_correlation_heatmap.png")
        plt.close()

        # Time series decomposition (Close price)
        try:
            decomp = seasonal_decompose(df.set_index('Date')['Close'], model='multiplicative', period=30)
            decomp.plot()
            plt.suptitle(f"{stock} - Seasonal Decomposition", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{stock}_decomposition.png")
            plt.close()
        except Exception as e:
            print(f"Decomposition failed for {stock}: {e}")

        # Save enriched data with target
        df.to_csv(f"{output_dir}/{stock}_eda.csv", index=False)
        print(f"{stock} EDA complete.")

print("\n Week 4 EDA complete. Outputs saved in 'week4/eda_outputs'")

# week 5
### Week 5: Fetch news for 10 companies
import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# structure
COMPANIES = [
    "Apple", "Microsoft", "Google", "Meta", "NVIDIA",
    "AMD", "Amazon", "Tesla", "Salesforce", "Intuit"
]
NEWS_API_KEY = "61a08f8e1e6c4f91b7af08392130ccbb"

# Use last 30 days (free NewsAPI plans can only go back 30 days)
today      = datetime.utcnow().date()
start_date = (today - timedelta(days=30)).isoformat()
end_date   = today.isoformat()

# Create week5/news_data directory inside repo
output_dir = os.path.join("week5", "news_data")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "news_headlines.csv")

# fetch up to 100 articles
def fetch_company_news(company, api_key, start, end):
    """Fetch up to 100 recent articles for one company."""
    params = {
        "q"        : company,
        "from"     : start,
        "to"       : end,
        "sortBy"   : "publishedAt",
        "pageSize" : 100,  # free tier limit
        "page"     : 1,
        "language" : "en",
        "apiKey"   : api_key
    }
    url = "https://newsapi.org/v2/everything"
    r   = requests.get(url, params=params)
    data = r.json()
    articles = []
    if data.get("status") == "ok":
        for art in data.get("articles", []):
            articles.append({
                "company"     : company,
                "title"       : art.get("title"),
                "description" : art.get("description"),
                "source"      : art.get("source", {}).get("name"),
                "published_at": art.get("publishedAt")
            })
    else:
        # Print the error for debugging but return an empty list
        print(f"NewsAPI error for {company}: {data}")
    return articles

# Fetch news for 10 companies
all_articles = []
for comp in COMPANIES:
    print(f"Fetching news for {comp}…")
    all_articles.extend(fetch_company_news(comp, NEWS_API_KEY, start_date, end_date))
# Save results
if all_articles:
    df = pd.DataFrame(all_articles)
    df["published_at"] = pd.to_datetime(df["published_at"])
    df.sort_values("published_at", inplace=True)
    df.to_csv(csv_path, index=False)
    print(f"Collected {len(df)} articles.  Saved to {csv_path}")
else:
    print("No articles were returned.  Check your API key and date range.")

#week 6
###

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download Vader lexicon once
nltk.download('vader_lexicon')

#  Paths 
news_path  = os.path.join('week5', 'news_data', 'news_headlines.csv')
prices_dir = os.path.join('week2', 'cleaned')  
output_dir = os.path.join('week6', 'sentiment')
os.makedirs(output_dir, exist_ok=True)

#  Load news and compute sentiment 
df_news = pd.read_csv(news_path)
df_news['published_at'] = pd.to_datetime(df_news['published_at'])
df_news['date'] = df_news['published_at'].dt.date

sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    if pd.isna(text):
        return pd.Series([np.nan, np.nan])
    score = sia.polarity_scores(str(text))['compound']
    # classify based on compound score
    label = 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'
    return pd.Series([score, label])

df_news[['compound','sentiment']] = df_news['title'].apply(get_sentiment_scores)

#  Aggregate daily sentiment per company 
df_daily = (
    df_news
    .groupby(['company','date'])
    .agg(mean_compound = ('compound','mean'),
         positive_ratio = ('sentiment', lambda x: (x == 'Positive').mean()),
         negative_ratio = ('sentiment', lambda x: (x == 'Negative').mean()),
         neutral_ratio  = ('sentiment', lambda x: (x == 'Neutral').mean()),
         count          = ('sentiment', 'size'))
    .reset_index()
)

df_daily.to_csv(os.path.join(output_dir, 'daily_sentiment.csv'), index=False)

# Merge daily sentiment with stock prices 
merged_frames = []

for fname in sorted(os.listdir(prices_dir)):
    if fname.endswith('_price.csv'):
        ticker = fname.replace('_price.csv','')
        price_df = pd.read_csv(os.path.join(prices_dir, fname), parse_dates=['Date'])
        price_df['date'] = price_df['Date'].dt.date

        sentiment_df = df_daily[df_daily['company'] == ticker][['date','mean_compound','positive_ratio','negative_ratio','neutral_ratio']]
        merged = price_df[['date','Close']].merge(sentiment_df, on='date', how='left')

        # Forward-fill missing sentiment values (then backfill any leading NaNs)
        merged[['mean_compound','positive_ratio','negative_ratio','neutral_ratio']] = (
            merged[['mean_compound','positive_ratio','negative_ratio','neutral_ratio']]
            .ffill()
            .bfill()
        )

        merged['company'] = ticker
        merged_frames.append(merged)

df_merged = pd.concat(merged_frames, ignore_index=True)
df_merged.to_csv(os.path.join(output_dir, 'sentiment_price_merge.csv'), index=False)

#  Plot sentiment vs. closing price for each stock 
for ticker in df_merged['company'].unique():
    subset = df_merged[df_merged['company'] == ticker]

    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(subset['date'], subset['Close'], label='Close Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(subset['date'], subset['mean_compound'], label='Mean Sentiment', linestyle='--')
    ax2.set_ylabel('Mean Sentiment')

    plt.title(f'{ticker}: Daily Sentiment vs. Closing Price')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{ticker}_sentiment_price.png'))
    plt.close(fig)

print("Week 6 sentiment analysis complete.  Outputs saved to", output_dir)
3
