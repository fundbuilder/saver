import yfinance as yf
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download S&P 500 historical data
sp500 = yf.download("^GSPC", start="1980-01-01", end="2024-12-31")

# Save to CSV
sp500.to_csv("data/sp500_historical.csv")

print("S&P 500 data downloaded successfully to data/sp500_historical.csv")
