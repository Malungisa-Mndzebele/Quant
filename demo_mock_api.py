"""Quick demo script to test mock data functionality."""
import logging
from services.api_service import fetch_option_chain, fetch_historical_data

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("=== Testing Mock Data API Service ===\n")

# Test fetching option chain with mock data
print("1. Fetching mock option chain for AAPL...")
chain = fetch_option_chain("AAPL")
print(f"   Symbol: {chain.symbol}")
print(f"   Underlying Price: ${chain.underlyingPrice:.2f}")
print(f"   Volatility: {chain.volatility:.2%}")
print(f"   Number of options: {len(chain.options)}")
print(f"   Expiration dates: {len(chain.expiration_dates)}")
print()

# Test fetching historical data with mock data
print("2. Fetching mock historical data for AAPL...")
history = fetch_historical_data("AAPL")
df = history.to_dataframe()
print(f"   Number of candles: {len(df)}")
print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"   Latest close: ${df.iloc[-1]['close']:.2f}")
print()

print("SUCCESS: Mock data is working! You can now use the library without a TDAmeritrade API key.")
print("         All API calls will return realistic mock data for demonstration purposes.")
