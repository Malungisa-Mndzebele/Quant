"""
Example: Real-time quote streaming using WebSocket

This example demonstrates how to use the WebSocket streaming functionality
to receive real-time quote updates for multiple symbols.
"""

import asyncio
from datetime import datetime
from services.market_data_service import MarketDataService, Quote


async def simple_streaming_example():
    """
    Simple example: Stream quotes and print them.
    """
    print("=== Simple Streaming Example ===\n")
    
    # Initialize service (requires valid Alpaca credentials)
    service = MarketDataService(
        api_key='YOUR_API_KEY',
        api_secret='YOUR_SECRET_KEY',
        paper=True
    )
    
    # Stream quotes for multiple symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    print(f"Starting stream for: {symbols}\n")
    
    try:
        # Stream quotes for 30 seconds
        async for quote in service.stream_quotes(symbols):
            print(f"{quote.timestamp.strftime('%H:%M:%S')} | "
                  f"{quote.symbol:6s} | "
                  f"Bid: ${quote.bid_price:8.2f} | "
                  f"Ask: ${quote.ask_price:8.2f} | "
                  f"Mid: ${quote.mid_price:8.2f}")
            
            # Stop after 30 seconds (for demo purposes)
            if (datetime.now() - quote.timestamp).total_seconds() > 30:
                break
    
    except KeyboardInterrupt:
        print("\nStream interrupted by user")
    
    finally:
        # Always disconnect when done
        await service.disconnect_stream()
        print("\nStream disconnected")


async def callback_streaming_example():
    """
    Advanced example: Stream quotes with custom callback.
    """
    print("\n=== Callback Streaming Example ===\n")
    
    # Track quote counts
    quote_counts = {}
    
    def on_quote_update(quote: Quote):
        """Custom callback for quote updates"""
        symbol = quote.symbol
        quote_counts[symbol] = quote_counts.get(symbol, 0) + 1
        
        # Print every 10th quote
        if quote_counts[symbol] % 10 == 0:
            print(f"Received {quote_counts[symbol]} quotes for {symbol}")
    
    # Initialize service
    service = MarketDataService(
        api_key='YOUR_API_KEY',
        api_secret='YOUR_SECRET_KEY',
        paper=True
    )
    
    symbols = ['AAPL', 'TSLA']
    print(f"Starting stream with callback for: {symbols}\n")
    
    try:
        # Create a task to monitor stream status
        async def monitor_status():
            while True:
                await asyncio.sleep(5)
                status = service.get_stream_status()
                print(f"\nStream Status: {status}")
        
        monitor_task = asyncio.create_task(monitor_status())
        
        # Stream with callback
        count = 0
        async for quote in service.stream_quotes(symbols, callback=on_quote_update):
            count += 1
            
            # Stop after 100 quotes
            if count >= 100:
                break
        
        # Cancel monitoring task
        monitor_task.cancel()
        
    except KeyboardInterrupt:
        print("\nStream interrupted by user")
    
    finally:
        await service.disconnect_stream()
        print(f"\nFinal quote counts: {quote_counts}")
        print("Stream disconnected")


async def multi_stream_example():
    """
    Advanced example: Multiple concurrent streams.
    """
    print("\n=== Multi-Stream Example ===\n")
    
    service = MarketDataService(
        api_key='YOUR_API_KEY',
        api_secret='YOUR_SECRET_KEY',
        paper=True
    )
    
    async def stream_tech_stocks():
        """Stream tech stocks"""
        async for quote in service.stream_quotes(['AAPL', 'GOOGL', 'MSFT']):
            print(f"[TECH] {quote.symbol}: ${quote.mid_price:.2f}")
            await asyncio.sleep(0.1)  # Throttle output
    
    async def stream_auto_stocks():
        """Stream auto stocks"""
        async for quote in service.stream_quotes(['TSLA', 'F', 'GM']):
            print(f"[AUTO] {quote.symbol}: ${quote.mid_price:.2f}")
            await asyncio.sleep(0.1)  # Throttle output
    
    try:
        # Run both streams concurrently
        await asyncio.gather(
            stream_tech_stocks(),
            stream_auto_stocks(),
            return_exceptions=True
        )
    
    except KeyboardInterrupt:
        print("\nStreams interrupted by user")
    
    finally:
        await service.disconnect_stream()
        print("\nAll streams disconnected")


async def error_handling_example():
    """
    Example: Handling connection errors and reconnection.
    """
    print("\n=== Error Handling Example ===\n")
    
    service = MarketDataService(
        api_key='YOUR_API_KEY',
        api_secret='YOUR_SECRET_KEY',
        paper=True
    )
    
    symbols = ['AAPL']
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            print(f"Attempt {retry_count + 1}/{max_retries}")
            
            async for quote in service.stream_quotes(symbols):
                print(f"{quote.symbol}: ${quote.mid_price:.2f}")
            
            # If we get here, stream ended normally
            break
            
        except Exception as e:
            retry_count += 1
            print(f"Error: {e}")
            
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print("Max retries reached, giving up")
        
        finally:
            try:
                await service.disconnect_stream()
            except:
                pass


def main():
    """
    Run examples (uncomment the one you want to try).
    
    Note: You need valid Alpaca API credentials to run these examples.
    Set your credentials in the .env file or pass them directly.
    """
    
    # Choose which example to run:
    
    # Example 1: Simple streaming
    asyncio.run(simple_streaming_example())
    
    # Example 2: Streaming with callbacks
    # asyncio.run(callback_streaming_example())
    
    # Example 3: Multiple concurrent streams
    # asyncio.run(multi_stream_example())
    
    # Example 4: Error handling
    # asyncio.run(error_handling_example())


if __name__ == '__main__':
    print("WebSocket Streaming Examples")
    print("=" * 50)
    print("\nNote: These examples require valid Alpaca API credentials.")
    print("Update the API keys in the code or set them in your .env file.\n")
    
    # Uncomment to run:
    # main()
    
    print("\nTo run an example, uncomment the main() call at the bottom of this file.")
