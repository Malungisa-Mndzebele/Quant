import optlib.gbs as gbs
import logging

# Configure logging to see the output
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_demo():
    print("--- Optlib Demo ---\n")

    # Example 1: Black-Scholes Call Option
    # Stock Price (fs) = 100
    # Strike Price (x) = 100
    # Time to Expiration (t) = 1 year
    # Risk-free Rate (r) = 5%
    # Volatility (v) = 20%
    call_price = gbs.black_scholes(option_type='c', fs=100, x=100, t=1, r=0.05, v=0.20)
    print(f"Black-Scholes Call Option Price: {call_price[0]:.4f}")
    print(f"  Delta: {call_price[1]:.4f}")
    print(f"  Gamma: {call_price[2]:.4f}")
    print(f"  Theta: {call_price[3]:.4f}")
    print(f"  Vega:  {call_price[4]:.4f}")
    print(f"  Rho:   {call_price[5]:.4f}")
    print("-" * 30)

    # Example 2: Black-Scholes Put Option
    # Same parameters
    put_price = gbs.black_scholes(option_type='p', fs=100, x=100, t=1, r=0.05, v=0.20)
    print(f"Black-Scholes Put Option Price:  {put_price[0]:.4f}")
    print("-" * 30)

    # Example 3: Implied Volatility
    # We know the call price is approx 10.45 from Example 1. Let's see if we can back out the volatility.
    target_price = call_price[0]
    implied_vol = gbs.euro_implied_vol(option_type='c', fs=100, x=100, t=1, r=0.05, q=0, cp=target_price)
    print(f"Implied Volatility for Price {target_price:.4f}: {implied_vol:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    run_demo()
