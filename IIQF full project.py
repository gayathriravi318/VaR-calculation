#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install yfinance


# In[7]:


import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm



def calculate_parametric_var(start_date, end_date, weights, confidence_level=0.99):
    # Tickers and Weights
    tickers = ['DLF.NS', 'NTPC.NS', 'HDFCBANK.NS']
    
    # Download historical data for the given tickers
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Calculate portfolio variance and standard deviation
    portfolio_variance = np.dot(weights.T, np.dot(returns.cov(), weights))
    portfolio_std = np.sqrt(portfolio_variance)
    
    # Calculate Z-score for the given confidence level
    z_score = norm.ppf(1 - confidence_level)
    
    # Calculate the Parametric VaR
    var = z_score * portfolio_std * np.sqrt(len(portfolio_returns))
    
    return var

# Define the weights for DLF, NTPC, and HDFC Bank
weights = np.array([0.40, 0.20, 0.40])

# Define the start and end date for the data
start_date = '2017-09-14'  # 500 trading days before October 1, 2019
end_date = '2019-09-30'

# Calculate the Parametric VaR
var_99 = calculate_parametric_var(start_date, end_date, weights, confidence_level=0.99)
print(f"The 99% Parametric VaR of the portfolio is: {var_99:.2f}")


# In[8]:


import numpy as np
import yfinance as yf

def calculate_historical_var(start_date, end_date, weights, confidence_level=0.99):
    # Tickers and Weights
    tickers = ['DLF.NS', 'NTPC.NS', 'HDFCBANK.NS']
    
    # Download historical data for the given tickers
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Sort portfolio returns
    sorted_returns = np.sort(portfolio_returns)
    
    # Calculate the VaR using historical simulation
    var_index = int((1 - confidence_level) * len(sorted_returns))
    var = -sorted_returns[var_index]  # VaR is a positive number
    
    return var

# Define the weights for DLF, NTPC, and HDFC Bank
weights = np.array([0.40, 0.20, 0.40])

# Define the start and end date for the data
start_date = '2017-09-14'  # 500 trading days before October 1, 2019
end_date = '2019-09-30'

# Calculate the Historical Simulation VaR
var_99_historical = calculate_historical_var(start_date, end_date, weights, confidence_level=0.99)
print(f"The 99% Historical Simulation VaR of the portfolio is: {var_99_historical:.2f}")


# In[9]:


import numpy as np
import yfinance as yf

def monte_carlo_option_pricing(S, K, T, r, sigma, option_type='call', simulations=10000):
    """
    Calculate the price of a European option using Monte Carlo simulation.

    Parameters:
    S : float : Current price of the underlying asset (Nifty)
    K : float : Strike price of the option
    T : float : Time to maturity in years
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying asset (annualized)
    option_type : str : 'call' for call option, 'put' for put option
    simulations : int : Number of simulations to run

    Returns:
    option_price : float : The simulated option price
    """
    # Generate random price paths using Geometric Brownian Motion (GBM)
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate payoffs for call or put option
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    
    # Calculate the present value of the expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price

# Parameters for Nifty Option
S = yf.download('^NSEI', end='2023-09-01')['Adj Close'][-1]  # Current price of Nifty
K = 19500  # Example strike price
T = 1/12  # Time to maturity (1 month = 1/12 year)
r = 0.05  # Risk-free interest rate (5%)
sigma = 0.18  # Example annualized volatility (you might need to estimate this based on historical data)

# Calculate the option price using Monte Carlo Simulation
simulated_call_price = monte_carlo_option_pricing(S, K, T, r, sigma, option_type='call', simulations=10000)
simulated_put_price = monte_carlo_option_pricing(S, K, T, r, sigma, option_type='put', simulations=10000)

print(f"Simulated Call Option Price: {simulated_call_price:.2f}")
print(f"Simulated Put Option Price: {simulated_put_price:.2f}")

# Fetch the current market prices for comparison (assumed example)
market_call_price = 320.00  # Replace with actual market data
market_put_price = 200.00   # Replace with actual market data

# Decision making
if simulated_call_price > market_call_price:
    print("The Call Option is undervalued in the market. Consider buying.")
else:
    print("The Call Option is overvalued in the market. Consider selling.")

if simulated_put_price > market_put_price:
    print("The Put Option is undervalued in the market. Consider buying.")
else:
    print("The Put Option is overvalued in the market. Consider selling.")


# In[10]:


import numpy as np
import matplotlib.pyplot as plt

def bond_price(face_value, coupon_rate, yield_curve, maturity, payment_frequency=1):
    """
    Calculate the price of a bond given a yield curve.

    face_value : float : The face value of the bond.
    coupon_rate : float : The annual coupon rate.
    yield_curve : float : The yield for a given maturity.
    maturity : int : The bond's maturity in years.
    payment_frequency : int : Number of coupon payments per year.

    Returns:
    bond_price : float : The price of the bond.
    """
    coupon_payment = (coupon_rate / payment_frequency) * face_value
    price = 0
    
    for t in range(1, maturity * payment_frequency + 1):
        discount_factor = 1 / (1 + yield_curve / payment_frequency) ** t
        price += coupon_payment * discount_factor

    # Add the face value's present value
    price += face_value / (1 + yield_curve / payment_frequency) ** (maturity * payment_frequency)
    
    return price

def dv01(face_value, coupon_rate, yield_curve, maturity, shift=0.0001, payment_frequency=1):
    """
    Calculate the DV01 for a bond.
    
    face_value : float : The face value of the bond.
    coupon_rate : float : The annual coupon rate.
    yield_curve : float : The yield for a given maturity.
    maturity : int : The bond's maturity in years.
    shift : float : The interest rate shift in decimal (1 basis point = 0.0001).
    payment_frequency : int : Number of coupon payments per year.

    Returns:
    dv01_value : float : The DV01 of the bond.
    """
    price_original = bond_price(face_value, coupon_rate, yield_curve, maturity, payment_frequency)
    price_shifted = bond_price(face_value, coupon_rate, yield_curve + shift, maturity, payment_frequency)
    
    dv01_value = price_original - price_shifted
    return dv01_value

def plot_dv01_curve(face_value, coupon_rate, yield_curve, maturity, put_date, payment_frequency=1):
    """
    Plot the DV01 curve for a puttable bond.
    
    face_value : float : The face value of the bond.
    coupon_rate : float : The annual coupon rate.
    yield_curve : float : The yield for a given maturity.
    maturity : int : The bond's maturity in years.
    put_date : int : The put date in years.
    payment_frequency : int : Number of coupon payments per year.
    """
    dv01_values = []
    years = np.arange(1, maturity + 1)
    
    for year in years:
        if year <= put_date:
            # Bond is valued with maturity up to the put date
            dv01_values.append(dv01(face_value, coupon_rate, yield_curve, year, payment_frequency=payment_frequency))
        else:
            # Beyond the put date, maturity is capped at the put date
            dv01_values.append(dv01(face_value, coupon_rate, yield_curve, put_date, payment_frequency=payment_frequency))
    
    plt.plot(years, dv01_values, label='DV01 Curve for Puttable Bond')
    plt.axvline(x=put_date, color='r', linestyle='--', label='Put Date')
    plt.title('DV01 Curve for Puttable Bond')
    plt.xlabel('Years to Maturity')
    plt.ylabel('DV01')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example parameters for the puttable bond
face_value = 1000
coupon_rate = 0.05  # 5% annual coupon rate
yield_curve = 0.03  # 3% yield (can be adjusted for a real LIBOR curve)
maturity = 30  # 30 years maturity
put_date = 15  # Put option at 15 years

# Plot the DV01 curve
plot_dv01_curve(face_value, coupon_rate, yield_curve, maturity, put_date)


# In[ ]:




