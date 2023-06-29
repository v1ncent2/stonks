import requests
import json
import numpy as np
from datetime import datetime, timedelta

# Alpha Vantage API key
api_key = 'PDXSLYTR1BCGV4A2'  # Replace with your actual API key

# Ticker symbols of the stocks
symbols = ['NVDA', 'AAPL', 'META', 'GOOGL', 'MSFT']

# Number of weeks of historical data to retrieve
weeks = 300

# Calculate the start and end dates
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')

# Initialize a matrix to store the weekly prices for each stock
weekly_prices_matrix = np.empty((weeks, len(symbols)))

# Initialize a vector to store the AAPL prices offset by 1 week
aapl_label_vector = np.empty(weeks-1)

# Iterate over the ticker symbols
for i, symbol in enumerate(symbols):
    # API URL for Alpha Vantage
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={api_key}'

    # Send a GET request to the API
    response = requests.get(url)

    # Parse the JSON response
    data = json.loads(response.text)

    # Extract the weekly stock prices
    weekly_prices = []

    if 'Weekly Time Series' in data:
        time_series = data['Weekly Time Series']

        # Iterate over the weeks and retrieve the closest available closing price
        for week in range(weeks + 4, 4, -1):
            # Calculate the date for the specific week
            date = (datetime.now() - timedelta(weeks=week)).strftime('%Y-%m-%d')

            # Check if the date is available in the time series data
            if date in time_series:
                # Retrieve the closing price for the date
                close_price = float(time_series[date]['4. close'])
                weekly_prices.append(close_price)
            else:
                # Find the closest available date by iterating forward
                current_date = datetime.strptime(date, '%Y-%m-%d')
                closest_date = None

                while not closest_date:
                    current_date += timedelta(days=1)
                    closest_date_str = current_date.strftime('%Y-%m-%d')
                    if closest_date_str in time_series:
                        closest_date = closest_date_str

                # Retrieve the closest available closing price
                close_price = float(time_series[closest_date]['4. close'])
                weekly_prices.append(close_price)
    else:
        print(f"Error: Unable to retrieve stock prices for {symbol}.")

    # Store the weekly prices in the matrix column
    weekly_prices_matrix[:, i] = weekly_prices
    
    # Store the AAPL prices offset by 1 week in the label vector
    if symbol == symbols[0]:
        aapl_label_vector = weekly_prices[1:]


# Separate the last row of the matrix into a separate vector
last_row_vector = weekly_prices_matrix[-1, :]

# Remove the last row from the weekly prices matrix
weekly_prices_matrix = np.delete(weekly_prices_matrix, -1, axis=0)

# Perform the regression calculation
X = weekly_prices_matrix

Y = aapl_label_vector

# Calculate the coefficients using the formula (X^T X)^{-1} X^{T}Y
X_transpose = np.transpose(X)
coefficients = np.linalg.inv(X_transpose @ X) @ X_transpose @ Y
Y_pred = np.dot(coefficients, last_row_vector)

# Print the matrix of weekly stock prices
print("Weekly Prices: ")
print(' '.join(symbols))
print(weekly_prices_matrix)

# Print the last row as a separate vector
print("\nThis week's prices:")
print(last_row_vector)

# Correlation Vector
print("\nCoefficients vector:")
print(coefficients)

# Print the prediction
print("\n Prediction for next week:")
print(Y_pred)

