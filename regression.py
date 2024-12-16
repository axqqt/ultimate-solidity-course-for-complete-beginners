# Import required libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 1. Training data
sizes = np.array([500, 1000, 1500, 2000]).reshape(-1, 1)  # Feature: Size (sq ft)
prices = np.array([150000, 300000, 450000, 600000])       # Target: Price ($)

# 2. Create and train the model
model = LinearRegression()
model.fit(sizes, prices)

# 3. Make predictions
new_sizes = np.array([1200, 1700, 2500]).reshape(-1, 1)
predicted_prices = model.predict(new_sizes)


# plt.plot()


# Output predictions
for size, price in zip(new_sizes.flatten(), predicted_prices):
    print(f"Predicted price for {size} sq ft: ${price:.2f}")
