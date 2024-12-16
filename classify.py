# Import required libraries
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# 1. Training data: Sizes and corresponding price categories (classes)
sizes = np.array([500, 1000, 1500, 2000]).reshape(-1, 1)  # Feature: Size (sq ft)
price_categories = np.array([0, 0, 1, 1])  # Target: 0 for <300k, 1 for >=300k

# 2. Create and train the logistic regression model
model = LogisticRegression()
model.fit(sizes, price_categories)

# 3. Make predictions for new house sizes
new_sizes = np.array([1200, 1700, 2500]).reshape(-1, 1)
predicted_categories = model.predict(new_sizes)

# 4. Visualize the results
plt.scatter(sizes, price_categories, color="blue", label="Training Data")
plt.plot(new_sizes, predicted_categories, color="red", label="Predictions", marker="o")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price Category (0: <300k, 1: >=300k)")
plt.title("House Size vs Price Category")
plt.legend()
plt.show()

# Output predictions
for size, category in zip(new_sizes.flatten(), predicted_categories):
    category_label = "Below $300k" if category == 0 else "$300k or Above"
    print(f"Predicted category for {size} sq ft: {category_label}")
