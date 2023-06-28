import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv('/content/house_data.csv')  # Update the file path if necessary

# Select relevant columns
selected_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                    'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
                    'yr_built', 'yr_renovated', 'price']

data = data[selected_columns]

# Convert categorical features using one-hot encoding
categorical_cols = ['waterfront', 'view', 'condition']
data_encoded = pd.get_dummies(data, columns=categorical_cols)

# Separate the features (X) and the target variable (y)
X = data_encoded.drop('price', axis=1)
y = data_encoded['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Plotting the predicted prices against the actual prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='b')
sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color='r', linestyle='--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted House Prices')
plt.tight_layout()
plt.show()
