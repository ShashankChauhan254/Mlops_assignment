import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

file_path = 'BostonHousing.csv'
boston_housing_df = pd.read_csv(file_path)

print(boston_housing_df.head())

print("First 5 rows of the dataset:")
print(boston_housing_df.head())

print("\nDataset Information:")
print(boston_housing_df.info())

print("\nSummary Statistics:")
print(boston_housing_df.describe())

print("\nColumn Names:")
print(boston_housing_df.columns)

print("\nMissing Values:")
print(boston_housing_df.isnull().sum())

print("Summary Statistics:")
print(boston_housing_df.describe())

print("\nMissing Values:")
print(boston_housing_df.isnull().sum())

plt.figure(figsize=(10, 6))
plt.hist(boston_housing_df['medv'], bins=30)
plt.title('Distribution of Median Home Values (MEDV)')
plt.xlabel('Median Home Value in $1000s')
plt.ylabel('Frequency')
plt.show()

correlation_matrix = boston_housing_df.corr()

plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix of Boston Housing Dataset')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(boston_housing_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_heatmap.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='rm', y='medv', data=boston_housing_df)
plt.title('Relationship between Average Number of Rooms (RM) and Median Home Value (MEDV)')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Home Value ($1000s)')
plt.savefig('rm_vs_medv.png')
plt.show()

from sklearn.model_selection import train_test_split

X = boston_housing_df.drop('medv', axis=1)
y = boston_housing_df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared (R2):", r2)

y_pred = model.predict(X_test)

print("Predicted values:")
print(y_pred[:5])

print("Actual values:")
print(y_test[:5])

diff = y_test - y_pred
print("Difference between predicted and actual values:")
print(diff[:5])

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Predicted vs Actual Values')
plt.show()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

medae = median_absolute_error(y_test, y_pred)
print("Median Absolute Error (MedAE):", medae)

r2 = r2_score(y_test, y_pred)
print("R-squared (R2):", r2)

plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.savefig('residuals_distribution.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_test['rm'], y_test, color='blue', label='Actual Prices')
plt.plot(X_test['rm'], y_pred, color='red', label='Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Average Number of Rooms (rm)')
plt.ylabel('House Price ($1000s)')
plt.legend()
plt.savefig('actual_vs_predicted_prices.png')
plt.show()