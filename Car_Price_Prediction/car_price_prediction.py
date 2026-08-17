import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("car data.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# Remove missing values
df = df.dropna()

# Convert categorical columns into numbers
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Predict a sample car price
sample_prediction = model.predict(X_test.iloc[[0]])
print("\nPredicted Car Price:", round(sample_prediction[0], 2))

# Actual vs Predicted graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()