from sklearn.datasets import load_iris
import pandas as pd

# Load Iris Dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add Target Column
df["Species"] = iris.target

print(df.head())
# Dataset Information
print("\nDataset Information:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Count of Each Species
print("\nSpecies Count:")
print(df["Species"].value_counts())
import matplotlib.pyplot as plt

# Plot Sepal Length
plt.figure(figsize=(8,5))
plt.hist(df["sepal length (cm)"], bins=10)
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show(block=False)
plt.pause(3)
plt.close()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Input and Output
X = df.drop("Species", axis=1)
y = df["Species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN Model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Check Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")
# Predict a New Flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

species = ["Setosa", "Versicolor", "Virginica"]

print("\nPredicted Flower Species:", species[prediction[0]])