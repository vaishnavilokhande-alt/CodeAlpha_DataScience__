import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("Unemployment in India.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Average Unemployment Rate
average_unemployment = df["Estimated Unemployment Rate (%)"].mean()

print("\nAverage Unemployment Rate:",
      round(average_unemployment, 2), "%")


# -------------------------------
# Unemployment Rate Over Time
# -------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df["Date"],
    df["Estimated Unemployment Rate (%)"]
)

plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -------------------------------
# State-wise Unemployment Rate
# -------------------------------

state_average = df.groupby("Region")[
    "Estimated Unemployment Rate (%)"
].mean().sort_values(ascending=False)

print("\nState-wise Average Unemployment Rate:")
print(state_average)


plt.figure(figsize=(12, 6))

state_average.plot(kind="bar")

plt.title("Average Unemployment Rate by State")
plt.xlabel("State")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# -------------------------------
# COVID-19 Impact Analysis
# -------------------------------

covid_data = df[
    (df["Date"] >= "2020-03-01") &
    (df["Date"] <= "2020-06-30")
]

print("\nCOVID-19 Period Data:")
print(covid_data.head())

covid_average = covid_data[
    "Estimated Unemployment Rate (%)"
].mean()

print("\nAverage Unemployment Rate During COVID-19:",
      round(covid_average, 2), "%")


# COVID-19 Unemployment Trend

plt.figure(figsize=(10, 5))

plt.plot(
    covid_data["Date"],
    covid_data["Estimated Unemployment Rate (%)"],
    marker="o"
)

plt.title("Unemployment Rate During COVID-19")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


print("\nTask 2 Analysis Completed Successfully!")