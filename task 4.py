import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "c:/Users/sk/Downloads/internship/movie_ratings.csv"

# Try tab separator first, fallback to comma
try:
    df = pd.read_csv(file_path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(file_path, sep=",")
except:
    df = pd.read_csv(file_path, sep=",")

# Step 2: Inspect columns
print("Columns detected:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

# Step 3: Preprocess data
# Convert Rating to numeric
df['Rating'] = pd.to_numeric(df['Rating'].astype(str).str.strip(), errors='coerce')
df = df.dropna(subset=['Rating'])

# For regression: user and movie IDs as numeric features
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
if 'Rating' not in numeric_cols:
    raise ValueError("Dataset must contain a numeric 'Rating' column")

features = [col for col in numeric_cols if col != 'Rating']  # userId, movieId
X = df[features]
y = df['Rating']

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict unseen ratings
y_pred = model.predict(X_test)

# Evaluate performance
print("\nMean Squared Error (MSE):", round(mean_squared_error(y_test, y_pred),2))
print("R² Score:", round(r2_score(y_test, y_pred),2))

# Sample predictions
comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print("\nSample predictions:")
print(comparison.head(10))

# Visualize predictions
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect line
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.show()
