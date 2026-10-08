import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Project folder
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# File paths
DATASET_PATH = os.path.join(
    BASE_DIR, "ml", "datasets", "Crop_recommendation.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")

print("Loading dataset...")

# Load dataset
data = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# Input features
features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X = data[features]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Crop Recommendation model...")

# Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Create model folder
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)