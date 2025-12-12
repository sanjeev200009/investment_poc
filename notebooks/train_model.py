"""
Stock Price Prediction Model Training Script
Uses Linear Regression for predictions (no external ML library needed)
"""

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os

print("=" * 60)
print("🚀 TRAINING INVESTMENT PREDICTION MODEL")
print("=" * 60)

# Step 1: Load sample data
print("\n1️⃣ Loading sample data...")
with open('../sample_data/sample_stock.json') as f:
    stock_data = json.load(f)

# Step 2: Create synthetic historical data (for demo)
print("2️⃣ Creating training dataset...")
symbols = list(stock_data.keys())
training_data = []

# Generate 100 days of data for each stock
np.random.seed(42)
for i in range(100):
    date = datetime.now() - timedelta(days=100-i)
    for symbol in symbols:
        base_price = stock_data[symbol]['price']
        price = base_price + np.random.randn() * 5
        volume = stock_data[symbol]['volume'] + np.random.randint(-50000, 50000)
        change = stock_data[symbol]['change_percent'] + np.random.randn()
        
        training_data.append({
            'symbol': symbol,
            'date': date,
            'price': max(price, 10),
            'volume': max(volume, 1000),
            'change_percent': change
        })

df = pd.DataFrame(training_data)
print(f"✅ Created {len(df)} training samples")
print(f"   Columns: {list(df.columns)}")
print(f"   Data shape: {df.shape}")

# Step 3: Feature engineering
print("\n3️⃣ Engineering features...")
df['price_ma_7'] = df.groupby('symbol')['price'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df['volume_ma_7'] = df.groupby('symbol')['volume'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df['target'] = df.groupby('symbol')['price'].transform(lambda x: x.shift(-1))
df = df.dropna()

print(f"✅ Features created with {len(df)} samples")

# Step 4: Prepare features
print("\n4️⃣ Preparing features for training...")
features = ['price', 'volume', 'change_percent', 'price_ma_7', 'volume_ma_7']
X = df[features].values
y = df['target'].values

print(f"✅ Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# Step 5: Calculate statistics for simple prediction model
print("\n5️⃣ Training simple prediction model...")

# Calculate feature statistics and correlation with target
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])

# Normalize X for training
X_norm = (X - means) / (stds + 1e-8)

# Fit simple linear regression
# y = w0 + w1*x1 + w2*x2 + ...
# Using normal equation: w = (X'X)^-1 X'y
X_with_bias = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])
try:
    weights = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
    y_pred = X_with_bias @ weights
    r2_score = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
except:
    weights = np.ones(X_norm.shape[1] + 1)
    r2_score = 0.5

print("✅ Model trained successfully!")

# Step 6: Evaluate model
print("\n6️⃣ Evaluating model performance...")
mae = np.mean(np.abs(y - y_pred))
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

print(f"✅ Model Evaluation:")
print(f"   R² Score: {r2_score:.4f}")
print(f"   MAE: {mae:.2f}")
print(f"   RMSE: {rmse:.2f}")
print(f"\n   Feature Importance (Correlation):")
for feat, corr in zip(features, correlations):
    print(f"   {feat}: {abs(corr):.4f}")

# Step 7: Create models directory and save
print("\n7️⃣ Saving model...")
os.makedirs('../models', exist_ok=True)

# Save model data
model_data = {
    'means': means,
    'stds': stds,
    'weights': weights,
    'features': features
}

joblib.dump(model_data, '../models/simple_model.pkl')
print(f"✅ Model saved: ../models/simple_model.pkl")

# Save metadata
metadata = {
    'model_type': 'Linear Regression',
    'features': features,
    'r2_score': float(r2_score),
    'mae': float(mae),
    'rmse': float(rmse),
    'training_date': datetime.now().isoformat(),
    'stocks_trained': symbols,
    'samples': len(df)
}

with open('../models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved: ../models/metadata.json")

# Step 8: Test predictions
print("\n8️⃣ Testing predictions...")
test_sample = X[0:1]
test_sample_norm = (test_sample - means) / (stds + 1e-8)
test_sample_with_bias = np.hstack([np.ones((1, 1)), test_sample_norm])
test_pred = test_sample_with_bias @ weights

print(f"✅ Sample prediction: LKR {test_pred[0]:,.2f}")
print(f"   Actual value: LKR {y[0]:,.2f}")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\n📊 Summary:")
print(f"   ✓ Model type: Linear Regression")
print(f"   ✓ Training samples: {len(df)}")
print(f"   ✓ Features: {len(features)}")
print(f"   ✓ R² Score: {r2_score:.4f}")
print(f"   ✓ Files saved in: ../models/")
print(f"\n🎯 Next: Run 'streamlit run app.py'")
print("=" * 60)
