"""
Stock Price Prediction Model Training Script
Trains 4 different models using real trading data:
1. Linear Regression
2. Random Forest
3. Gradient Boosting
4. Support Vector Regression
"""

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("🚀 TRAINING INVESTMENT PREDICTION MODELS WITH REAL DATA")
print("=" * 70)

# Step 1: Load real data
print("\n1️⃣ Loading real trading data...")
csv_path = 'investment_poc/Dataset/trade-summary-merged.csv'
df_raw = pd.read_csv(csv_path)

print(f"✅ Loaded {len(df_raw)} records from merged dataset")
print(f"   Columns: {list(df_raw.columns)}")
print(f"   Shape: {df_raw.shape}")

# Step 2: Data Preprocessing
print("\n2️⃣ Preprocessing data...")

# Handle the column name with special characters
if '**Last Trade (Rs.)' in df_raw.columns:
    df_raw.rename(columns={'**Last Trade (Rs.)': 'Last Trade (Rs.)'}, inplace=True)

# Remove rows with missing critical values
df_raw = df_raw.dropna(subset=['Last Trade (Rs.)', 'Change (%)'])

# Create numeric features
df_raw['Change (%)'] = pd.to_numeric(df_raw['Change (%)'], errors='coerce')
df_raw['Share Volume'] = pd.to_numeric(df_raw['Share Volume'], errors='coerce')
df_raw['Trade Volume'] = pd.to_numeric(df_raw['Trade Volume'], errors='coerce')
df_raw['Open (Rs.)'] = pd.to_numeric(df_raw['Open (Rs.)'], errors='coerce')
df_raw['High (Rs.)'] = pd.to_numeric(df_raw['High (Rs.)'], errors='coerce')
df_raw['Low (Rs.)'] = pd.to_numeric(df_raw['Low (Rs.)'], errors='coerce')

df = df_raw.dropna()

print(f"✅ After preprocessing: {len(df)} records")

# Step 3: Feature Engineering
print("\n3️⃣ Engineering features...")

# Create additional features
df['Price_Range'] = df['High (Rs.)'] - df['Low (Rs.)']
df['Price_Mid'] = (df['High (Rs.)'] + df['Low (Rs.)']) / 2
df['Volume_Ratio'] = df['Share Volume'] / (df['Trade Volume'] + 1)
df['Price_Movement'] = df['Last Trade (Rs.)'] - df['Open (Rs.)']

# Target variable: Next day price prediction (using price movement percentage as proxy)
# Since we only have one snapshot, we'll predict if price will go up or down based on features
df['Target'] = (df['Change (%)'] > 0).astype(int)  # Binary: 1 for increase, 0 for decrease
df['Price_Target'] = df['Last Trade (Rs.)']  # For regression

print(f"✅ Features created")

# Step 4: Prepare features for training
print("\n4️⃣ Preparing features for training...")

# Select features for modeling
feature_columns = [
    'Share Volume', 'Trade Volume', 'Open (Rs.)', 'High (Rs.)', 'Low (Rs.)',
    'Price_Range', 'Price_Mid', 'Volume_Ratio', 'Price_Movement'
]

X = df[feature_columns].values
y = df['Price_Target'].values  # Predict actual last trade price

print(f"✅ Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")
print(f"   Features: {feature_columns}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")

# Step 5: Train 4 Models
print("\n5️⃣ Training 4 different models...")

models = {}
results = {}

# Model 1: Linear Regression
print("\n   📊 Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
models['Linear Regression'] = lr_model
results['Linear Regression'] = {
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'R2': r2_score(y_test, y_pred_lr),
    'predictions': y_pred_lr
}

# Model 2: Random Forest
print("   📊 Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
models['Random Forest'] = rf_model
results['Random Forest'] = {
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'R2': r2_score(y_test, y_pred_rf),
    'predictions': y_pred_rf
}

# Model 3: Gradient Boosting
print("   📊 Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
models['Gradient Boosting'] = gb_model
results['Gradient Boosting'] = {
    'MAE': mean_absolute_error(y_test, y_pred_gb),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
    'R2': r2_score(y_test, y_pred_gb),
    'predictions': y_pred_gb
}

# Model 4: Support Vector Regression
print("   📊 Training Support Vector Regression...")
svr_model = SVR(kernel='rbf', C=100, gamma='scale')
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)
models['Support Vector Regression'] = svr_model
results['Support Vector Regression'] = {
    'MAE': mean_absolute_error(y_test, y_pred_svr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svr)),
    'R2': r2_score(y_test, y_pred_svr),
    'predictions': y_pred_svr
}

print("\n✅ All 4 models trained successfully!")

# Step 6: Model Comparison and Selection
print("\n6️⃣ Model Performance Comparison...")

comparison_df = pd.DataFrame({
    model_name: {
        'MAE': result['MAE'],
        'RMSE': result['RMSE'],
        'R² Score': result['R2']
    }
    for model_name, result in results.items()
}).round(4)

print("\n" + comparison_df.to_string())

# Select best model based on R² score
best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
best_model = models[best_model_name]
best_metrics = results[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {best_metrics['R2']:.4f}")
print(f"   MAE: {best_metrics['MAE']:.2f}")
print(f"   RMSE: {best_metrics['RMSE']:.2f}")

# Step 7: Create Visualizations
print("\n7️⃣ Creating visualizations...")

os.makedirs('investment_poc/visualizations', exist_ok=True)

# Visualization 1: Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

metrics = ['MAE', 'RMSE', 'R² Score']
for idx, metric in enumerate(metrics):
    values = [results[name][metric if metric != 'R² Score' else 'R2'] for name in results.keys()]
    axes[idx].bar(results.keys(), values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[idx].set_title(metric)
    axes[idx].set_ylabel(metric)
    axes[idx].tick_params(axis='x', rotation=45)
    for i, v in enumerate(values):
        axes[idx].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('investment_poc/visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: model_comparison.png")
plt.close()

# Visualization 2: Predictions vs Actual (Best Model)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Best Model ({best_model_name}) - Predictions vs Actual', fontsize=14, fontweight='bold')

# Scatter plot
axes[0].scatter(y_test, best_metrics['predictions'], alpha=0.6, s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price (Rs.)')
axes[0].set_ylabel('Predicted Price (Rs.)')
axes[0].set_title('Predictions vs Actual Values')
axes[0].grid(True, alpha=0.3)

# Residuals plot
residuals = y_test - best_metrics['predictions']
axes[1].scatter(best_metrics['predictions'], residuals, alpha=0.6, s=20)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Price (Rs.)')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('investment_poc/visualizations/best_model_performance.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: best_model_performance.png")
plt.close()

# Visualization 3: Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_columns[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('investment_poc/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: feature_importance.png")
    plt.close()

# Visualization 4: Distribution of Predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribution Analysis', fontsize=14, fontweight='bold')

axes[0].hist(y_test, bins=30, alpha=0.7, label='Actual', color='blue')
axes[0].hist(best_metrics['predictions'], bins=30, alpha=0.7, label='Predicted', color='orange')
axes[0].set_xlabel('Price (Rs.)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution: Actual vs Predicted')
axes[0].legend()

# Box plot comparison
data_for_box = [y_test, best_metrics['predictions']]
axes[1].boxplot(data_for_box, labels=['Actual', 'Predicted'])
axes[1].set_ylabel('Price (Rs.)')
axes[1].set_title('Price Distribution - Actual vs Predicted')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('investment_poc/visualizations/distribution_analysis.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: distribution_analysis.png")
plt.close()

# Step 8: Save Models and Metadata
print("\n8️⃣ Saving models and metadata...")

os.makedirs('investment_poc/models', exist_ok=True)

# Save best model
joblib.dump(best_model, 'investment_poc/models/best_model.pkl')
joblib.dump(scaler, 'investment_poc/models/scaler.pkl')
print(f"   ✅ Saved: best_model.pkl")

# Save all models for comparison
models_info = {name: {'type': type(model).__name__} for name, model in models.items()}
with open('investment_poc/models/all_models_info.json', 'w') as f:
    json.dump(models_info, f, indent=2)

# Save comprehensive metadata
metadata = {
    'model_type': best_model_name,
    'features': feature_columns,
    'r2_score': float(best_metrics['R2']),
    'mae': float(best_metrics['MAE']),
    'rmse': float(best_metrics['RMSE']),
    'training_date': datetime.now().isoformat(),
    'dataset_records': len(df),
    'train_set_size': len(X_train),
    'test_set_size': len(X_test),
    'model_comparison': {
        name: {
            'MAE': float(result['MAE']),
            'RMSE': float(result['RMSE']),
            'R2': float(result['R2'])
        }
        for name, result in results.items()
    },
    'data_source': 'trade-summary-merged.csv'
}

with open('investment_poc/models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✅ Saved: metadata.json")

# Step 9: Create Summary Report
print("\n9️⃣ Creating summary report...")

summary = f"""
{'='*70}
STOCK PRICE PREDICTION MODEL - TRAINING REPORT
{'='*70}

📊 DATASET INFORMATION
   Total Records: {len(df)}
   Features Used: {len(feature_columns)}
   Train/Test Split: {len(X_train)}/{len(X_test)}

🤖 MODELS TRAINED
   1. Linear Regression
   2. Random Forest
   3. Gradient Boosting
   4. Support Vector Regression

🏆 BEST MODEL: {best_model_name}
   R² Score:     {best_metrics['R2']:.4f}
   MAE:          {best_metrics['MAE']:.2f} Rs.
   RMSE:         {best_metrics['RMSE']:.2f} Rs.

📈 MODEL COMPARISON
{comparison_df.to_string()}

📁 OUTPUT FILES
   ✅ Models: ../models/best_model.pkl
   ✅ Scaler: ../models/scaler.pkl
   ✅ Metadata: ../models/metadata.json
   
📊 VISUALIZATIONS
   ✅ model_comparison.png
   ✅ best_model_performance.png
   ✅ feature_importance.png
   ✅ distribution_analysis.png

⏰ Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

with open('investment_poc/models/training_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(summary)

print("\n✅ MODEL TRAINING COMPLETE!")
print("=" * 70)
