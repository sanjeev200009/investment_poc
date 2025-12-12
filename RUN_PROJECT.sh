#!/bin/bash
# Complete Project Execution Script

echo "================================"
echo "AI Investment Assistant - Setup"
echo "================================"
echo ""

# Step 1: Verify Python Environment
echo "Step 1: Checking Python environment..."
python --version
echo "✓ Python is available"
echo ""

# Step 2: Navigate to project directory
echo "Step 2: Changing to project directory..."
cd investment_poc
echo "✓ In investment_poc directory"
echo ""

# Step 3: Check dataset
echo "Step 3: Verifying merged dataset..."
if [ -f "Dataset/trade-summary-merged.csv" ]; then
    echo "✓ Dataset found: Dataset/trade-summary-merged.csv"
    wc -l Dataset/trade-summary-merged.csv | awk '{print "  Records: " $1}'
else
    echo "✗ Dataset not found!"
    exit 1
fi
echo ""

# Step 4: Check trained models
echo "Step 4: Checking trained models..."
if [ -f "models/best_model.pkl" ] && [ -f "models/scaler.pkl" ]; then
    echo "✓ Trained models found:"
    echo "  - best_model.pkl"
    echo "  - scaler.pkl"
    echo "  - metadata.json"
else
    echo "! Models not found. Running training script..."
    python notebooks/train_model.py
    echo "✓ Training complete!"
fi
echo ""

# Step 5: Check visualizations
echo "Step 5: Checking visualizations..."
ls visualizations/*.png 2>/dev/null && echo "✓ Visualization files found:" || echo "! No visualizations found"
ls -1 visualizations/*.png 2>/dev/null | sed 's/^/  - /'
echo ""

# Step 6: Verify requirements
echo "Step 6: Checking required packages..."
python -c "import streamlit, pandas, numpy, sklearn, joblib; print('✓ All packages installed')"
echo ""

# Step 7: Display model performance
echo "Step 7: Model Performance Summary"
echo "================================="
python -c "
import json
with open('models/metadata.json') as f:
    data = json.load(f)
    print(f'Best Model: {data[\"model_type\"]}')
    print(f'R² Score: {data[\"r2_score\"]:.4f}')
    print(f'MAE: {data[\"mae\"]:.4f} Rs.')
    print(f'RMSE: {data[\"rmse\"]:.4f} Rs.')
    print(f'Dataset Records: {data[\"dataset_records\"]}')
    print()
    print('Model Comparison:')
    for model, metrics in data['model_comparison'].items():
        print(f'  {model}: R²={metrics[\"R2\"]:.4f}')
"
echo ""

# Step 8: Launch Streamlit
echo "Step 8: Starting Streamlit Application..."
echo "================================="
echo ""
echo "🚀 Web app will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run app.py
