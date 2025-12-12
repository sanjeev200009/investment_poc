@echo off
REM Complete Project Execution Script for Windows

echo ================================
echo AI Investment Assistant - Setup
echo ================================
echo.

REM Step 1: Verify Python Environment
echo Step 1: Checking Python environment...
python --version
echo OK - Python is available
echo.

REM Step 2: Check dataset
echo Step 2: Verifying merged dataset...
if exist "investment_poc\Dataset\trade-summary-merged.csv" (
    echo OK - Dataset found: investment_poc\Dataset\trade-summary-merged.csv
) else (
    echo ERROR - Dataset not found!
    exit /b 1
)
echo.

REM Step 3: Navigate to project
echo Step 3: Changing to project directory...
cd investment_poc
echo OK - In investment_poc directory
echo.

REM Step 4: Check trained models
echo Step 4: Checking trained models...
if exist "models\best_model.pkl" (
    if exist "models\scaler.pkl" (
        echo OK - Trained models found:
        echo   - best_model.pkl
        echo   - scaler.pkl
        echo   - metadata.json
    ) else (
        echo INFO - Running training script...
        python notebooks\train_model.py
        echo OK - Training complete!
    )
) else (
    echo INFO - Running training script...
    python notebooks\train_model.py
    echo OK - Training complete!
)
echo.

REM Step 5: Check visualizations
echo Step 5: Checking visualizations...
if exist "visualizations\model_comparison.png" (
    echo OK - Visualization files found:
    dir visualizations\*.png
) else (
    echo INFO - Visualizations not found. They will be generated during app run.
)
echo.

REM Step 6: Verify requirements
echo Step 6: Checking required packages...
python -c "import streamlit, pandas, numpy, sklearn, joblib; print('OK - All packages installed')"
echo.

REM Step 7: Display model performance
echo Step 7: Model Performance Summary
echo =================================
python -c "
import json
try:
    with open('models\metadata.json') as f:
        data = json.load(f)
        print(f'Best Model: {data[\"model_type\"]}')
        print(f'R Score: {data[\"r2_score\"]:.4f}')
        print(f'MAE: {data[\"mae\"]:.4f} Rs.')
        print(f'RMSE: {data[\"rmse\"]:.4f} Rs.')
        print(f'Dataset Records: {data[\"dataset_records\"]}')
        print()
        print('Model Comparison:')
        for model, metrics in data['model_comparison'].items():
            print(f'  {model}: R={metrics[\"R2\"]:.4f}')
except:
    print('Model metadata not available yet.')
"
echo.

REM Step 8: Launch Streamlit
echo Step 8: Starting Streamlit Application...
echo =================================
echo.
echo Web app will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run app.py

pause
