# Enhanced ML Model Training for 2025 Indian Car Resale Price Prediction

## Overview

This document describes the enhanced machine learning model training script (`train_enhanced_model_2025.py`) that predicts realistic used car resale prices in INR for the 2025 Indian market.

## Key Features

### 1. **Realistic Price Conversion**
The script converts ex-showroom prices to realistic used car resale prices based on:
- **Age-based depreciation**: Progressive depreciation rates (10% first year, 10-15% per year after)
- **Mileage impact**: Penalties for high mileage, bonuses for low mileage
- **Fuel type effects**: 
  - Petrol: Standard depreciation
  - Diesel: Faster depreciation after 8 years (due to emission norms)
  - CNG: Lower resale value
  - Electric: Higher value retention when new, faster depreciation when old
  - Hybrid: Better value retention
- **Brand reliability**: Popular brands (Maruti, Hyundai, Honda, Toyota, Tata, Mahindra) hold value better
- **Luxury brands**: Faster depreciation when older

### 2. **Feature Engineering**
Additional features created:
- `car_age`: Current year - manufacturing year
- `km_per_year`: Average kilometers driven per year
- `is_reliable_brand`: Binary flag for popular/reliable brands
- `is_luxury_brand`: Binary flag for luxury brands

### 3. **Log Transformation**
- Uses `log1p()` transformation for price prediction (handles wide price ranges better)
- Predictions are exponentiated back using `expm1()` for final output
- This improves model performance on price ranges from ₹50k to ₹50L+

### 4. **Model Selection**
Compares two models:
- **RandomForest**: 300 trees, max_depth=12, with regularization
- **GradientBoosting**: 300 estimators, learning_rate=0.03, with subsampling

Selects the model with lowest RMSE on test set.

### 5. **Evaluation Metrics**
- **MAE** (Mean Absolute Error): Average absolute deviation in INR
- **RMSE** (Root Mean Square Error): Penalizes larger errors more
- **R² Score**: Coefficient of determination
- **MAPE** (Mean Absolute Percentage Error): Percentage deviation from actual prices

**Target**: MAPE < 15% (10-15% average deviation)

## Usage

### Training the Model

```bash
cd "car price predictor\car price predictor"
python train_enhanced_model_2025.py
```

### Output Files

1. **model.pkl**: Trained model (can be GradientBoosting or RandomForest)
2. **price_stats.pkl**: Price statistics for validation

### Model Input Features

The model expects these features (in order):
1. `name`: Car model name
2. `company`: Car brand/company
3. `fuel_type`: Fuel type (Petrol, Diesel, CNG, Electric, Hybrid)
4. `year`: Manufacturing year
5. `kms_driven`: Total kilometers driven
6. `car_age`: Calculated as CURRENT_YEAR - year
7. `km_per_year`: Calculated as kms_driven / (car_age + 1)
8. `is_reliable_brand`: Binary (1 if popular brand, 0 otherwise)
9. `is_luxury_brand`: Binary (1 if luxury brand, 0 otherwise)

### Important Notes

1. **Log Transformation**: The model predicts on log scale. Predictions must be exponentiated:
   ```python
   predicted_log = model.predict(input_df)[0]
   predicted_price = np.expm1(predicted_log)  # Convert back from log
   ```

2. **Feature Engineering Required**: Before prediction, you must calculate:
   - `car_age = CURRENT_YEAR - year`
   - `km_per_year = kms_driven / (car_age + 1)`
   - `is_reliable_brand` (1 for: Maruti, Hyundai, Honda, Toyota, Tata, Mahindra)
   - `is_luxury_brand` (1 for: BMW, Mercedes-Benz, Audi, Jaguar, Land Rover, Volvo)

3. **Current Year**: The script uses `datetime.now().year` to get current year (2025, 2026, etc.)

## Model Performance

The model aims for:
- **MAPE < 15%**: Average percentage deviation from actual prices
- **Good R² Score**: > 0.85 on test set indicates good fit
- **Realistic Prices**: Predictions align with 2025 Indian market conditions

## Integration with Application

To use this model in `application.py`, you need to:

1. Load the model (already handled)
2. Create input DataFrame with all required features
3. Handle log transformation:
   ```python
   predicted_log = model.predict(input_df)[0]
   predicted_price = np.expm1(predicted_log)
   ```

## Dataset Requirements

The script works with:
- `cars_ds_final.csv` (preferred)
- `Cars Datasets 2025.csv`
- `Cleaned_Car_data.csv` (fallback)

Required columns:
- `Make` or `Company Names` → mapped to `company`
- `Model` or `Cars Names` → mapped to `name`
- `Ex-Showroom_Price` or `Cars Prices` → mapped to `exshowroom_price`
- `Fuel_Type` or `Fuel Types` → mapped to `fuel_type`

If `year` and `kms_driven` are missing, they are generated with realistic defaults.

