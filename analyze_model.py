import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the trained model
model = joblib.load("california_model.pkl")

print("=" * 80)
print("COMPREHENSIVE MODEL ANALYSIS - CALIFORNIA HOUSING PRICE PREDICTOR")
print("=" * 80)

# 1. MODEL TYPE AND BASIC INFO
print("\n1. MODEL TYPE AND ARCHITECTURE:")
print(f"   - Model Class: {type(model).__name__}")
print(f"   - Module: {type(model).__module__}")
print(f"   - Model Description: Linear Regression for California Housing Prices")

# 2. MODEL PARAMETERS (HYPERPARAMETERS)
print("\n2. MODEL HYPERPARAMETERS:")
print(f"   - fit_intercept: {model.fit_intercept}")
print(f"   - copy_X: {model.copy_X}")
print(f"   - n_jobs: {model.n_jobs}")
print(f"   - positive: {model.positive}")

# 3. TRAINED WEIGHTS (COEFFICIENTS)
print("\n3. TRAINED WEIGHTS (COEFFICIENTS):")
feature_names = [
    "MedInc (Median Income)",
    "HouseAge (House Age)",
    "AveRooms (Average Rooms)",
    "AveBedrms (Average Bedrooms)",
    "Population (Population)",
    "AveOccup (Average Occupancy)",
    "Latitude (Latitude)",
    "Longitude (Longitude)"
]

for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
    print(f"   {i+1}. {name}")
    print(f"      - Coefficient Value: {coef:.10f}")
    print(f"      - Scientific Notation: {coef:.4e}")

# 4. INTERCEPT
print("\n4. INTERCEPT (BIAS TERM):")
print(f"   - Intercept Value: {model.intercept_:.10f}")
print(f"   - Scientific Notation: {model.intercept_:.4e}")

# 5. TOTAL NUMBER OF PARAMETERS
print("\n5. TOTAL PARAMETERS:")
total_params = len(model.coef_) + 1  # coefficients + intercept
print(f"   - Number of Weights (Coefficients): {len(model.coef_)}")
print(f"   - Number of Bias Terms (Intercept): 1")
print(f"   - Total Parameters: {total_params}")

# 6. PARAMETER STATISTICS
print("\n6. COEFFICIENT STATISTICS:")
print(f"   - Maximum Coefficient: {np.max(model.coef_):.10f}")
print(f"   - Minimum Coefficient: {np.min(model.coef_):.10f}")
print(f"   - Mean Coefficient: {np.mean(model.coef_):.10f}")
print(f"   - Std Dev of Coefficients: {np.std(model.coef_):.10f}")
print(f"   - Sum of Absolute Coefficients: {np.sum(np.abs(model.coef_)):.10f}")

# 7. MODEL ATTRIBUTES AND STATE
print("\n7. MODEL ATTRIBUTES:")
print(f"   - Number of Features (n_features_in_): {model.n_features_in_}")
print(f"   - Feature Names (in order):")
for i, name in enumerate(feature_names, 1):
    print(f"      {i}. {name}")

# 8. PREDICTION FORMULA
print("\n8. PREDICTION FORMULA:")
print("   Y (House Price) = Intercept + (Coef₁ × MedInc) + (Coef₂ × HouseAge) +")
print("                     (Coef₃ × AveRooms) + (Coef₄ × AveBedrms) +")
print("                     (Coef₅ × Population) + (Coef₆ × AveOccup) +")
print("                     (Coef₇ × Latitude) + (Coef₈ × Longitude)")

# 9. SAMPLE PREDICTION
print("\n9. SAMPLE PREDICTION:")
sample_input = np.array([[8.3252, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]])
sample_prediction = model.predict(sample_input)[0]
print(f"   Sample Input: MedInc=8.3252, HouseAge=41.0, AveRooms=6.98,")
print(f"                 AveBedrms=1.02, Population=322.0, AveOccup=2.55,")
print(f"                 Latitude=37.88, Longitude=-122.23")
print(f"   Predicted House Price: ${sample_prediction:.2f} (in hundred thousands)")

# 10. TRAINED DATA INSIGHTS
print("\n10. TRAINING INSIGHTS:")
print("   - Training Algorithm: Ordinary Least Squares (OLS)")
print("   - Model Type: Simple Linear Regression (No feature scaling in model)")
print("   - Loss Function: Mean Squared Error (MSE)")
print("   - Output: Continuous values (House prices in hundred thousands)")

print("\n" + "=" * 80)
print("END OF MODEL ANALYSIS")
print("=" * 80)
