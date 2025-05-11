# =============================================
# HANDLING MISSING DATA FOR RETAIL SALES DATASET
# =============================================

# --------------------------
# 1. IMPORT LIBRARIES
# --------------------------
import pandas as pd  # For data handling
from sklearn.model_selection import train_test_split  # For splitting data
import os  # For managing file paths

# --------------------------
# 2. LOAD THE MISSING DATASET
# --------------------------
# Path to missing data CSV
data_path = os.path.join("data", "synthetic_retail_sales_missing.csv")

# Load dataset
try:
    df = pd.read_csv(data_path)
    print("🔍 Initial missing values per column:")
    print(df.isnull().sum())
except FileNotFoundError:
    raise FileNotFoundError(f"Missing data file not found at {data_path}")

# --------------------------
# 3. DROP ROWS WHERE TARGET IS MISSING
# --------------------------
# Target column is 'Returned' (1/0), we can't impute it — remove rows with missing target
df = df.dropna(subset=["Returned"])

# --------------------------
# 4. CLEAN THE DATA (Drop ID, Encode Target)
# --------------------------
df = df.drop("TransactionID", axis=1)
df["Returned"] = df["Returned"].map({"Yes": 1, "No": 0})

# --------------------------
# 5. HANDLE MISSING VALUES
# --------------------------

# Split into numerical and categorical columns
numeric_cols = ['UnitPrice', 'QuantitySold', 'TotalSale', 'CustomerAge']
categorical_cols = ['ProductCategory', 'StoreLocation', 'PaymentMethod']

print("\n🔧 Imputing numerical columns with mean values...")
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

print("🔧 Imputing categorical columns with mode values...")
for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f" - {col}: Filled with '{mode_val}'")

# --------------------------
# 6. FINAL CHECK: ALL MISSING DATA HANDLED?
# --------------------------
assert df.isnull().sum().sum() == 0, "❌ Error: Missing values still exist!"
print("\n✅ All missing values handled successfully!")

# --------------------------
# 7. SPLIT INTO FEATURES (X) AND TARGET (y)
# --------------------------
X = df.drop("Returned", axis=1)
y = df["Returned"]

# --------------------------
# 8. SPLIT INTO TRAIN/TEST SETS
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------
# 9. SAVE CLEANED & SPLIT DATA
# --------------------------
# Create folder if it doesn't exist
os.makedirs("preprocessing", exist_ok=True)

X_train.to_csv("preprocessing/X_train_missing.csv", index=False)
X_test.to_csv("preprocessing/X_test_missing.csv", index=False)
y_train.to_csv("preprocessing/y_train_missing.csv", index=False)
y_test.to_csv("preprocessing/y_test_missing.csv", index=False)

print("\n📦 Saved cleaned missing dataset:")
print("- preprocessing/X_train_missing.csv")
print("- preprocessing/X_test_missing.csv")
print("- preprocessing/y_train_missing.csv")
print("- preprocessing/y_test_missing.csv")

# --------------------------
# 10. DONE!
# --------------------------
print("\n🎉 Missing data preprocessing complete.")