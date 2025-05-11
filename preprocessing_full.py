import pandas as pd  
from sklearn.model_selection import train_test_split 
import os  # For handling file paths

data_path = os.path.join("data", "synthetic_retail_sales.csv")

try:
    df = pd.read_csv(data_path)
    print("✅ Data loaded successfully! First 3 rows:")
    print(df.head(3))  
except FileNotFoundError:
    print(f"❌ Error: File not found at {data_path}")
    print("Make sure:")
    print("- The file exists in the 'data' folder")
    print("- You're running the script from the PROJECT ROOT folder")
    exit()

print("\n🧹 Cleaning data...")

df = df.drop("TransactionID", axis=1)  

df["Returned"] = df["Returned"].map({"Yes": 1, "No": 0})
df["CalculatedTotal"] = df["UnitPrice"] * df["QuantitySold"]
if not df["TotalSale"].equals(df["CalculatedTotal"]):
    print("⚠️ Warning: Some TotalSale values don't match UnitPrice × QuantitySold!")
df = df.drop("CalculatedTotal", axis=1) 

X = df.drop("Returned", axis=1)

y = df["Returned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
              
    random_state=42,    
    stratify=y         
)
os.makedirs("preprocessing", exist_ok=True)

# Save all files
X_train.to_csv("preprocessing/X_train_full.csv", index=False)
X_test.to_csv("preprocessing/X_test_full.csv", index=False)
y_train.to_csv("preprocessing/y_train_full.csv", index=False)
y_test.to_csv("preprocessing/y_test_full.csv", index=False)

print("\n🎉 Preprocessing complete! Saved files:")
print("- preprocessing/X_train_full.csv")
print("- preprocessing/y_train_full.csv (target)")
print("- preprocessing/X_test_full.csv")
print("- preprocessing/y_test_full.csv (target)")

print(df)