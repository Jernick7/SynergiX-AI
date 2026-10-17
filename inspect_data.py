import os
import pandas as pd

# Check file existence
file_name = "Training_set.csv"
if not os.path.exists(file_name):
  print(f"❌ Error: '{file_name}' not found in current directory.")
  print(f"Current working path: {os.getcwd()}")
  exit()

# Load and inspect dataset
df = pd.read_csv(file_name)

print("=" * 50)
print(f"Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("=" * 50)
print("\nColumns and Data Types:")
print(df.dtypes)

print("\nMissing Values per Column:")
print(df.isnull().sum())

print("\nFirst row preview:")
print(df.head(1).T)