#!/usr/bin/env python3
"""
Quick script to explore coordinate data in the master dataset
"""

import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/data/MASTER_INTEGRATED_DATASET.csv", nrows=1000)

print("Dataset shape:", df.shape)
print("\nColumns:")
for i, col in enumerate(df.columns):
    print(f"{i:2d}: {col}")

print("\nLooking for coordinate columns...")
coord_cols = [col for col in df.columns if any(word in col.lower() for word in ['lat', 'lon', 'coord', 'x', 'y'])]
print("Coordinate columns found:", coord_cols)

for col in coord_cols:
    print(f"\n{col}:")
    print(f"  Type: {df[col].dtype}")
    print(f"  Non-null: {df[col].notna().sum()}/{len(df)}")
    if df[col].notna().sum() > 0:
        print(f"  Sample values: {df[col].dropna().head().tolist()}")

# Check if any columns have numeric values that could be coordinates
print("\nChecking for numeric columns that might be coordinates...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    values = df[col].dropna()
    if len(values) > 0:
        min_val, max_val = values.min(), values.max()
        # Check if values look like Johannesburg coordinates
        if (-35 <= min_val <= -20 and 20 <= max_val <= 35) or (-35 <= max_val <= -20 and 20 <= min_val <= 35):
            print(f"  {col}: range [{min_val:.6f}, {max_val:.6f}] - COULD BE COORDINATES")