#!/bin/bash

# Set default values for arguments
FILE_PATH="datasets/DGA_data_pre.csv"
FEATURE_METHOD="original"
NORMALIZE=false
MODEL="knn"

# Parse arguments
while getopts f:m:ns opt; do
  case $opt in
    f) FILE_PATH="$OPTARG" ;;  # Path to the data file
    m) FEATURE_METHOD="$OPTARG" ;;  # Feature engineering method (original, three_ratios, diff_ratio)
    n) NORMALIZE=true ;;  # Apply normalization
    s) MODEL="xgb" ;;  # Specify model as XGBoost
    *) echo "Usage: $0 [-f file_path] [-m feature_method] [-n] [-s model]"; exit 1 ;;
  esac
done

# Normalize flag
if [ "$NORMALIZE" = true ]; then
  NORMALIZE_FLAG="--normalize"
else
  NORMALIZE_FLAG=""
fi

# Run the Python script
python3 main.py --file "$FILE_PATH" --method "$FEATURE_METHOD" $NORMALIZE_FLAG --model "$MODEL"
