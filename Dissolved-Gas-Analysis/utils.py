import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_data(file_path):
    """Loads the dataset from the given file path."""
    df = pd.read_csv(file_path)
    # Ensure correct data types
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.reset_index(drop=True)
    return df

def preprocess_data(df, normalize=False):
    """Preprocess data (e.g., normalization, handling missing values)."""
    if normalize:
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df

def feature_engineering(df, method="original"):
    """Create features based on the chosen method."""
    if method == "original":
        return df  # use original features
    elif method == "three_ratios":
        # Example of creating new features from ratios
        df['ch4/h2'] = df['ch4'] / (df['h2'] + 1e-6)  # Ratio 1: CH4 / H2
        df['c2h6/ch4'] = df['c2h6'] / (df['ch4'] + 1e-6)  # Ratio 2: C2H6 / CH4
        df['c2h4/c2h6'] = df['c2h4'] / (df['c2h6'] + 1e-6)  # Ratio 3: C2H4 / C2H6
        df['c2h2/c2h4'] = df['c2h2'] / (df['c2h4'] + 1e-6)  # Ratio 4: C2H2 / C2H4
        return df
    elif method == "diff_ratio":
        # Example of creating differences and ratios
        df['ch4_c2h4_diff'] = df['ch4'] - df['c2h4']
        return df
    else:
        raise ValueError("Unknown feature method!")

def train_model(df, model_type='knn', normalize=False):
    """Train a model based on the selected method (KNN, XGB, Random Forest)."""
    X = df.drop(columns='act')  # Features
    y = df['act']  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if normalize:
        X_train = preprocess_data(X_train, normalize=True)
        X_test = preprocess_data(X_test, normalize=True)
    
    if model_type == 'knn':
        model = KNeighborsClassifier()
    elif model_type == 'xgb':
        model = XGBClassifier()
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    else:
        raise ValueError("Unknown model type!")

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score
