import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

def plot_knn_importance(X_train, y_train, X_test, y_test, result_dir="result"):
    """Generate and save KNN feature importance plot."""
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(X_train, y_train)

    result = permutation_importance(classifier, X_test, y_test, n_repeats=10, random_state=42)

    sorted_idx = result.importances_mean.argsort()

    # Plot permutation importance
    plt.figure(figsize=(10, 6))
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
    plt.title("KNN Permutation Importance (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "knn_feature_importance.png"))
    plt.close()

    # Print feature importance
    for i in sorted_idx:
        print(f"{X_train.columns[i]}: {result.importances_mean[i]:.4f}")

def plot_xgb_importance(X_train, y_train, result_dir="result"):
    """Generate and save XGB feature importance plot."""
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # Get feature importance
    importance = xgb_model.feature_importances_

    # Plot feature importance
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(importance)), importance)
    plt.xticks(range(len(importance)), X_train.columns, rotation='vertical')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("XGB Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "xgb_feature_importance.png"))
    plt.close()

    # Print feature importance
    for i, feature in enumerate(X_train.columns):
        print(f"{feature}: {importance[i]:.4f}")

def plot_rf_importance(X_train, y_train, X_test, y_test, result_dir="result"):
    """Generate and save Random Forest feature importance plot."""
    clf = RandomForestClassifier(max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

    # Get feature importance
    importance = clf.feature_importances_

    # Plot feature importance
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(importance)), importance)
    plt.xticks(range(len(importance)), X_train.columns, rotation='vertical')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "rf_feature_importance.png"))
    plt.close()

    # Print feature importance
    for i, feature in enumerate(X_train.columns):
        print(f"{feature}: {importance[i]:.4f}")

def generate_feature_importance_plots(X_train, y_train, X_test, y_test, result_dir="result"):
    """Generate and save feature importance plots for KNN, XGB, and Random Forest models."""
    # Ensure the result directory exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # KNN feature importance
    plot_knn_importance(X_train, y_train, X_test, y_test, result_dir)

    # XGB feature importance
    plot_xgb_importance(X_train, y_train, result_dir)

    # Random Forest feature importance
    plot_rf_importance(X_train, y_train, X_test, y_test, result_dir)
