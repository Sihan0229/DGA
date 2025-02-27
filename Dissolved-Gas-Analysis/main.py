import argparse
from utils import load_data, preprocess_data, feature_engineering, train_model

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model on DGA dataset")
    parser.add_argument('--file', type=str, required=True, help="Path to the dataset file")
    parser.add_argument('--method', type=str, choices=['original', 'three_ratios', 'diff_ratio'], default='original', help="Feature engineering method")
    parser.add_argument('--normalize', action='store_true', help="Whether to normalize the data")
    parser.add_argument('--model', type=str, choices=['knn', 'xgb', 'random_forest'], default='knn', help="Model to train")
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load data
    df = load_data(args.file)

    # Feature engineering
    df = feature_engineering(df, method=args.method)

    # Train model
    model, score = train_model(df, model_type=args.model, normalize=args.normalize)

    # Output results
    print(f"Model: {args.model}, Score: {score:.4f}")

if __name__ == "__main__":
    main()
