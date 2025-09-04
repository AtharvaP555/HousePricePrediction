from src.data_prep import load_and_prepare
from src.train import train_models
from src.evaluate import evaluate_models
from src.visualize import plot_results, plot_feature_importance

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare("../data/AmesHousing.csv")

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate
    results = evaluate_models(models, X_test, y_test)
    print(results)

    # Visualizations
    plot_results(results)
    plot_feature_importance(models["XGBoost"], X_train)

if __name__ == "__main__":
    main()
