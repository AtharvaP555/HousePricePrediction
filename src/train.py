from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_models(X_train, y_train):
    """
    Train multiple regression models and return them as a dictionary.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=200)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models