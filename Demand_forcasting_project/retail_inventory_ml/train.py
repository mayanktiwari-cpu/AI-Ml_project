import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from .feature_engineering import RetailFeaturePipeline
from .preprocessing import build_preprocessing_pipeline
from .logger import logger
from .utils import load_config, save_model

def run_training():
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info("Loading raw sales data...")
    raw_df = pd.read_csv(config["dataset"]["raw_path"])
    
    logger.info("Applying feature engineering...")
    fe = RetailFeaturePipeline()
    df = fe.transform(raw_df)

    # Define Feature Sets
    cat_cols = ["Category", "Region", "Weather Condition", "Seasonality"]
    ignore_cols = ["Date", "Store ID", "Product ID", "Demand", "Units Sold", "Stockout_Occurred"]
    num_cols = [c for c in df.columns if c not in cat_cols + ignore_cols]
    
    X = df[num_cols + cat_cols]
    y = df["Stockout_Occurred"]

    # Time-based Train-Test Split (Last 20% by date)
    split_idx = int(len(df) * (1 - config["model"]["test_size"]))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # MLflow Tracking
    with mlflow.start_run(run_name="XGBoost_Hyperparameter_Tuning"):
        xgb = XGBClassifier(random_state=42, eval_metric="logloss")
        
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0]
        }

        search = RandomizedSearchCV(
            xgb, param_distributions=param_grid, n_iter=5, cv=3, scoring="f1", random_state=42
        )
        search.fit(X_train_proc, y_train)

        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)

        # Log & Save Pipeline Artifacts
        save_model(best_model, config["model"]["model_save_path"])
        save_model(preprocessor, "models/preprocessor.pkl")
        save_model(list(X.columns), "models/feature_names.pkl")
        
        mlflow.xgboost.log_model(best_model, "stockout_xgb_model")
        logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    run_training()