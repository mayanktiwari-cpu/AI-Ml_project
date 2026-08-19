import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from .logger import logger
from .utils import load_config, load_model

def run_evaluation():
    config = load_config()
    model = load_model(config["model"]["model_save_path"])
    preprocessor = load_model("models/preprocessor.pkl")
    features = load_model("models/feature_names.pkl")

    df = pd.read_csv(config["dataset"]["processed_path"]) if os.path.exists(config["dataset"]["processed_path"]) else None
    if df is None:
        from .feature_engineering import RetailFeaturePipeline
        raw_df = pd.read_csv(config["dataset"]["raw_path"])
        df = RetailFeaturePipeline().transform(raw_df)

    split_idx = int(len(df) * (1 - config["model"]["test_size"]))
    X_test = df[features].iloc[split_idx:]
    y_test = df["Stockout_Occurred"].iloc[split_idx:]

    X_test_proc = preprocessor.transform(X_test)
    y_preds = model.predict(X_test_proc)
    y_probs = model.predict_proba(X_test_proc)[:, 1]

    # Metrics Report
    auc = float(roc_auc_score(y_test, y_probs))
    cm = confusion_matrix(y_test, y_preds).tolist()
    report = classification_report(y_test, y_preds, output_dict=True)

    metrics_payload = {"ROC_AUC": auc, "Confusion_Matrix": cm, "Classification_Report": report}
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=4)

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Stockout Risk ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png")
    plt.close()

    # SHAP Explainability
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_test_proc[:500])
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_test_proc[:500], feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png")
    plt.close()

    logger.info("Evaluation metrics and plots exported to reports/")

if __name__ == "__main__":
    run_evaluation()