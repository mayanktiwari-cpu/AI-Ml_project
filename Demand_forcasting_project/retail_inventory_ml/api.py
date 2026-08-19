import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .utils import load_config, load_model

app = FastAPI(title="Retail Inventory Stockout Risk API", version="1.0.0")

model = load_model("C:/Users/DELL/repo/Inventory_Optimization_Demand_forcasting/models/stockout_xgboost.pkl")
preprocessor = load_model("C:/Users/DELL/repo/Inventory_Optimization_Demand_forcasting/models/preprocessor.pkl")
features = load_model("C:/Users/DELL/repo/Inventory_Optimization_Demand_forcasting/models/feature_names.pkl")

class InferencePayload(BaseModel):
    Category: str
    Region: str
    Weather_Condition: str
    Seasonality: str
    Inventory_Level: float
    Units_Ordered: float
    Price: float
    Discount: float
    Competitor_Pricing: float
    Lag_Sales_1: float
    Lag_Sales_7: float
    Rolling_Mean_Sales_7: float

@app.post("/predict-stockout")
def predict(payload: InferencePayload):
    try:
        data = payload.dict()
        df = pd.DataFrame([{
            "Category": data["Category"],
            "Region": data["Region"],
            "Weather Condition": data["Weather_Condition"],
            "Seasonality": data["Seasonality"],
            "Inventory Level": data["Inventory_Level"],
            "Units Ordered": data["Units_Ordered"],
            "Price": data["Price"],
            "Discount": data["Discount"],
            "Competitor Pricing": data["Competitor_Pricing"],
            "Lag_Sales_1": data["Lag_Sales_1"],
            "Lag_Sales_7": data["Lag_Sales_7"],
            "Lag_Sales_14": data["Lag_Sales_7"],
            "Rolling_Mean_Sales_7": data["Rolling_Mean_Sales_7"],
            "Rolling_Mean_Sales_14": data["Rolling_Mean_Sales_7"],
            "Rolling_Std_Sales_7": 1.0,
            "Rolling_Std_Sales_14": 1.0,
            "DayOfWeek": 1,
            "Month": 5,
            "Quarter": 2,
            "Is_Weekend": 0,
            "Effective_Price": data["Price"] * (1 - data["Discount"] / 100.0),
            "Price_Diff_Competitor": data["Competitor_Pricing"] - (data["Price"] * (1 - data["Discount"] / 100.0)),
            "Price_Ratio_Competitor": (data["Price"] * (1 - data["Discount"] / 100.0)) / (data["Competitor_Pricing"] + 1e-5),
            "Inventory_To_Order_Ratio": data["Inventory_Level"] / (data["Units_Ordered"] + 1e-5),
            "Inventory_To_Sales_Ratio": data["Inventory_Level"] / (data["Lag_Sales_1"] + 1e-5)
        }])

        df = df[features]
        df_proc = preprocessor.transform(df)
        prob = float(model.predict_proba(df_proc)[:, 1][0])
        pred = int(prob >= 0.5)

        return {
            "stockout_risk_prediction": pred,
            "stockout_probability": round(prob, 4),
            "risk_level": "HIGH" if prob > 0.6 else ("MEDIUM" if prob > 0.3 else "LOW")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))