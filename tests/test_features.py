import pandas as pd
import sys
from pathlib import Path

# Add the directory containing feature_engineering.py to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "Demand_forcasting_project" / "retail_inventory_ml"
sys.path.append(str(SRC_DIR))


from feature_engineering import RetailFeaturePipeline

def test_feature_generation():
    df = pd.DataFrame([{
        "Date": "2023-01-01", "Store ID": "S001", "Product ID": "P001",
        "Category": "Electronics", "Region": "North", "Inventory Level": 10,
        "Units Sold": 15, "Units Ordered": 20, "Price": 100.0, "Discount": 10,
        "Weather Condition": "Sunny", "Promotion": 1, "Competitor Pricing": 95.0,
        "Seasonality": "Winter", "Epidemic": 0, "Demand": 20
    }])
    
    # Duplicate row to allow rolling operations
    df_multi = pd.concat([df, df], ignore_index=True)
    df_multi["Date"] = ["2023-01-01", "2023-01-02"]
    
    fe = RetailFeaturePipeline()
    transformed = fe.transform(df_multi)
    assert "Stockout_Occurred" in transformed.columns
    assert "Effective_Price" in transformed.columns