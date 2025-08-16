from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# --- Load model ---
MODEL_PATH = "dynamic_price_model.pkl"
COLUMNS_PATH = "model_column.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
    raise FileNotFoundError("Model files not found in project folder!")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(COLUMNS_PATH)

# --- Create FastAPI app ---
app = FastAPI(title="Dynamic Pricing API")

# --- Define request format ---
class PriceRequest(BaseModel):
    costing: float
    demand_level: str
    stock_availability: str
    installation_cost: float
    customer_type: str
    est_competitor_cost: float
    comp_markup: float = 25.0
    our_min_margin: float = 15.0

# --- Test route ---
@app.get("/")
def home():
    return {"message": "Dynamic Pricing API is running"}

# --- Prediction route ---
@app.post("/predict")
def predict_price(req: PriceRequest):
    df = pd.DataFrame([{
        "costing": req.costing,
        "Demand_Level": req.demand_level,
        "Stock_Availability": req.stock_availability,
        "Installation_Cost": req.installation_cost,
        "Customer_Type": req.customer_type,
        "est_competitor_cost": req.est_competitor_cost
    }])

    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    base_price = model.predict(df_encoded)[0]
    competitor_price = req.est_competitor_cost * (1 + req.comp_markup / 100)
    our_min_price = req.costing * (1 + req.our_min_margin / 100)
    competitive_price = min(base_price, competitor_price * 0.95)
    competitive_price = max(competitive_price, our_min_price)
    competitive_margin = (competitive_price - req.costing) / competitive_price * 100

    return {
        "base_price": round(float(base_price), 2),
        "recommended_price": round(float(competitive_price), 2),
        "profit_margin_percent": round(float(competitive_margin), 2),
        "competitor_price": round(float(competitor_price), 2)
    }
