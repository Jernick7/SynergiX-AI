# serve_api.py
import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Descriptors

app = FastAPI(title="SynergiX AI Prediction Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load trained Random Forest model
model_payload = joblib.load("synergy_rf_model.joblib")
model = model_payload["model"]
feature_names = model_payload["feature_names"]

with open("metrics.json", "r") as f:
    metrics = json.load(f)

# 2. Load all 496 real experimental pairs for instant ground truth lookup
csv_path = "Training_set.csv" if os.path.exists("Training_set.csv") else "Training_set (1).csv"
df_train = pd.read_csv(csv_path)
df_train.columns = df_train.columns.str.strip().str.replace("'", "")

exp_pairs = {}
for _, row in df_train.iterrows():
    n1 = str(row["Sample Name 1"]).strip().lower()
    n2 = str(row["Sample Name 2"]).strip().lower()
    g_val = float(row["Gamma"])
    exp_pairs[(n1, n2)] = g_val
    exp_pairs[(n2, n1)] = g_val

class DrugPairRequest(BaseModel):
    name1: str
    smiles1: str
    pot1: float
    name2: str
    smiles2: str
    pot2: float

def extract_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol)
    }

@app.get("/metrics")
def get_metrics():
    return metrics

@app.post("/predict")
def predict_synergy(req: DrugPairRequest):
    d1 = extract_descriptors(req.smiles1)
    d2 = extract_descriptors(req.smiles2)
    
    if not d1 or not d2:
        return {"error": "Invalid SMILES structure detected"}
        
    feat = {}
    for k, v in d1.items():
        feat[f"D1_{k}"] = v
    for k, v in d2.items():
        feat[f"D2_{k}"] = v
        
    feat["Potency_D1"] = req.pot1
    feat["Potency_D2"] = req.pot2
    feat["Potency_Diff"] = abs(req.pot1 - req.pot2)
    feat["Potency_Mean"] = (req.pot1 + req.pot2) / 2.0
    
    for col in feature_names:
        if col not in feat:
            feat[col] = 0.0
            
    input_df = pd.DataFrame([feat])[feature_names]
    gamma = float(model.predict(input_df)[0])
    
    # Ground truth lookup across the 496 real experimental pairs
    k1 = req.name1.strip().lower()
    k2 = req.name2.strip().lower()
    exp_gamma = exp_pairs.get((k1, k2), None)
    
    return {
        "gamma": round(gamma, 4),
        "synergistic": bool(gamma < 0.95),
        "exp_gamma": round(exp_gamma, 4) if exp_gamma is not None else None,
        "descriptors1": d1,
        "descriptors2": d2
    }