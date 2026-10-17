# features.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

def get_molecular_descriptors(smiles_str: str) -> dict:
    """Calculates chemical descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(str(smiles_str))
    if mol is None:
        return {
            "MW": np.nan,
            "LogP": np.nan,
            "TPSA": np.nan,
            "HBD": np.nan,
            "HBA": np.nan,
            "RotatableBonds": np.nan
        }
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol)
    }

def engineer_features(input_path: str = "Training_set.csv", output_path: str = "engineered_features.csv"):
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Clean whitespace from column names
    df.columns = df.columns.str.strip()

    # Identify relevant columns dynamically
    col_smiles1 = "Structure 1"
    col_smiles2 = "Structure 2"
    col_pot1 = "Log AC50 (M) 1"
    col_pot2 = "Log AC50 (M) 2"
    col_moa1 = "Mechanism of Action (MOA) 1"
    col_moa2 = "Mechanism of Action (MOA) 2"

    print("Extracting chemical descriptors using RDKit...")
    desc1 = df[col_smiles1].apply(get_molecular_descriptors).apply(pd.Series)
    desc2 = df[col_smiles2].apply(get_molecular_descriptors).apply(pd.Series)

    # Prefix columns
    desc1.columns = [f"D1_{c}" for c in desc1.columns]
    desc2.columns = [f"D2_{c}" for c in desc2.columns]

    # Combine descriptors with single-agent potency values
    feats = pd.concat([desc1, desc2], axis=1)

    # 2. Potency relations
    if col_pot1 in df.columns and col_pot2 in df.columns:
        feats["Potency_D1"] = df[col_pot1]
        feats["Potency_D2"] = df[col_pot2]
        feats["Potency_Diff"] = (df[col_pot1] - df[col_pot2]).abs()
        feats["Potency_Mean"] = (df[col_pot1] + df[col_pot2]) / 2.0

    # 3. Target / Mechanism relationship (binary flag: different target = 1, same target = 0)
    if col_moa1 in df.columns and col_moa2 in df.columns:
        feats["MOA_Different"] = (df[col_moa1].str.lower() != df[col_moa2].str.lower()).astype(int)

    # 4. Attach ground truth targets
    if "Gamma" in df.columns:
        feats["Gamma"] = df["Gamma"]
    if "Label" in df.columns:
        feats["Label"] = df["Label"]

    # Fill any parsing misses with column medians
    feats = feats.fillna(feats.median(numeric_only=True))

    feats.to_csv(output_path, index=False)
    print(f"Engineered features successfully saved to {output_path} with shape {feats.shape}")

if __name__ == "__main__":
    engineer_features()