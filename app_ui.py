# app_ui.py
import json
import joblib
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

st.set_page_config(page_title="SynergiX AI | Live Chemical Screener", layout="wide", page_icon="🧬")

# --- Load Trained Proof-of-Work Model ---
@st.cache_resource
def load_trained_ai():
    payload = joblib.load("synergy_rf_model.joblib")
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    return payload["model"], payload["feature_names"], metrics

model, feature_names, metrics = load_trained_ai()

def calculate_molecular_features(smiles_str: str):
    """Real-time RDKit feature extraction engine."""
    mol = Chem.MolFromSmiles(str(smiles_str).strip())
    if mol is None:
        return None, None
    
    props = {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "HBD": int(Descriptors.NumHDonors(mol)),
        "HBA": int(Descriptors.NumHAcceptors(mol)),
        "RotatableBonds": int(Descriptors.NumRotatableBonds(mol))
    }
    img = Draw.MolToImage(mol, size=(400, 180))
    return props, img

# --- Top Header & CV Metrics ---
st.title("🧬 SynergiX AI — Live De Novo Synergy Predictor")
st.caption("Real-Time Molecular Descriptor Extraction & Random Forest Ensemble Inference")

m1, m2, m3, m4 = st.columns(4)
m1.metric("5-Fold CV ROC-AUC", f"{metrics['roc_auc']:.4f}")
m2.metric("5-Fold CV RMSE", f"{metrics['rmse']:.4f}")
m3.metric("5-Fold CV R²", f"{metrics['r2']:.4f}")
m4.metric("Trained Combinations", metrics["sample_count"])

st.markdown("---")

tab_live, tab_batch, tab_proof, tab_why = st.tabs([
    "🧪 Live Chemical Screener",
    "📁 Batch Drug Screening (CSV)",
    "📊 Model Proof & Importances",
    "🧠 Why Random Forest & Architecture"
])

# ----------------- TAB 1: LIVE CHEMICAL INFERENCE -----------------
with tab_live:
    st.subheader("Dual-Compound Molecular Entry")
    st.write("Paste the chemical structure (SMILES) and single-agent potency ($\log \text{AC}_{50}$) for each drug. The AI extracts chemical features and predicts combination efficacy dynamically.")

    # Quick Presets for Demo
    preset = st.selectbox(
        "⚡ Quick-Fill Preset for Live Demo (or edit the SMILES boxes directly below):",
        [
            "Preset 1: BI-2536 + Daporinad (Synergistic Hit in PANC-1)",
            "Preset 2: GMX-1778 + Daporinad (Strong Dual Pathway Inhibitor)",
            "Preset 3: Panobinostat + Oprozomib (HDAC + Proteasome Combo)",
            "Preset 4: Custom / Type Your Own Molecules"
        ]
    )

    if preset == "Preset 1: BI-2536 + Daporinad (Synergistic Hit in PANC-1)":
        init_name1 = "BI-2536"
        init_smiles1 = "CC[C@H]1N(C2CCCC2)C3=NC(NC4=CC=C(C=C4OC)C(=O)NC5CCN(C)CC5)=NC=C3N(C)C1=O"
        init_pot1 = -7.4816
        init_name2 = "Daporinad"
        init_smiles2 = "O=C(NCCCCC1CCN(CC1)C(=O)C2=CC=CC=C2)\\C=C\\C3=CN=CC=C3"
        init_pot2 = -7.5316
    elif preset == "Preset 2: GMX-1778 + Daporinad (Strong Dual Pathway Inhibitor)":
        init_name1 = "GMX-1778"
        init_smiles1 = "ClC1=CC=C(OCCCCCCN\\C(NC2=CC=NC=C2)=N/C#N)C=C1"
        init_pot1 = -8.2000
        init_name2 = "Daporinad"
        init_smiles2 = "O=C(NCCCCC1CCN(CC1)C(=O)C2=CC=CC=C2)\\C=C\\C3=CN=CC=C3"
        init_pot2 = -7.5316
    elif preset == "Preset 3: Panobinostat + Oprozomib (HDAC + Proteasome Combo)":
        init_name1 = "Panobinostat"
        init_smiles1 = "CC1=C(CCNCC2=CC=C(\\C=C\\C(=O)NO)C=C2)C3=CC=CC=C3N1"
        init_pot1 = -7.5000
        init_name2 = "Oprozomib"
        init_smiles2 = "CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C2=NC=CO2)C(=O)C3(C)OC3"
        init_pot2 = -6.8000
    else:
        init_name1 = "Compound A"
        init_smiles1 = "CC1=C(CCNCC2=CC=C(\\C=C\\C(=O)NO)C=C2)C3=CC=CC=C3N1"
        init_pot1 = -7.0000
        init_name2 = "Compound B"
        init_smiles2 = "O=C(NCCCCC1CCN(CC1)C(=O)C2=CC=CC=C2)\\C=C\\C3=CN=CC=C3"
        init_pot2 = -7.0000

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("### 🔵 Drug / Compound 1")
        name1 = st.text_input("Compound 1 Name", value=init_name1)
        smiles1 = st.text_area("SMILES String 1", value=init_smiles1, height=90)
        pot1 = st.number_input("Log AC50 (M) 1", value=float(init_pot1), step=0.1, format="%.4f")
        
        props1, img1 = calculate_molecular_features(smiles1)
        if img1:
            st.image(img1, caption=f"2D RDKit Structure: {name1}")
        else:
            st.error("❌ Invalid Chemical SMILES for Compound 1. Structure cannot be parsed.")

    with col_d2:
        st.markdown("### 🟢 Drug / Compound 2")
        name2 = st.text_input("Compound 2 Name", value=init_name2)
        smiles2 = st.text_area("SMILES String 2", value=init_smiles2, height=90)
        pot2 = st.number_input("Log AC50 (M) 2", value=float(init_pot2), step=0.1, format="%.4f")
        
        props2, img2 = calculate_molecular_features(smiles2)
        if img2:
            st.image(img2, caption=f"2D RDKit Structure: {name2}")
        else:
            st.error("❌ Invalid Chemical SMILES for Compound 2. Structure cannot be parsed.")

    st.markdown("---")
    
    # Live Inference Execution
    if props1 and props2:
        feature_dict = {}
        for prop_name, val in props1.items():
            feature_dict[f"D1_{prop_name}"] = val
        for prop_name, val in props2.items():
            feature_dict[f"D2_{prop_name}"] = val
            
        feature_dict["Potency_D1"] = pot1
        feature_dict["Potency_D2"] = pot2
        feature_dict["Potency_Diff"] = abs(pot1 - pot2)
        feature_dict["Potency_Mean"] = (pot1 + pot2) / 2.0

        for col in feature_names:
            if col not in feature_dict:
                feature_dict[col] = 0.0

        # Run forward pass through trained Random Forest
        vector = pd.DataFrame([feature_dict])[feature_names]
        predicted_gamma = float(model.predict(vector)[0])
        is_synergy = predicted_gamma < 0.95

        out_col1, out_col2 = st.columns([1, 2])
        
        with out_col1:
            if is_synergy:
                st.success(f"### ✅ Synergistic Pair\n**{name1} + {name2}**")
            else:
                st.warning(f"### ⚠️ Antagonistic / Additive\n**{name1} + {name2}**")
            
            st.metric("Predicted Gamma (γ)", f"{predicted_gamma:.4f}")
            st.caption("Threshold: γ < 0.95 indicates biological synergy (lower γ = stronger cancer cell kill rate).")

        with out_col2:
            st.write("**Real-Time Extracted RDKit Descriptors Fed into the Model:**")
            st.dataframe(pd.DataFrame({
                "Physicochemical Property": ["Molecular Weight", "LogP (Lipophilicity)", "Polar Surface Area (TPSA)", "H-Bond Donors", "H-Bond Acceptors", "Rotatable Bonds", "Single-Agent Log AC50"],
                f"{name1}": [f"{props1['MW']:.2f} g/mol", f"{props1['LogP']:.2f}", f"{props1['TPSA']:.1f} Å²", props1["HBD"], props1["HBA"], props1["RotatableBonds"], f"{pot1:.4f} M"],
                f"{name2}": [f"{props2['MW']:.2f} g/mol", f"{props2['LogP']:.2f}", f"{props2['TPSA']:.1f} Å²", props2["HBD"], props2["HBA"], props2["RotatableBonds"], f"{pot2:.4f} M"]
            }), hide_index=True, use_container_width=True)

# ----------------- TAB 2: BATCH CSV -----------------
with tab_batch:
    st.subheader("Batch Discovery Pipeline (Screen Multiple Pairs)")
    st.write("Upload a CSV file with columns: `Structure 1`, `Structure 2`, `Log AC50 (M) 1`, `Log AC50 (M) 2`.")
    
    file = st.file_uploader("Upload Combination Library (CSV)", type=["csv"])
    if file:
        bdf = pd.read_csv(file)
        bdf.columns = bdf.columns.str.strip().str.replace("'", "")
        st.write(f"Loaded {len(bdf)} candidate pairs.")
        
        if st.button("🚀 Run Live Batch AI Screening", type="primary"):
            with st.spinner("Extracting RDKit 2D descriptors and executing model inference..."):
                preds = []
                for _, r in bdf.iterrows():
                    p1_res, _ = calculate_molecular_features(r.get("Structure 1", ""))
                    p2_res, _ = calculate_molecular_features(r.get("Structure 2", ""))
                    if p1_res and p2_res:
                        row_f = {f"D1_{k}": v for k, v in p1_res.items()}
                        row_f.update({f"D2_{k}": v for k, v in p2_res.items()})
                        pt1 = float(r.get("Log AC50 (M) 1", -7.0))
                        pt2 = float(r.get("Log AC50 (M) 2", -7.0))
                        row_f["Potency_D1"] = pt1
                        row_f["Potency_D2"] = pt2
                        row_f["Potency_Diff"] = abs(pt1 - pt2)
                        row_f["Potency_Mean"] = (pt1 + pt2) / 2.0
                        for col in feature_names:
                            if col not in row_f:
                                row_f[col] = 0.0
                        score = float(model.predict(pd.DataFrame([row_f])[feature_names])[0])
                        preds.append(round(score, 4))
                    else:
                        preds.append(None)
                        
                bdf["Predicted_Gamma"] = preds
                bdf["Synergy_Call"] = bdf["Predicted_Gamma"].apply(lambda g: "Synergistic (γ < 0.95)" if (g and g < 0.95) else "Non-Synergistic")
                st.dataframe(bdf, use_container_width=True)
                
                csv_bytes = bdf.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Scored Pairs (CSV)", data=csv_bytes, file_name="synergix_scored_library.csv", mime="text/csv")

# ----------------- TAB 3: PROOF & EXPLAINABILITY -----------------
with tab_proof:
    st.subheader("Model Proof of Work: Feature Importance Breakdown")
    st.write("Gini split importance calculated across all 150 decision tree estimators in `synergy_rf_model.joblib`:")
    fi_df = pd.DataFrame(
        list(metrics["feature_importances"].items()), 
        columns=["Molecular / Potency Descriptor", "Gini Importance Score"]
    ).sort_values(by="Gini Importance Score", ascending=True)
    st.bar_chart(fi_df.set_index("Molecular / Potency Descriptor"))

# ----------------- TAB 4: ARCHITECTURE -----------------
with tab_why:
    st.subheader("Why Random Forest for SynergiX AI?")
    st.markdown("""
    1. **Prevents Overfitting on Small-N Biology ($N=496$ combinations):**
       - High-capacity neural networks (like Deep MLPs) require hundreds of thousands of datapoints to avoid memorizing assay noise. Random Forest constructs an ensemble of 150 decorrelated decision trees, aggregating their predictions to reduce variance without data leakage.
    2. **Multi-Threshold Pharmacological Non-Linearity:**
       - Drug synergy operates on threshold mechanics (e.g., cell apoptosis triggers only when both pathways achieve critical target occupancy). Decision trees naturally model these step-function interactions without complex polynomial transformations.
    3. **Robustness to Multicollinearity:**
       - Molecular descriptors (Molecular Weight, TPSA, Rotatable Bonds) are frequently correlated. Random subspace feature sampling ensures dominant descriptors do not suppress complementary signals.
    4. **Explainable Discoveries for Clinicians:**
       - Provides clear feature-importance rankings, enabling oncologists to verify which chemical traits drive combination efficacy.
    """)