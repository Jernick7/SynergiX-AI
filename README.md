# 🧬 SynergiX AI: AI-Driven Discovery of Synergistic Drug Combinations for Pancreatic Cancer (PANC-1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-orange.svg)](https://www.rdkit.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-f7931e.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **MED-04 Medical AI Hackathon Project** — A full-stack artificial intelligence and cheminformatics platform designed to predict and explain synergistic anti-cancer drug combinations for Pancreatic Ductal Adenocarcinoma (PANC-1) cell lines.

---

## 🔬 Overview
Pancreatic cancer (PANC-1) carries a 5-year survival rate of under 11% due to rapid compensatory resistance against monotherapy. Combining two drugs with complementary Mechanisms of Action (MOA) can block primary and backup tumor survival routes simultaneously. 

**SynergiX AI** automates this discovery process by translating chemical structures (SMILES) and single-agent potencies ($\log \text{AC}_{50}$) into real-time synergy predictions ($\gamma$ scores) via a 5-fold cross-validated **Random Forest Ensemble** trained on NCATS high-throughput screening data.

---

## 📊 Model Performance & Benchmarking
Before selecting our production model, we systematically benchmarked 5 distinct model families using strict 5-fold cross-validation on all 496 experimental PANC-1 assay combinations:

| Model Architecture | Family | CV RMSE ↓ | CV MAE ↓ | CV R² ↑ | ROC-AUC ↑ | Accuracy ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Selected)** | **Bagged CART Ensemble (150 Trees)** | **0.1188** | **0.0843** | **0.4423** | **0.8073** | **73.2%** |
| **Stacked Ensemble** | Meta-Regressor (RF + GBDT + MLP + Ridge) | 0.1189 | 0.0848 | 0.4417 | 0.7999 | 72.0% |
| **Gradient Boosting** | Boosted Decision Trees (GBDT) | 0.1190 | 0.0857 | 0.4406 | 0.7857 | 70.8% |
| **Ridge Regression** | L2-Regularized Linear Model | 0.1243 | 0.0909 | 0.3903 | 0.7838 | 70.4% |
| **Deep Synergy MLP** | 2-Layer Neural Network (ReLU + Adam) | 0.1278 | 0.0905 | 0.3555 | 0.7803 | 70.6% |

---

## 🛠️ Repository File Structure
```text
📦 SynergiX-AI
 ┣ 📜 Training_set.csv           # NCATS PANC-1 combinatorial screen (496 assay rows)
 ┣ 📜 inspect_data.py           # Pre-flight data audit & schema verification script
 ┣ 📜 features.py               # RDKit cheminformatics feature extraction pipeline
 ┣ 📜 engineered_features.csv   # Numerical feature matrix (496 samples × 18 features)
 ┣ 📜 train.py                  # 5-fold CV trainer and artifact exporter
 ┣ 📜 synergy_rf_model.joblib   # Frozen trained Random Forest ensemble brain
 ┣ 📜 metrics.json              # Saved cross-validation metrics & Gini importances
 ┣ 📜 serve_api.py              # FastAPI backend serving live model predictions
 ┣ 📜 dashboard.html            # Responsive dark-mode biotech UI with SmilesDrawer 2D
 ┗ 📜 README.md                 # Project documentation
``` 


# 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Jernick7/SynergiX-AI.git
cd SynergiX-AI
```

### 2. Install Python Dependencies
```bash
pip install pandas numpy scikit-learn rdkit fastapi uvicorn pydantic joblib
```

### 3. Run the ML Training Pipeline (Optional)
To retrain the model and regenerate serialized artifacts from raw data:
```bash
python inspect_data.py
python features.py
python train.py
```

### 4. Launch the Live FastAPI Backend
Start the Uvicorn server hosting your trained model:
```bash
python -m uvicorn serve_api:app --reload --port 8000
```
*(Interactive API Swagger documentation is available at http://127.0.0.1:8000/docs)*

### 5. Launch the Biotech UI
Double-click `dashboard.html` or open it in your web browser to interact with the live synergy screener.

---

# 🧪 Key Features

* **Real-Time RDKit Cheminformatics:** Instantly parses SMILES strings into 2D chemical drawings, Molecular Weight, Lipophilicity $$\log P$$, Polar Surface Area (TPSA), and H-bond profiles.
* **Dual-Level Validation:**
  * **Macro:** 5-Fold Cross-Validation (ROC-AUC = 0.8073, RMSE = 0.1355).
  * **Micro:** Automated NCATS wet-lab ground truth lookup displaying real-time model absolute error $\vert\Delta\gamma\vert$.
* **De Novo Discovery & Batch Screening:** Supports custom SMILES input and CSV batch uploading for high-throughput library screening.
* **Explainable AI (XAI):** Gini split importances identifying Potency Asymmetry $|\Delta \log \text{AC}_{50}|$ and Lipophilicity as primary drivers of combination efficacy.

---

# 👥 Authors

* **V Jernick Samuel** — AI & Full-Stack Development
* *Built for the MED-04 Medical AI Hackathon.*

---

# 📄 License

This project is licensed under the terms of the **MIT License**.
