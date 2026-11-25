# Credit Risk Modeling & PSAK 71 Expected Credit Loss (ECL/CKPN) Platform

An end-to-end quantitative credit risk modeling framework and interactive stress-testing platform compliant with **PSAK 71 / IFRS 9** accounting standards. This project models **Probability of Default (PD)**, **Loss Given Default (LGD)**, and **Exposure at Default (EAD)**, incorporates **Discrete-Time Markov Chain** transition matrices for lifetime risk projection, and integrates **forward-looking macroeconomic overlays**.

---

## Business Context & Objectives
Under **PSAK 71 (IFRS 9)**, financial institutions are required to transition from an *incurred loss* model to a *forward-looking Expected Credit Loss (ECL / CKPN)* framework. Accurate provisioning ensures capital adequacy during economic downturns while preventing over-provisioning that restricts lending capacity.

**Key Objectives:**
1. Implement **3-Stage Impairment Allocation** (Stage 1: 12-Month ECL; Stage 2 & 3: Lifetime ECL) using delinquency and Significant Increase in Credit Risk (SICR) triggers.
2. Develop an interpretable **PD Scorecard** via **Weight of Evidence (WoE)** & **Information Value (IV)** binning paired with Logistic Regression.
3. Project multi-year cumulative default curves using **Markov Chain Transition Matrices ($P^t$)**.
4. Quantify **Probability-Weighted ECL** across three macroeconomic scenarios (*25% Optimistic, 50% Base Case, 25% Downturn*) and deploy an interactive **Stress-Testing Simulator**.

---

## Project Architecture & Pipeline

```text
credit-risk-ecl-psak71/
├── data/
│   ├── raw/                  # Raw loan performance dataset (Lending Club 2015–2018)
│   └── processed/            # Staged & risk-parameterized Parquet datasets
├── models/
│   └── pd_scorecard_model.pkl # Serialized Logistic Regression PD model & WoE mappings
├── notebooks/
│   ├── 01_data_prep_staging.ipynb       # Data cleaning, outlier handling & PSAK 71 staging
│   ├── 02_woe_pd_lgd_modeling.ipynb     # WoE/IV feature engineering, PD model, Gini/KS & LGD
│   └── 03_markov_macro_ecl.ipynb        # Markov Chain lifetime PD, macro overlay & ECL engine
├── app/
│   └── streamlit_app.py      # Interactive CKPN analytics & stress-testing dashboard
├── requirements.txt
└── README.md