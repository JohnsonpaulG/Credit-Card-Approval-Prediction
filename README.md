# Credit Card Approval Prediction

A complete, end-to-end machine learning project that predicts whether a credit
card application should be **APPROVED** or **REJECTED**, wrapped in a
professional, banking-style Flask web application.

----------------------------------------------------------------------------------
Live demo: https://credai.onrender.com

---
>>>>>>> 45268298a99c5f5e8d44d1321495c3cb5437081a

## 1. Project Overview

Banks need to decide, quickly and consistently, whether an applicant is likely
to become a reliable credit card customer. This project builds that decision
engine end to end:

- **`application_record.csv`** supplies demographic, income and employment
  data for each applicant.
- **`credit_record.csv`** supplies each applicant's monthly repayment history,
  which is used to engineer the ground-truth `TARGET` label (good vs. bad
  client).
- A Jupyter notebook explores the data, engineers features, trains four
  classification models (Logistic Regression, Decision Tree, Random Forest,
  XGBoost), and saves the best-performing model.
- A Flask web application loads that saved model and serves a clean,
  three-page banking-style UI where a user can fill in an applicant profile
  and instantly receive an **APPROVED / REJECTED** decision.

---

## 2. Folder Structure

```
Credit-Card-Approval-Prediction/
│
├── notebook/
│   └── Credit_Card_Approval_Prediction.ipynb   # Full EDA + model training notebook
│
├── model/
│   ├── credit_card_approval_model.pkl          # Best trained classifier
│   ├── scaler.pkl                              # Fitted StandardScaler
│   ├── label_encoders.pkl                      # Dict of fitted LabelEncoders
│   └── feature_order.pkl                       # Exact column order used at train time
│
├── templates/
│   ├── index.html                              # Home page
│   ├── prediction.html                         # Application form page
│   └── result.html                             # Approved / Rejected result page
│
├── static/
│   └── css/
│       └── style.css                           # Banking-style theme
│
├── data/
│   ├── application_record.csv
│   └── credit_record.csv
│
├── app.py                                      # Flask application
├── requirements.txt
└── README.md
```

---

## 3. Dataset

| File | Description |
|---|---|
| `application_record.csv` | One row per applicant: gender, car/property ownership, children, income, income type, education, family status, housing type, birth/employment days, occupation, family size. |
| `credit_record.csv` | One row per applicant per month: `STATUS` of that month's account (`0`–`5` = days past due, `C` = paid off, `X` = no loan that month). |

**Target engineering:** an applicant is labeled `TARGET = 1` (bad client →
**Rejected**) if they were ever **60+ days past due** (`STATUS` of `2`, `3`,
`4`, or `5`) in their credit history. Otherwise they are labeled `TARGET = 0`
(good client → **Approved**).

---

## 4. Installation

```bash
# 1. Clone / unzip the project, then move into it
cd Credit-Card-Approval-Prediction

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. How to Run

### Retrain the model (optional — a trained model is already included in `model/`)

```bash
cd notebook
jupyter notebook Credit_Card_Approval_Prediction.ipynb
```
Run every cell top to bottom. This regenerates all three files inside `model/`.

### Launch the web application

```bash
# From the project root
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

1. **Home page** — project overview, click **Start Prediction**.
2. **Prediction page** — fill in every field (all dropdowns start on
   `Select...`, no pre-filled values) and click **Get My Decision**.
3. **Result page** — see a large **APPROVED** (green) or **REJECTED** (red)
   verdict with a model confidence score, then **Predict Again** or
   **Back Home**.

---

## 6. Screenshots

> _Add screenshots here after running the app locally._

- `screenshots/home.png`
- `screenshots/prediction-form.png`
- `screenshots/result-approved.png`
- `screenshots/result-rejected.png`

---

## 7. Future Scope

- Hyperparameter tuning (GridSearchCV / Optuna) for the selected model.
- Handle class imbalance more robustly with SMOTE or ADASYN.
- Add explainability (SHAP / LIME) to show *why* an application was approved
  or rejected.
- Add authentication and a database to persist submitted applications.
- Containerize with Docker and deploy to a cloud platform.
- Add a REST API endpoint (`/api/predict`) returning JSON for programmatic use.

---

## 8. Tech Stack

Python · Pandas · NumPy · Scikit-Learn · XGBoost · Flask · HTML · CSS
