# CreditAI — Credit Card Approval Prediction

A complete, end-to-end machine learning project that predicts whether a credit
card application should be **APPROVED** or **REJECTED**, wrapped in a
professional, banking-style Flask web application with a live dashboard,
prediction history, and a full ML model comparison page.

---

## 1. Project Overview

- A Jupyter notebook explores the data, engineers features, trains four
  classification models (Logistic Regression, Decision Tree, Random Forest,
  XGBoost), and saves the best-performing model — **Random Forest**
  (91.6% test accuracy) — as the production model.
- A Flask web application loads that saved model and serves:
  - A **Home** page with a project overview, team section, workflow and
    feature highlights.
  - A **Prediction** page where a user fills in an applicant profile and
    instantly receives an **APPROVED / REJECTED** decision with a confidence
    score, risk level, top contributing factors and a recommendation.
  - An **ML Models** page comparing all four trained models side by side
    (accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
    classification report, feature importance).
  - A **Dashboard** summarizing every prediction made in the current
    deployment.
  - A **History** page to search, filter, sort, paginate and export past
    predictions to CSV.
  - A downloadable **PDF report** for every approved prediction.

---

## 2. Folder Structure

```
CredAI/
│
├── notebook/
│   └── Credit_Card_Approval_Prediction.ipynb   # Full EDA + model training notebook
│
├── model/
│   ├── credit_card_approval_model.pkl          # Production model (Random Forest)
│   ├── scaler.pkl                              # Fitted StandardScaler
│   ├── label_encoders.pkl                      # Dict of fitted LabelEncoders
│   ├── feature_order.pkl                       # Exact column order used at train time
│   └── model_metrics.json                      # Pre-computed metrics for all 4 models
│
├── templates/
│   ├── _navbar.html / _footer.html             # Shared layout partials
│   ├── index.html                              # Home page
│   ├── about.html / contact.html                # Info pages
│   ├── features.html / how_it_works.html         # Info pages
│   ├── ml_models.html                          # Model comparison page
│   ├── prediction.html                         # Application form page
│   ├── result.html                             # Approved / Rejected result page
│   ├── dashboard.html                          # Live stats dashboard
│   └── history.html                            # Prediction history log
│
├── static/
│   ├── css/style.css                           # Banking-style theme (glassmorphism)
│   └── js/main.js                               # Counters, scroll reveal, tabs, history table
│
├── data/                                        # SQLite prediction history is created here at runtime
│
├── app.py                                      # Flask application
├── requirements.txt
├── runtime.txt                                 # Python version pin for Render
├── render.yaml                                 # Render deployment config
└── README.md
```

> **Note:** the raw `application_record.csv` / `credit_record.csv` training
> files are **not** included in this deployment package to keep it
> lightweight — they are only needed if you want to re-run the notebook and
> retrain the model from scratch. The dataset is the public
> ["Credit Card Approval Prediction"](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
> dataset on Kaggle. The app itself only needs the four `.pkl` files already
> committed inside `model/`.

---

## 3. How Predictions Work

**Target engineering (at training time):** an applicant is labeled
`TARGET = 1` (bad client → **Rejected**) if they were ever **60+ days past
due** (`STATUS` of `2`, `3`, `4`, or `5`) in their credit history. Otherwise
they are labeled `TARGET = 0` (good client → **Approved**).

**At prediction time**, the app:
1. Encodes the submitted form using the exact `LabelEncoder`s fitted during
   training.
2. Scales the feature vector with the fitted `StandardScaler`.
3. Runs it through the production Random Forest model.
4. Ranks the applicant's own features by `importance × deviation from the
   training mean` to surface the top contributing factors for *that specific*
   application.
5. Logs the result (inputs, decision, confidence, risk level, factors) to a
   local SQLite database, which powers the Dashboard and History pages.

---

## 4. Installation

```bash
# 1. Clone / unzip the project, then move into it
cd CredAI

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. How to Run Locally

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. A `data/history.db`
SQLite file is created automatically on first run.

### Retrain the model (optional — a trained model is already included in `model/`)

Download `application_record.csv` and `credit_record.csv` from the Kaggle
dataset linked above into a local `data/` folder, then:

```bash
cd notebook
jupyter notebook Credit_Card_Approval_Prediction.ipynb
```

Run every cell top to bottom to regenerate the files inside `model/`.

---

## 6. Deploying to Render

This project ships ready to deploy as-is:

1. Push this project to a GitHub repository.
2. On [Render](https://render.com), create a **New Web Service** from that
   repo (Render will detect `render.yaml` automatically), or configure
   manually:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
3. Deploy. No environment variables are required.

> Render's free tier uses an ephemeral filesystem, so `data/history.db`
> resets on every redeploy/restart — expected behavior for this
> demonstration project, not a bug.

---

## 7. Team

| Name | Roll Number | Role |
|---|---|---|
| _Shaik Asha_ | _238X1A42A1_ | Project Leader |
| _Sravanthi Choudaboina_ | _238X1A4220_ | Machine Learning Developer |
| _Yagnesh challa_ | _238X1A4214_ | Backend Developer |
| _Gandham Johnson Paul_ | _238X1A4225_ | Frontend Developer |
| _Akkala Shanmukha Reddy_ | _238X1A4249_ | Documentation & Testing |



---

## 8. Future Scope

- Hyperparameter tuning (GridSearchCV / Optuna) for the selected model.
- Handle class imbalance more robustly with SMOTE or ADASYN.
- Add SHAP/LIME-based explainability for even richer per-prediction insight.
- Add authentication and a persistent (non-ephemeral) database.
- Containerize with Docker.
- Add a REST API endpoint (`/api/predict`) returning JSON for programmatic use.

---

## 9. Tech Stack

Python · Pandas · NumPy · Scikit-Learn · Flask · gunicorn · fpdf2 · SQLite · HTML · CSS · JavaScript
