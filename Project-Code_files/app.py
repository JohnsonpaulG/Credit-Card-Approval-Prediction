"""
CreditAI - Credit Card Approval Prediction
Flask Web Application

Loads the trained model, scaler and label encoders and serves a banking-style
web interface for predicting credit card approval, plus a prediction history
log and an ML model comparison page.
"""

import os
import io
import csv
import json
import pickle
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request,
    send_file, Response, abort
)

app = Flask(__name__)

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'history.db')

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------
with open(os.path.join(MODEL_DIR, 'credit_card_approval_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'label_encoders.pkl'), 'rb') as f:
    label_encoders = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'feature_order.pkl'), 'rb') as f:
    FEATURE_ORDER = pickle.load(f)

METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.json')
with open(METRICS_PATH, 'r') as f:
    MODEL_METRICS = json.load(f)

PRODUCTION_MODEL_NAME = 'Random Forest'

FEATURE_LABELS = {
    'CODE_GENDER': 'Gender',
    'FLAG_OWN_CAR': 'Car Ownership',
    'FLAG_OWN_REALTY': 'Property Ownership',
    'CNT_CHILDREN': 'Number of Children',
    'AMT_INCOME_TOTAL': 'Annual Income',
    'NAME_INCOME_TYPE': 'Income Type',
    'NAME_EDUCATION_TYPE': 'Education Level',
    'NAME_FAMILY_STATUS': 'Family Status',
    'NAME_HOUSING_TYPE': 'Housing Type',
    'OCCUPATION_TYPE': 'Occupation',
    'CNT_FAM_MEMBERS': 'Family Size',
    'AGE': 'Age',
    'YEARS_EMPLOYED': 'Years Employed',
}

IMPROVEMENT_SUGGESTIONS = {
    'AMT_INCOME_TOTAL': 'Increasing verified annual income can strengthen a future application.',
    'YEARS_EMPLOYED': 'A longer, uninterrupted employment history improves lender confidence.',
    'NAME_INCOME_TYPE': 'Stable income sources, such as salaried employment, are viewed favorably.',
    'OCCUPATION_TYPE': 'Documented, stable employment in a recognised occupation helps your profile.',
    'FLAG_OWN_REALTY': 'Owning property is a positive signal of financial stability.',
    'FLAG_OWN_CAR': 'Owning a vehicle can modestly support your application.',
    'CNT_FAM_MEMBERS': 'A lower dependency ratio relative to income can improve your standing.',
    'CNT_CHILDREN': 'Household size relative to income is a factor lenders weigh carefully.',
    'NAME_EDUCATION_TYPE': 'Additional documentation of qualifications can support your file.',
    'NAME_FAMILY_STATUS': 'This factor reflects household structure and carries lower actionability.',
    'NAME_HOUSING_TYPE': 'Stable, long-term housing arrangements are viewed positively.',
    'AGE': 'This factor reflects applicant age and is not directly actionable.',
    'CODE_GENDER': 'This is a demographic factor and is not actionable.',
}

# ---------------------------------------------------------------
# Mapping between the user-friendly text shown in the HTML form
# and the exact text values the LabelEncoders were fitted on.
# ---------------------------------------------------------------
GENDER_MAP = {'Male': 'M', 'Female': 'F'}
YES_NO_MAP = {'Yes': 'Y', 'No': 'N'}

INCOME_TYPE_MAP = {
    'Working': 'Working',
    'Commercial Associate': 'Commercial associate',
    'Pensioner': 'Pensioner',
    'State Servant': 'State servant',
    'Student': 'Student'
}

EDUCATION_MAP = {
    'Higher Education': 'Higher education',
    'Secondary Education': 'Secondary / secondary special',
    'Lower Secondary': 'Lower secondary',
    'Incomplete Higher': 'Incomplete higher',
    'Academic Degree': 'Academic degree'
}

FAMILY_STATUS_MAP = {
    'Married': 'Married',
    'Single / Not Married': 'Single / not married',
    'Civil Marriage': 'Civil marriage',
    'Separated': 'Separated',
    'Widow': 'Widow'
}

HOUSING_TYPE_MAP = {
    'House/Apartment': 'House / apartment',
    'With Parents': 'With parents',
    'Municipal Apartment': 'Municipal apartment',
    'Rented Apartment': 'Rented apartment',
    'Office Apartment': 'Office apartment',
    'Co-op Apartment': 'Co-op apartment'
}

OCCUPATION_MAP = {
    'Accountants': 'Accountants',
    'Cleaning Staff': 'Cleaning staff',
    'Cooking Staff': 'Cooking staff',
    'Core Staff': 'Core staff',
    'Drivers': 'Drivers',
    'HR Staff': 'HR staff',
    'High Skill Tech Staff': 'High skill tech staff',
    'IT Staff': 'IT staff',
    'Laborers': 'Laborers',
    'Low-skill Laborers': 'Low-skill Laborers',
    'Managers': 'Managers',
    'Medicine Staff': 'Medicine staff',
    'Private Service Staff': 'Private service staff',
    'Realty Agents': 'Realty agents',
    'Sales Staff': 'Sales staff',
    'Secretaries': 'Secretaries',
    'Security Staff': 'Security staff',
    'Waiters/Barmen Staff': 'Waiters/barmen staff',
    'Other': 'Other'
}

# Exposed to templates so dropdowns are generated from a single source of truth
DROPDOWN_OPTIONS = {
    'gender': list(GENDER_MAP.keys()),
    'yes_no': list(YES_NO_MAP.keys()),
    'income_type': list(INCOME_TYPE_MAP.keys()),
    'education': list(EDUCATION_MAP.keys()),
    'family_status': list(FAMILY_STATUS_MAP.keys()),
    'housing_type': list(HOUSING_TYPE_MAP.keys()),
    'occupation': list(OCCUPATION_MAP.keys()),
}


def format_datetime_12h(value):
    """Convert a stored ISO-8601 UTC timestamp into standard 12-hour format,
    e.g. '10 Jul 2025, 03:42 PM'."""
    if not value:
        return ''
    try:
        text = value.replace('Z', '+00:00')
        dt = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return value
    return dt.strftime('%d %b %Y, %I:%M %p')


app.jinja_env.filters['fmt_dt'] = format_datetime_12h


def encode_value(column, raw_text_value):
    """Use the fitted LabelEncoder for `column` to transform raw_text_value."""
    encoder = label_encoders[column]
    return int(encoder.transform([raw_text_value])[0])


# ---------------------------------------------------------------
# SQLite-backed prediction history
# ---------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            applicant_name TEXT,
            gender TEXT, own_car TEXT, own_house TEXT,
            children INTEGER, income REAL, income_type TEXT,
            education TEXT, family_status TEXT, housing_type TEXT,
            occupation TEXT, family_members INTEGER, age INTEGER,
            years_employed INTEGER,
            approved INTEGER NOT NULL,
            probability REAL,
            risk_level TEXT,
            top_factors TEXT,
            recommendation TEXT
        )
    ''')
    existing_cols = {row['name'] for row in conn.execute('PRAGMA table_info(predictions)').fetchall()}
    if 'applicant_name' not in existing_cols:
        conn.execute('ALTER TABLE predictions ADD COLUMN applicant_name TEXT')
    if 'inference_ms' not in existing_cols:
        conn.execute('ALTER TABLE predictions ADD COLUMN inference_ms REAL')
    conn.commit()
    conn.close()


init_db()


def record_prediction(form_values, approved, probability, risk_level, top_factors, recommendation, inference_ms=None):
    conn = get_db()
    conn.execute('''
        INSERT INTO predictions (
            created_at, applicant_name, gender, own_car, own_house, children, income, income_type,
            education, family_status, housing_type, occupation, family_members,
            age, years_employed, approved, probability, risk_level, top_factors, recommendation,
            inference_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now(timezone.utc).isoformat(timespec='seconds'),
        form_values['applicant_name'],
        form_values['gender_text'], form_values['own_car_text'], form_values['own_house_text'],
        int(form_values['children']), float(form_values['income']), form_values['income_type_text'],
        form_values['education_text'], form_values['family_status_text'], form_values['housing_type_text'],
        form_values['occupation_text'], int(form_values['family_members']), int(form_values['age']),
        int(form_values['years_employed']),
        int(approved), probability, risk_level, json.dumps(top_factors), recommendation,
        inference_ms
    ))
    conn.commit()
    new_id = conn.execute('SELECT last_insert_rowid() AS id').fetchone()['id']
    conn.close()
    return new_id


def fetch_all_predictions():
    conn = get_db()
    rows = conn.execute('SELECT * FROM predictions ORDER BY id DESC').fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_prediction(pred_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM predictions WHERE id = ?', (pred_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------------------------------------------------------
# Explainability helpers
# ---------------------------------------------------------------
def _feature_importances():
    if hasattr(model, 'feature_importances_'):
        return np.asarray(model.feature_importances_)
    if hasattr(model, 'coef_'):
        return np.abs(np.asarray(model.coef_[0]))
    return np.ones(len(FEATURE_ORDER))


def top_contributing_factors(encoded_row_scaled, limit=4):
    """Rank features for THIS applicant using model feature importance
    weighted by how far (in standard deviations) their scaled value sits
    from the training-set mean. Returns a list of readable strings."""
    importances = _feature_importances()
    deviation = np.abs(encoded_row_scaled[0])
    contribution = importances * deviation
    order = np.argsort(contribution)[::-1]

    factors = []
    for idx in order[:limit]:
        col = FEATURE_ORDER[idx]
        factors.append(FEATURE_LABELS.get(col, col))
    return factors


def build_recommendation(approved, encoded_row_scaled):
    importances = _feature_importances()
    deviation = np.abs(encoded_row_scaled[0])
    contribution = importances * deviation
    order = np.argsort(contribution)[::-1]

    if approved:
        return ('Your profile currently meets CreditAI\'s approval criteria. '
                'Maintain stable income and continue timely repayments to preserve this standing.')

    for idx in order:
        col = FEATURE_ORDER[idx]
        if col in ('AGE', 'CODE_GENDER', 'NAME_FAMILY_STATUS'):
            continue
        return IMPROVEMENT_SUGGESTIONS.get(
            col, 'Strengthen your income stability and repayment history, then consider re-applying.'
        )
    return 'Strengthen your income stability and repayment history, then consider re-applying.'


def risk_level_from_probability(prob_rejected):
    if prob_rejected < 0.2:
        return 'Low Risk'
    if prob_rejected < 0.5:
        return 'Medium Risk'
    return 'High Risk'


# ---------------------------------------------------------------
# Routes — marketing / informational pages
# ---------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html', active='home')


@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html', active='how_it_works')


@app.route('/features')
def features():
    return render_template('features.html', active='features')


@app.route('/ml-models')
def ml_models():
    return render_template(
        'ml_models.html',
        active='ml_models',
        metrics=MODEL_METRICS,
        production_model=PRODUCTION_MODEL_NAME,
        feature_labels=FEATURE_LABELS,
        model_order=['Random Forest', 'XGBoost', 'Decision Tree', 'Logistic Regression'],
    )


# ---------------------------------------------------------------
# Routes — prediction flow
# ---------------------------------------------------------------
@app.route('/prediction')
def prediction():
    return render_template('prediction.html', options=DROPDOWN_OPTIONS, active='prediction')


@app.route('/predict', methods=['POST'])
def predict():
    form = request.form

    applicant_name = (form.get('applicant_name') or '').strip()
    if not applicant_name:
        abort(400, description='Applicant Name is required.')

    form_values = {
        'applicant_name': applicant_name,
        'gender_text': form.get('gender'),
        'own_car_text': form.get('own_car'),
        'own_house_text': form.get('own_house'),
        'children': form.get('children'),
        'income': form.get('income'),
        'income_type_text': form.get('income_type'),
        'education_text': form.get('education'),
        'family_status_text': form.get('family_status'),
        'housing_type_text': form.get('housing_type'),
        'occupation_text': form.get('occupation'),
        'family_members': form.get('family_members'),
        'age': form.get('age'),
        'years_employed': form.get('years_employed'),
    }

    # ---- Map friendly text -> dataset text -----------------------
    gender_raw = GENDER_MAP.get(form_values['gender_text'], form_values['gender_text'])
    own_car_raw = YES_NO_MAP.get(form_values['own_car_text'], form_values['own_car_text'])
    own_house_raw = YES_NO_MAP.get(form_values['own_house_text'], form_values['own_house_text'])
    income_type_raw = INCOME_TYPE_MAP.get(form_values['income_type_text'], form_values['income_type_text'])
    education_raw = EDUCATION_MAP.get(form_values['education_text'], form_values['education_text'])
    family_status_raw = FAMILY_STATUS_MAP.get(form_values['family_status_text'], form_values['family_status_text'])
    housing_type_raw = HOUSING_TYPE_MAP.get(form_values['housing_type_text'], form_values['housing_type_text'])
    occupation_raw = OCCUPATION_MAP.get(form_values['occupation_text'], form_values['occupation_text'])

    # ---- Encode categorical fields using label_encoders.pkl ------
    encoded = {
        'CODE_GENDER': encode_value('CODE_GENDER', gender_raw),
        'FLAG_OWN_CAR': encode_value('FLAG_OWN_CAR', own_car_raw),
        'FLAG_OWN_REALTY': encode_value('FLAG_OWN_REALTY', own_house_raw),
        'CNT_CHILDREN': float(form_values['children']),
        'AMT_INCOME_TOTAL': float(form_values['income']),
        'NAME_INCOME_TYPE': encode_value('NAME_INCOME_TYPE', income_type_raw),
        'NAME_EDUCATION_TYPE': encode_value('NAME_EDUCATION_TYPE', education_raw),
        'NAME_FAMILY_STATUS': encode_value('NAME_FAMILY_STATUS', family_status_raw),
        'NAME_HOUSING_TYPE': encode_value('NAME_HOUSING_TYPE', housing_type_raw),
        'OCCUPATION_TYPE': encode_value('OCCUPATION_TYPE', occupation_raw),
        'CNT_FAM_MEMBERS': float(form_values['family_members']),
        'AGE': float(form_values['age']),
        'YEARS_EMPLOYED': float(form_values['years_employed']),
    }

    # ---- Build feature vector in the exact order used at training time
    feature_vector = pd.DataFrame([[encoded[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)

    # ---- Scale + Predict (timed for display on the result page) ------
    inference_start = time.perf_counter()
    feature_vector_scaled = scaler.transform(feature_vector)
    prediction_value = int(model.predict(feature_vector_scaled)[0])

    if hasattr(model, 'predict_proba'):
        proba_row = model.predict_proba(feature_vector_scaled)[0]
        probability = float(proba_row[prediction_value])
        prob_rejected = float(proba_row[1])
    else:
        probability = None
        prob_rejected = 1.0 if prediction_value == 1 else 0.0
    inference_ms = round((time.perf_counter() - inference_start) * 1000, 1)

    approved = (prediction_value == 0)

    factors = top_contributing_factors(feature_vector_scaled)
    recommendation = build_recommendation(approved, feature_vector_scaled)
    risk_level = risk_level_from_probability(prob_rejected)

    pred_id = record_prediction(form_values, approved, probability, risk_level, factors, recommendation, inference_ms)

    return render_template(
        'result.html',
        active='prediction',
        applicant_name=applicant_name,
        approved=approved,
        probability=probability,
        risk_level=risk_level,
        factors=factors,
        recommendation=recommendation,
        pred_id=pred_id,
        model_name=PRODUCTION_MODEL_NAME,
        inference_ms=inference_ms,
    )


# ---------------------------------------------------------------
# Routes — history
# ---------------------------------------------------------------
@app.route('/result/<int:pred_id>')
def view_result(pred_id):
    record = fetch_prediction(pred_id)
    if record is None:
        abort(404)
    factors = json.loads(record['top_factors']) if record['top_factors'] else []
    return render_template(
        'result.html',
        active='history',
        applicant_name=record['applicant_name'],
        approved=bool(record['approved']),
        probability=record['probability'],
        risk_level=record['risk_level'],
        factors=factors,
        recommendation=record['recommendation'],
        pred_id=record['id'],
        model_name=PRODUCTION_MODEL_NAME,
        inference_ms=record['inference_ms'],
    )


@app.route('/history')
def history():
    rows = fetch_all_predictions()
    for r in rows:
        r['top_factors'] = json.loads(r['top_factors']) if r['top_factors'] else []
    return render_template('history.html', active='history', predictions=rows)


@app.route('/history/export.csv')
def export_history_csv():
    rows = fetch_all_predictions()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'ID', 'Date & Time', 'Applicant Name', 'Decision', 'Confidence %', 'Risk Level', 'Gender', 'Age',
        'Income', 'Income Type', 'Education', 'Family Status', 'Housing Type',
        'Occupation', 'Children', 'Family Members', 'Years Employed', 'Top Factors'
    ])
    for r in rows:
        writer.writerow([
            r['id'], format_datetime_12h(r['created_at']), r['applicant_name'] or '', 'Approved' if r['approved'] else 'Rejected',
            f"{(r['probability'] or 0) * 100:.1f}", r['risk_level'], r['gender'], r['age'],
            r['income'], r['income_type'], r['education'], r['family_status'], r['housing_type'],
            r['occupation'], r['children'], r['family_members'], r['years_employed'],
            '; '.join(json.loads(r['top_factors']) if r['top_factors'] else [])
        ])
    csv_bytes = output.getvalue().encode('utf-8')
    return Response(
        csv_bytes,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=creditai_history.csv'}
    )


@app.route('/result/<int:pred_id>/download-pdf')
def download_pdf(pred_id):
    record = fetch_prediction(pred_id)
    if record is None:
        abort(404)

    try:
        from fpdf import FPDF
    except ImportError:
        abort(500, description='PDF generation library not installed.')

    approved = bool(record['approved'])
    factors = json.loads(record['top_factors']) if record['top_factors'] else []

    pdf = FPDF(format='A4', unit='mm')
    pdf.add_page()
    pdf.set_fill_color(10, 23, 48)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_xy(12, 9)
    pdf.cell(0, 10, 'CreditAI - Prediction Report')

    pdf.set_text_color(20, 30, 50)
    pdf.set_xy(12, 38)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f"Report ID: #{record['id']:06d}    Generated: {format_datetime_12h(record['created_at'])}")

    pdf.set_xy(12, 45)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 7, f"Applicant: {record['applicant_name'] or 'N/A'}")

    pdf.set_xy(12, 58)
    pdf.set_font('Helvetica', 'B', 16)
    if approved:
        pdf.set_text_color(22, 163, 74)
        pdf.cell(0, 10, 'DECISION: APPROVED')
    else:
        pdf.set_text_color(220, 38, 38)
        pdf.cell(0, 10, 'DECISION: REJECTED')

    pdf.set_text_color(20, 30, 50)
    pdf.set_xy(12, 70)
    pdf.set_font('Helvetica', '', 11)
    conf = f"{(record['probability'] or 0) * 100:.1f}%" if record['probability'] is not None else 'N/A'
    pdf.cell(0, 7, f"Model Confidence: {conf}")
    pdf.set_xy(12, 78)
    pdf.cell(0, 7, f"Risk Level: {record['risk_level']}")

    pdf.set_xy(12, 92)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Applicant Summary')
    pdf.set_font('Helvetica', '', 10)

    rows = [
        ('Gender', record['gender']), ('Age', record['age']),
        ('Annual Income', record['income']), ('Income Type', record['income_type']),
        ('Education', record['education']), ('Family Status', record['family_status']),
        ('Housing Type', record['housing_type']), ('Occupation', record['occupation']),
        ('Children', record['children']), ('Family Members', record['family_members']),
        ('Years Employed', record['years_employed']),
        ('Owns Car', record['own_car']), ('Owns Property', record['own_house']),
    ]
    y = 102
    for label, value in rows:
        pdf.set_xy(12, y)
        pdf.cell(60, 6, f"{label}:")
        pdf.set_xy(72, y)
        pdf.cell(0, 6, str(value))
        y += 7

    y += 4
    pdf.set_xy(12, y)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Top Contributing Factors')
    y += 9
    pdf.set_font('Helvetica', '', 10)
    for factor in factors:
        pdf.set_xy(16, y)
        pdf.cell(0, 6, f"- {factor}")
        y += 6

    y += 6
    pdf.set_xy(12, y)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Recommendation')
    y += 9
    pdf.set_font('Helvetica', '', 10)
    pdf.set_xy(12, y)
    pdf.multi_cell(180, 6, record['recommendation'] or '')

    pdf.set_y(-20)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, 'CreditAI is a machine learning demonstration project. Not a real financial institution.')

    pdf_bytes = bytes(pdf.output())
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'creditai_report_{record["id"]:06d}.pdf'
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
