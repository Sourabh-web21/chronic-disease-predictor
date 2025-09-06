import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import base64
import io
from flask import Flask, render_template, request, redirect, url_for, flash

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "supersecretkey"

# -----------------------------
# Base directory (works both locally and on Render)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "patients.db")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
TARGETS = ["Heart_Y", "Diabetes_Y", "Kidney_Y"]

# -----------------------------
# Load models
# -----------------------------
models = {}
for target in TARGETS:
    path = os.path.join(MODEL_FOLDER, f"{target}.joblib")
    if os.path.exists(path):
        models[target] = joblib.load(path)

# -----------------------------
# Load patients
# -----------------------------
# -----------------------------
# Load patients from SQLite
# -----------------------------
patients = {}
if os.path.exists(DB_PATH):
    import sqlite3
    conn = sqlite3.connect(DB_PATH)

    # get all unique Patient_IDs
    pids = pd.read_sql("SELECT DISTINCT Patient_ID FROM patient_data", conn)["Patient_ID"].astype(str).tolist()

    for pid in pids:
        df = pd.read_sql(f"SELECT * FROM patient_data WHERE Patient_ID = ?", conn, params=(pid,))
        df['Patient_ID'] = df['Patient_ID'].astype(str)
        if 'Day' not in df.columns:
            df['Day'] = range(len(df))
        patients[pid] = df

    conn.close()


# -----------------------------
# Helper functions
# -----------------------------
def generate_risk_gauge(risk_score):
    fig, ax = plt.subplots(figsize=(3, 1.5))
    ax.barh([0], [risk_score], color='red' if risk_score > 0.66 else 'orange' if risk_score > 0.33 else 'green')
    ax.barh([0], [1], left=[risk_score], color='lightgray', alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def feature_engineering(df):
    feats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in TARGETS or col == "Day":
            continue
        feats[f"{col}_mean"] = df[col].mean()
        feats[f"{col}_std"] = df[col].std() if df[col].std() > 0 else df[col].mean()*0.1
        feats[f"{col}_last"] = df[col].iloc[-1]
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if col in ['Patient_ID', 'Day'] + TARGETS:
            continue
        feats[f"{col}_last"] = str(df[col].iloc[-1])
    X_pred = pd.DataFrame([feats])
    for col in X_pred.select_dtypes(include='object').columns:
        X_pred[col] = X_pred[col].astype('category').cat.codes
    return X_pred

def generate_plot(df, col):
    fig, ax = plt.subplots(figsize=(4,3))
    series = pd.to_numeric(df[col], errors='coerce')
    ax.plot(df['Day'], series, marker='o')
    ax.set_title(col)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_base64

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        if patient_id in patients:
            return redirect(url_for("patient_detail", patient_id=patient_id))
        else:
            flash("Patient ID not found!", "danger")

    patient_list = []
    for pid, df in patients.items():
        X_pred = feature_engineering(df)
        risk_vals = []

        for target, model in models.items():
            try:
                expected_cols = model.get_booster().feature_names
                for c in expected_cols:
                    if c not in X_pred.columns:
                        X_pred[c] = 0
                X_pred_ordered = X_pred[expected_cols]
                prob = model.predict_proba(X_pred_ordered)[0,1]
                risk_vals.append(prob)
            except:
                continue

        risk_score = np.mean(risk_vals) if risk_vals else 0
        if risk_score > 0.66:
            risk_level = "High"
        elif risk_score > 0.33:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        patient_list.append({
            "id": pid,
            "risk_score": risk_score,
            "risk_level": risk_level
        })

    stats = {
        "total": len(patient_list),
        "high_risk": sum(1 for p in patient_list if p["risk_level"]=="High"),
        "medium_risk": sum(1 for p in patient_list if p["risk_level"]=="Medium"),
        "low_risk": sum(1 for p in patient_list if p["risk_level"]=="Low"),
        "models_loaded": len(models)
    }

    return render_template(
        "index.html",
        patient_ids=list(patients.keys()),
        stats=stats
    )

@app.route("/upload", methods=["POST"])
def upload_csv():
    file = request.files.get("csv_file")
    if not file:
        flash("No file selected", "danger")
        return redirect(url_for("index"))

    df = pd.read_csv(file)
    df['Patient_ID'] = df['Patient_ID'].astype(str)
    if 'Day' not in df.columns:
        df['Day'] = range(len(df))
    patient_id = df['Patient_ID'].iloc[0]

    # save into SQLite
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("patient_data", conn, if_exists="append", index=False)
    conn.close()

    # also update in-memory cache
    patients[patient_id] = df

    flash(f"Patient {patient_id} uploaded successfully", "success")
    return redirect(url_for("patient_detail", patient_id=patient_id))


@app.route("/patient/<patient_id>")
def patient_detail(patient_id):
    if patient_id not in patients:
        flash("Patient not found!", "danger")
        return redirect(url_for("index"))

    df = patients[patient_id]
    X_pred = feature_engineering(df)

    predictions = {}
    shap_info_out = {}
    overall_risks = []

    for target, model in models.items():
        try:
            expected_cols = model.get_booster().feature_names
            for c in expected_cols:
                if c not in X_pred.columns:
                    if "_mean" in c:
                        base = c.replace("_mean","_last")
                        X_pred[c] = X_pred[base] if base in X_pred.columns else 0
                    elif "_std" in c:
                        base = c.replace("_std","_last")
                        X_pred[c] = 0.1*X_pred[base] if base in X_pred.columns else 0
                    else:
                        X_pred[c] = 0

            X_ordered = X_pred[expected_cols]
            prob = model.predict_proba(X_ordered)[0,1]
            overall_risks.append(prob)

            if prob > 0.66:
                risk_level = "High"
            elif prob > 0.33:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            predictions[target] = {
                "probability": prob,
                "risk_level": risk_level,
                "confidence": round(prob * 100, 1)
            }

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_ordered)
            shap_series = pd.Series(shap_vals[0], index=X_ordered.columns).abs().sort_values(ascending=False)
            top_features = [{"feature": f, "importance": shap_series[f]} for f in shap_series.head(3).index]
            shap_info_out[target] = top_features

        except Exception as e:
            predictions[target] = {
                "probability": 0,
                "risk_level": "Low",
                "confidence": 0
            }
            shap_info_out[target] = []

    overall_risk = np.mean(overall_risks) if overall_risks else 0

    cols_to_plot = [c for c in df.columns if c not in ['Patient_ID','Day'] + TARGETS]
    plots = {col: generate_plot(df, col) for col in cols_to_plot}

    patient_summary = {
        "total_records": len(df),
        "data_quality": 100,
        "last_update": df['Day'].iloc[-1] if 'Day' in df.columns else "N/A",
        "age": int(df['Age'].iloc[-1]) if 'Age' in df.columns else "N/A",
        "gender": df['Gender'].iloc[-1] if 'Gender' in df.columns else "N/A",
        "name": df['Name'].iloc[-1] if 'Name' in df.columns else f"Patient {patient_id}"
    }

    risk_gauge = generate_risk_gauge(overall_risk)

    return render_template(
        "patient_detail.html",
        patient_id=patient_id,
        df=df.tail(10).to_dict(orient='records'),
        predictions=predictions,
        shap_info=shap_info_out,
        plots=plots,
        overall_risk=overall_risk,
        patient_summary=patient_summary,
        risk_gauge=risk_gauge
    )

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    patient_list = []
    for pid, df in patients.items():
        X_pred = feature_engineering(df)
        risk_vals = []

        for target, model in models.items():
            try:
                expected_cols = model.get_booster().feature_names
                for c in expected_cols:
                    if c not in X_pred.columns:
                        X_pred[c] = 0
                X_pred_ordered = X_pred[expected_cols]
                prob = model.predict_proba(X_pred_ordered)[0,1]
                risk_vals.append(prob)
            except:
                continue

        risk_score = np.mean(risk_vals) if risk_vals else 0
        if risk_score > 0.66:
            risk_level = "High"
        elif risk_score > 0.33:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        patient_list.append({
            "id": pid,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "age": int(df['Age'].iloc[-1]) if 'Age' in df.columns else 0,
            "last_visit": df['Day'].iloc[-1] if 'Day' in df.columns else "N/A",
            "conditions": {t: models[t].predict_proba(feature_engineering(df)[models[t].get_booster().feature_names])[0,1]
                           if t in models else 0 for t in TARGETS},
            "alert": risk_score > 0.66
        })

    stats = {
        "total": len(patient_list),
        "high_risk": sum(1 for p in patient_list if p["risk_level"]=="High"),
        "medium_risk": sum(1 for p in patient_list if p["risk_level"]=="Medium"),
        "low_risk": sum(1 for p in patient_list if p["risk_level"]=="Low"),
        "models_loaded": len(models)
    }

    risk_distribution = {
        "Low": stats["low_risk"],
        "Medium": stats["medium_risk"],
        "High": stats["high_risk"]
    }

    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        if patient_id in patients:
            return redirect(url_for("patient_detail", patient_id=patient_id))
        else:
            flash("Patient not found!", "danger")

    return render_template(
        "dashboard.html",
        patients=patient_list,
        stats=stats,
        risk_distribution=risk_distribution
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

