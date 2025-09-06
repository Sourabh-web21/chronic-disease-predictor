import os
import sys
import traceback

print("=== STARTING APP INITIALIZATION ===")

try:
    import joblib
    print("✓ joblib imported successfully")
except Exception as e:
    print(f"✗ joblib import failed: {e}")

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except Exception as e:
    print(f"✗ pandas import failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except Exception as e:
    print(f"✗ numpy import failed: {e}")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for server
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported successfully")
except Exception as e:
    print(f"✗ matplotlib import failed: {e}")

try:
    import shap
    print("✓ shap imported successfully")
except Exception as e:
    print(f"✗ shap import failed: {e}")

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

print(f"Base directory: {BASE_DIR}")
print(f"Database path: {DB_PATH}")
print(f"Model folder: {MODEL_FOLDER}")
print(f"Database exists: {os.path.exists(DB_PATH)}")
print(f"Model folder exists: {os.path.exists(MODEL_FOLDER)}")

# List all files in current directory
print(f"Files in current directory: {os.listdir('.')}")
if os.path.exists(MODEL_FOLDER):
    print(f"Files in models folder: {os.listdir(MODEL_FOLDER)}")

# -----------------------------
# Load models with error handling
# -----------------------------
models = {}
for target in TARGETS:
    try:
        path = os.path.join(MODEL_FOLDER, f"{target}.joblib")
        print(f"Attempting to load model: {path}")
        if os.path.exists(path):
            models[target] = joblib.load(path)
            print(f"✓ Successfully loaded model: {target}")
        else:
            print(f"✗ Model file not found: {path}")
    except Exception as e:
        print(f"✗ Error loading model {target}: {e}")
        print(f"✗ Traceback: {traceback.format_exc()}")

print(f"Total models loaded: {len(models)}")

# -----------------------------
# Load patients from SQLite with error handling
# -----------------------------
patients = {}
try:
    if os.path.exists(DB_PATH):
        print("Database file exists, attempting to connect...")
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        
        # Check if table exists
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        print(f"Tables in database: {tables['name'].tolist() if not tables.empty else 'No tables found'}")
        
        if 'patient_data' in tables['name'].tolist():
            # get all unique Patient_IDs
            pids = pd.read_sql("SELECT DISTINCT Patient_ID FROM patient_data", conn)["Patient_ID"].astype(str).tolist()
            print(f"Found {len(pids)} unique patients in database")

            for pid in pids:
                try:
                    df = pd.read_sql(f"SELECT * FROM patient_data WHERE Patient_ID = ?", conn, params=(pid,))
                    df['Patient_ID'] = df['Patient_ID'].astype(str)
                    if 'Day' not in df.columns:
                        df['Day'] = range(len(df))
                    patients[pid] = df
                    print(f"✓ Loaded patient: {pid}")
                except Exception as e:
                    print(f"✗ Error loading patient {pid}: {e}")

            conn.close()
            print(f"✓ Successfully loaded {len(patients)} patients")
        else:
            print("✗ patient_data table not found in database")
            conn.close()
    else:
        print("✗ Database file does not exist")
        
except Exception as e:
    print(f"✗ Database loading error: {e}")
    print(f"✗ Traceback: {traceback.format_exc()}")

print(f"Total patients loaded: {len(patients)}")

# -----------------------------
# Helper functions with error handling
# -----------------------------
def generate_risk_gauge(risk_score):
    try:
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
    except Exception as e:
        print(f"Error generating risk gauge: {e}")
        return ""

def feature_engineering(df):
    try:
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
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return pd.DataFrame()

def generate_plot(df, col):
    try:
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
    except Exception as e:
        print(f"Error generating plot for {col}: {e}")
        return ""

# -----------------------------
# Debug route
# -----------------------------
@app.route('/debug')
def debug():
    debug_info = {
        'base_dir': BASE_DIR,
        'db_path': DB_PATH,
        'model_folder': MODEL_FOLDER,
        'db_exists': os.path.exists(DB_PATH),
        'model_folder_exists': os.path.exists(MODEL_FOLDER),
        'current_dir_files': os.listdir('.'),
        'model_files': os.listdir(MODEL_FOLDER) if os.path.exists(MODEL_FOLDER) else [],
        'models_loaded': list(models.keys()),
        'patients_loaded': len(patients),
        'patient_ids': list(patients.keys())[:10],  # First 10
        'python_version': sys.version,
        'working_directory': os.getcwd()
    }
    return debug_info

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            patient_id = request.form.get("patient_id")
            if patient_id in patients:
                return redirect(url_for("patient_detail", patient_id=patient_id))
            else:
                flash("Patient ID not found!", "danger")

        patient_list = []
        for pid, df in patients.items():
            try:
                X_pred = feature_engineering(df)
                if X_pred.empty:
                    continue
                    
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
                    except Exception as e:
                        print(f"Error predicting for patient {pid}, model {target}: {e}")
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
            except Exception as e:
                print(f"Error processing patient {pid}: {e}")
                continue

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
    except Exception as e:
        print(f"Error in index route: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return f"Error in index: {str(e)}", 500

@app.route('/health')
def health():
    return {
        'status': 'healthy',
        'models_loaded': len(models),
        'patients_loaded': len(patients)
    }

# Keep your other routes but add similar error handling...
@app.route("/upload", methods=["POST"])
def upload_csv():
    try:
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
    except Exception as e:
        print(f"Error in upload: {e}")
        flash(f"Upload error: {str(e)}", "danger")
        return redirect(url_for("index"))

@app.route("/patient/<patient_id>")
def patient_detail(patient_id):
    try:
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
                print(f"Error processing model {target} for patient {patient_id}: {e}")
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
    except Exception as e:
        print(f"Error in patient_detail: {e}")
        return f"Error loading patient details: {str(e)}", 500

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    try:
        patient_list = []
        for pid, df in patients.items():
            try:
                X_pred = feature_engineering(df)
                if X_pred.empty:
                    continue
                    
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
                    except Exception as e:
                        print(f"Error predicting for patient {pid}, model {target}: {e}")
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
            except Exception as e:
                print(f"Error processing patient {pid} for dashboard: {e}")
                continue

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
    except Exception as e:
        print(f"Error in dashboard: {e}")
        return f"Error loading dashboard: {str(e)}", 500

print("=== APP INITIALIZATION COMPLETE ===")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)