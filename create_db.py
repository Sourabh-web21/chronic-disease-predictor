import pandas as pd
import sqlite3
import glob

csv_path = "Patients/*.csv"
conn = sqlite3.connect("patients.db")

first = True
for file in glob.glob(csv_path):
    df = pd.read_csv(file)

    # Just append, no need to add patient_id again
    df.to_sql("patient_data", conn, if_exists="replace" if first else "append", index=False)
    first = False

conn.close()
print("✅ All CSVs merged into patients.db")
