from fastapi import FastAPI
import pandas as pd
import sqlite3

app = FastAPI()

# Load CSV into SQLite
df = pd.read_csv("transaction_trains.csv")
conn = sqlite3.connect("fraud_detection.db")
df.to_sql("transactions", conn, if_exists="replace", index=False)

@app.get("/api/transactions")
def get_transactions():
    conn = sqlite3.connect("fraud_detection.db")
    df = pd.read_sql("SELECT * FROM transactions LIMIT 10", conn)
    return df.to_dict(orient="records")

@app.get("/api/fraud_detection/{transaction_id}")
def detect_fraud(transaction_id: str):
    conn = sqlite3.connect("fraud_detection.db")
    df = pd.read_sql(f"SELECT * FROM transactions WHERE transaction_id_anonymous='{transaction_id}'", conn)
    
    if df.empty:
        return {"message": "Transaction not found"}
    
    # Basic rule-based fraud detection
    is_fraud = df["transaction_amount"].iloc[0] > 1000000  # Example rule
    return {
        "transaction_id": transaction_id,
        "is_fraud": is_fraud,
        "fraud_reason": "High amount" if is_fraud else "Normal transaction"
    }

