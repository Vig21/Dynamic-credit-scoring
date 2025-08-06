import pandas as pd
import joblib

MODEL_PATH = "data/xgb_model.pkl"
NEW_DATA_PATH = "data/new_customers.csv"
SCORED_PATH = "data/scored_customers.csv"

def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(NEW_DATA_PATH)
    df["risk_score"] = model.predict_proba(df)[:, 1]
    df.to_csv(SCORED_PATH, index=False)
    print(f"✅ Scored results saved to {SCORED_PATH}")

if __name__ == "__main__":
    main()
