import pandas as pd
import joblib
import shap

MODEL_PATH = "data/xgb_model.pkl"
SCORED_PATH = "data/scored_customers.csv"
EXPLANATION_PATH = "data/shap_values.csv"

def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(SCORED_PATH)
    explainer = shap.Explainer(model)
    shap_values = explainer(df.drop(columns=["risk_score"]))

    shap_df = pd.DataFrame(shap_values.values, columns=df.drop(columns=["risk_score"]).columns)
    shap_df["risk_score"] = df["risk_score"]
    shap_df.to_csv(EXPLANATION_PATH, index=False)
    print(f"✅ SHAP explanations saved to {EXPLANATION_PATH}")

if __name__ == "__main__":
    main()
