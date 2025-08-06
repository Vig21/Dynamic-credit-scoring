import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "data/xgb_model.pkl"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='auc'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    print("AUC:", roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred > 0.5))

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
