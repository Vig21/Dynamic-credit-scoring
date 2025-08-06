import pandas as pd
import os

INPUT_PATH = "data/cs-training.csv"
OUTPUT_PATH = "data/processed_data.csv"

def clean_data(df):
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"SeriousDlqin2yrs": "target"})
    df = df.dropna(subset=["MonthlyIncome"])
    df = df[df["age"] > 0]
    df = df[df["DebtRatio"] < 10]
    return df

def main():
    df = pd.read_csv(INPUT_PATH, skiprows=1)
    df = clean_data(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Processed data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
