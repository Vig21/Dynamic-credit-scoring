import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = "data/scored_customers.csv"
SEGMENTED_PATH = "data/segmented_customers.csv"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
    "risk_score"
]

def main():
    df = pd.read_csv(INPUT_PATH)
    X = df[FEATURES].dropna()

    kmeans = KMeans(n_clusters=3, random_state=42)
    df.loc[X.index, "segment"] = kmeans.fit_predict(X)

    df.to_csv(SEGMENTED_PATH, index=False)
    print(f"✅ Segmented customer data saved to {SEGMENTED_PATH}")

    # Optional: Plot clusters
    sns.pairplot(df.loc[X.index], hue="segment", vars=FEATURES)
    plt.suptitle("Customer Segments Based on Financial Behavior")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
