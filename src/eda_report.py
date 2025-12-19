# src/eda_report.py
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data_loader import load_data


FIG_DIR = Path("reports") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_class_distribution(df: pd.DataFrame):
    counts = df["Class"].value_counts().sort_index()  # 0,1
    labels = ["Normal (0)", "Fraud (1)"]

    plt.figure()
    plt.bar(labels, counts.values) # type: ignore
    plt.title("Class Distribution (Imbalance)")
    plt.ylabel("Count")
    savefig(FIG_DIR / "class_distribution.png")

    fraud_ratio = counts.loc[1] / counts.sum()
    print(f"Fraud ratio: {fraud_ratio:.4f} ({fraud_ratio*100:.2f}%)")


def plot_amount_comparison(df: pd.DataFrame):
    normal = df.loc[df["Class"] == 0, "Amount"].astype(float)
    fraud = df.loc[df["Class"] == 1, "Amount"].astype(float)

    # Log-scale histogram to handle long tail amounts
    plt.figure()
    bins = np.logspace(np.log10(max(1e-3, df["Amount"].min() + 1e-3)), np.log10(df["Amount"].max() + 1), 50)

    plt.hist(normal + 1e-3, bins=bins, alpha=0.7, label="Normal")
    plt.hist(fraud + 1e-3, bins=bins, alpha=0.7, label="Fraud")
    plt.xscale("log")
    plt.title("Transaction Amount Distribution (log scale)")
    plt.xlabel("Amount (log scale)")
    plt.ylabel("Count")
    plt.legend()
    savefig(FIG_DIR / "amount_distribution_log.png")

    print("Amount stats:")
    print("Normal:", normal.describe()[["min", "mean", "50%", "max"]].to_dict())
    print("Fraud :", fraud.describe()[["min", "mean", "50%", "max"]].to_dict())


def plot_correlation_heatmap(df: pd.DataFrame):
    # Correlation on all numeric columns (includes V1-V28, Time, Amount, Class)
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.title("Correlation Matrix Heatmap")
    plt.colorbar()

    cols = corr.columns.tolist()
    plt.xticks(range(len(cols)), cols, rotation=90, fontsize=6)
    plt.yticks(range(len(cols)), cols, fontsize=6)

    savefig(FIG_DIR / "correlation_heatmap.png")


def main():
    df = load_data()

    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    plot_class_distribution(df)
    plot_amount_comparison(df)
    plot_correlation_heatmap(df)

    print("\n✅ EDA complete. Figures saved in reports/figures/")


if __name__ == "__main__":
    main()
