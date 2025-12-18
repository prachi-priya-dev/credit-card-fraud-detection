from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data") / "raw" / "creditcard.csv"
RANDOM_STATE = 42


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put creditcard.csv inside /data"
        )
    df = pd.read_csv(DATA_PATH)
    return df


def load_split_data(test_size=0.2):
    df = load_dataset()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
