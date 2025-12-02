"""Training script for the advanced candidate fit score model."""

import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from feature_engineering import build_vectorizer, make_features


def train_pipeline(data_path: str = "data/sample_data.csv") -> Dict:
    # 1. load data
    df = pd.read_csv(data_path)

    # 2. fit TF‑IDF on combined corpus
    combined_corpus = pd.concat([df["resume_text"], df["job_description"]])
    vectorizer = build_vectorizer(combined_corpus)

    # 3. create numeric features
    X, info = make_features(df, vectorizer)
    y = df["fit_score"].values

    # 4. train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 5. choose a reasonably strong model
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 6. evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_val, y_pred)

    print("Evaluation on validation set:")
    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R^2 : {r2:.2f}")

    artifact = {
        "model": model,
        "vectorizer": vectorizer,
    }
    return artifact


def main():
    artifact = train_pipeline()

    # 7. save trained objects
    with open("candidate_fit_advanced.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print("Saved pipeline to candidate_fit_advanced.pkl")


if __name__ == "__main__":
    main()
