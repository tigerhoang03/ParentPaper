import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily sentiment features per (date, ticker).

    Requires columns:
      date, ticker, sent_score, next_day_return
    """

    needed = {"date", "ticker", "sent_score", "next_day_return"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    x = df.copy()

    # MANA-style weight proxy: magnitude of sentiment
    x["weight"] = x["sent_score"].abs().clip(lower=1e-6)

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["weight"].to_numpy()
        s = g["sent_score"].to_numpy()

        weighted_sent = float(np.sum(w * s) / np.sum(w))
        avg_sent = float(np.mean(s))

        y = float(g["next_day_return"].iloc[0])

        return pd.Series(
            {
                "weighted_sent": weighted_sent,
                "avg_sent": avg_sent,
                "n_headlines": int(len(g)),
                "next_day_return": y,
            }
        )

    daily = (
        x.dropna(subset=["sent_score", "next_day_return"])
         .groupby(["date", "ticker"], as_index=False)
         .apply(agg)
         .reset_index(drop=True)
    )

    return daily


def fit_and_report(daily: pd.DataFrame, xcol: str, label: str) -> None:
    daily2 = daily.dropna(subset=[xcol, "next_day_return"]).copy()

    if len(daily2) < 2:
        print(f"{label}: not enough rows to fit.")
        return

    X = daily2[[xcol]].to_numpy()
    y = daily2["next_day_return"].to_numpy()

    reg = LinearRegression().fit(X, y)
    pred = reg.predict(X)

    r2 = r2_score(y, pred)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))

    print(f"\n=== {label} ===")
    print(f"n (daily groups) = {len(daily2)}")
    print(f"coef = {reg.coef_[0]:.6f}")
    print(f"intercept = {reg.intercept_:.6f}")
    print(f"R2 = {r2:.4f}")
    print(f"RMSE = {rmse:.6f}")


def run(split_path: str) -> None:
    df = pd.read_csv(split_path)
    daily = build_daily_features(df)

    print(f"\nLoaded: {split_path}")
    print("Headline count per (date,ticker):")
    print(daily["n_headlines"].describe())

    fit_and_report(daily, "avg_sent", "Baseline: uniform aggregation")
    fit_and_report(daily, "weighted_sent", "MANA-style: magnitude-weighted aggregation")


if __name__ == "__main__":
    for split in ["train.csv", "val.csv", "test.csv"]:
        run(split)