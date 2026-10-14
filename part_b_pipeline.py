from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")
SEED = 42
NUMERIC = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate",
           "avg_delivery_minutes", "delivery_unknown", "age", "tenure_days"]
CATEGORICAL = ["city", "gender", "favourite_category"]


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="raise")
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="raise")
    orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="raise")
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    return customers, orders


def build_rfm(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    completed = orders.loc[orders["status"].eq("completed")].copy()
    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    ).reset_index()
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
    rfm["frequency"] = rfm["frequency"].astype(int)
    return rfm


def cluster_segments(rfm: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    values = rfm[["recency", "frequency", "monetary"]].astype(float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    metrics = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = model.fit_predict(scaled)
        metrics.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(scaled, labels)),
        })
    pd.DataFrame(metrics).to_csv(OUT / "kmeans_metrics.csv", index=False)
    chosen_k = 4
    final_model = KMeans(n_clusters=chosen_k, n_init=10, random_state=SEED)
    result = rfm.copy()
    result["segment_id"] = final_model.fit_predict(scaled)
    profile = result.groupby("segment_id").agg(
        size=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    premium_id = int(profile["mean_monetary"].idxmax())
    remaining = profile.drop(index=premium_id)
    frequent_id = int(remaining["mean_frequency"].idxmax())
    remaining = remaining.drop(index=frequent_id)
    at_risk_id = int(remaining["mean_recency"].idxmax())
    occasional_id = int(next(index for index in remaining.index if index != at_risk_id))
    names = {
        premium_id: ("Premium", "Protect loyalty with an exclusive retention touchpoint."),
        frequent_id: ("Frequent Essentials", "Use replenishment reminders and relevant basket bundles."),
        at_risk_id: ("At Risk", "Test a timely win-back offer and monitor the next order."),
        occasional_id: ("Occasional", "Encourage a second monthly order with a focused category offer."),
    }
    result["segment_name"] = result["segment_id"].map({key: value[0] for key, value in names.items()})
    result["is_premium"] = result["segment_id"].eq(premium_id)
    profile["segment_name"] = profile.index.map({key: value[0] for key, value in names.items()})
    profile["recommended_action"] = profile.index.map({key: value[1] for key, value in names.items()})
    profile["is_premium"] = profile.index.eq(premium_id)
    profile.reset_index().to_csv(OUT / "segment_profile.csv", index=False)
    result[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
        OUT / "customer_segments.csv", index=False
    )
    pca = PCA(n_components=2)
    pca.fit(scaled)
    pca_report = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                  "total_explained_variance": float(pca.explained_variance_ratio_.sum())}
    (OUT / "pca_metrics.json").write_text(json.dumps(pca_report, indent=2), encoding="utf-8")
    report = {
        "rfm_customers": int(len(rfm)),
        "recency_median": float(rfm["recency"].median()),
        "recency_min": int(rfm["recency"].min()),
        "recency_max": int(rfm["recency"].max()),
        "frequency_mean": float(rfm["frequency"].mean()),
        "frequency_max": int(rfm["frequency"].max()),
        "monetary_mean": float(rfm["monetary"].mean()),
        "monetary_max": float(rfm["monetary"].max()),
        "chosen_k": chosen_k,
        "premium_segment_id": premium_id,
        "kmeans": metrics,
        "pca": pca_report,
    }
    return result, report


def build_lapse_cohort(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    before = orders.loc[orders["order_ts"].lt(CUTOFF)].copy()
    before_completed = before.loc[before["status"].eq("completed")].copy()
    cohort_ids = before_completed["customer_id"].unique()
    rows = []
    customer_index = customers.set_index("customer_id")
    product_categories = pd.read_csv(CLEAN / "products.csv").set_index("product_id")["category"]
    for customer_id in cohort_ids:
        cust_orders = before.loc[before["customer_id"].eq(customer_id)]
        completed = cust_orders.loc[cust_orders["status"].eq("completed")]
        completed_order_count = int(completed["order_id"].nunique())
        last_order = completed["order_ts"].max()
        favourite = completed["product_id"].map(product_categories).mode()
        delivery_average = completed["delivery_minutes"].mean()
        cust = customer_index.loc[customer_id]
        signup = pd.Timestamp(cust["signup_date"])
        rows.append({
            "customer_id": customer_id,
            "recency": int((CUTOFF - last_order).days),
            "frequency": completed_order_count,
            "monetary": float(completed["order_amount"].sum()),
            "avg_discount_pct": float(completed["discount_pct"].mean()),
            "cancellation_rate": float(cust_orders["status"].eq("cancelled").sum() / cust_orders["status"].isin(["completed", "cancelled"]).sum()),
            "avg_delivery_minutes": 0.0 if pd.isna(delivery_average) else float(delivery_average),
            "delivery_unknown": int(pd.isna(delivery_average)),
            "favourite_category": str(favourite.iloc[0]) if len(favourite) else "Unknown",
            "city": str(cust["city"]),
            "age": float(cust["age"]),
            "gender": str(cust["gender"]),
            "tenure_days": int((CUTOFF - signup).days),
        })
    features = pd.DataFrame(rows)
    later_completed = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(LABEL_END)
    ]
    active_after_cutoff = set(later_completed["customer_id"])
    features["lapsed"] = (~features["customer_id"].isin(active_after_cutoff)).astype(int)
    assert not features[NUMERIC + CATEGORICAL].isna().any().any(), "Feature matrix contains missing values"
    return features


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])


def _positive_metrics(y_true, y_pred) -> dict[str, object]:
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def train_lapse_model(cohort: pd.DataFrame, segments: pd.DataFrame) -> dict[str, object]:
    x = cohort[NUMERIC + CATEGORICAL].copy()
    y = cohort["lapsed"].astype(int)
    assert x.notna().all().all(), "Feature matrix must have no missing values"
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        x, y, cohort["customer_id"], test_size=0.2, stratify=y, random_state=SEED
    )
    baseline_prediction = np.zeros(len(y_test), dtype=int)
    baseline = {
        "accuracy": float(accuracy_score(y_test, baseline_prediction)),
        **_positive_metrics(y_test, baseline_prediction),
    }
    logistic = Pipeline([
        ("preprocess", _preprocessor()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    logistic_grid = GridSearchCV(
        logistic, {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1", n_jobs=1
    )
    logistic_grid.fit(x_train, y_train)
    forest = Pipeline([
        ("preprocess", _preprocessor()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=SEED)),
    ])
    forest_grid = GridSearchCV(
        forest,
        {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
        cv=5, scoring="f1", n_jobs=1,
    )
    forest_grid.fit(x_train, y_train)
    logistic_model = logistic_grid.best_estimator_
    forest_model = forest_grid.best_estimator_
    logistic_test = _positive_metrics(y_test, logistic_model.predict(x_test))
    forest_test = _positive_metrics(y_test, forest_model.predict(x_test))
    selected_name, selected_model = ("random_forest", forest_model)
    if forest_test["f1"] < 0.55 and logistic_test["f1"] >= 0.55:
        selected_name, selected_model = "logistic_regression", logistic_model
    if max(logistic_test["f1"], forest_test["f1"]) < 0.55:
        raise RuntimeError("Neither tuned model meets the required positive-class test F1 of 0.55")

    positive_id = int(y_train.value_counts().idxmin())
    minority = x_train.loc[y_train.eq(positive_id)]
    majority_count = int(y_train.value_counts().max())
    needed = majority_count - len(minority)
    sampled = minority.sample(n=needed, replace=True, random_state=SEED) if needed else minority.iloc[0:0]
    balanced_x = pd.concat([x_train, sampled], ignore_index=True)
    balanced_y = pd.concat([
        y_train.reset_index(drop=True),
        pd.Series([positive_id] * len(sampled), name="lapsed"),
    ], ignore_index=True)
    best_c = logistic_grid.best_params_["model__C"]
    balanced_logistic = Pipeline([
        ("preprocess", _preprocessor()),
        ("model", LogisticRegression(max_iter=1000, C=best_c)),
    ])
    balanced_logistic.fit(balanced_x, balanced_y)
    balanced_test = _positive_metrics(y_test, balanced_logistic.predict(x_test))
    balanced_class_counts = {str(int(label)): int(count) for label, count in balanced_y.value_counts().items()}

    scaler = logistic_model.named_steps["preprocess"].named_transformers_["numeric"]
    numeric_mean = dict(zip(NUMERIC, scaler.mean_.tolist()))
    full_frequency_mean = float(cohort["frequency"].mean())
    forest_pre = forest_model.named_steps["preprocess"]
    feature_names = forest_pre.get_feature_names_out()
    forest_importance = forest_model.named_steps["model"].feature_importances_
    importance = sorted(zip(feature_names, forest_importance), key=lambda item: item[1], reverse=True)[:5]
    logistic_pre = logistic_model.named_steps["preprocess"]
    logistic_names = logistic_pre.get_feature_names_out()
    logistic_coefficients = logistic_model.named_steps["model"].coef_[0]
    coefficient_report = {}
    for feature in ("recency", "frequency", "monetary"):
        name = f"numeric__{feature}"
        coefficient = float(logistic_coefficients[list(logistic_names).index(name)])
        coefficient_report[feature] = {"coefficient": coefficient, "sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero"}

    selected_prediction = selected_model.predict(x_test)
    selected_probability = selected_model.predict_proba(x_test)[:, 1]
    selected_metrics = _positive_metrics(y_test, selected_prediction)
    artifact_path = OUT / "lapse_model.joblib"
    joblib.dump(selected_model, artifact_path)
    reloaded = joblib.load(artifact_path)
    reloaded_prediction = reloaded.predict(x_test)
    reload_identical = bool(np.array_equal(selected_prediction, reloaded_prediction))
    assert reload_identical, "Reloaded model predictions differ"

    premium_ids = set(segments.loc[segments["is_premium"].astype(bool), "customer_id"])
    retention = pd.DataFrame({
        "customer_id": id_test.to_numpy(),
        "lapse_probability": selected_probability,
        "actually_lapsed": y_test.to_numpy(),
    })
    retention = retention.loc[retention["customer_id"].isin(premium_ids)]
    retention = retention.sort_values("lapse_probability", ascending=False).head(10)
    retention.to_csv(OUT / "retention_list.csv", index=False)

    # Train-only resampling and train-only preprocessing are recorded for audit.
    report = {
        "cohort_size": int(len(cohort)),
        "lapsed_count": int(y.sum()),
        "lapse_rate": float(y.mean()),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "feature_columns": NUMERIC + CATEGORICAL,
        "missing_feature_cells": int(x.isna().sum().sum()),
        "baseline": baseline,
        "logistic": {"best_params": logistic_grid.best_params_, "best_cv_f1": float(logistic_grid.best_score_), "test": logistic_test},
        "random_forest": {"best_params": forest_grid.best_params_, "best_cv_f1": float(forest_grid.best_score_), "test": forest_test},
        "selected_model": selected_name,
        "selected_test": selected_metrics,
        "balanced_logistic_test": balanced_test,
        "balanced_train_class_counts": balanced_class_counts,
        "scaler_training_mean_frequency": numeric_mean["frequency"],
        "full_cohort_mean_frequency": full_frequency_mean,
        "random_forest_top_features": [{"feature": str(name), "importance": float(value)} for name, value in importance],
        "logistic_coefficients": coefficient_report,
        "joblib_predictions_identical": reload_identical,
        "retention_list_rows": int(len(retention)),
        "retention_list_lapsed": int(retention["actually_lapsed"].sum()),
        "premium_flag_is_posthoc": True,
        "seed": SEED,
    }
    (OUT / "part_b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    customers, orders = load_clean()
    rfm = build_rfm(customers, orders)
    segments, rfm_report = cluster_segments(rfm)
    cohort = build_lapse_cohort(customers, orders)
    report = {"rfm": rfm_report, "lapse": train_lapse_model(cohort, segments)}
    (OUT / "part_b_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
