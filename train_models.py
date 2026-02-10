"""
ML Model Training Pipeline for Waterborne Disease Prediction
Trains 5 models with comprehensive visualization graphs.
  1. Disease Classification (XGBoost + LightGBM ensemble)
  2. Water Quality Index Prediction (XGBoost Regressor)
  3. Disease Outbreak Risk (LightGBM Binary)
  4. Water Safety Classification (XGBoost 3-class)
  5. Sanitation Risk Score (XGBoost Regressor)

Usage: python train_models.py
"""
import numpy as np
import pandas as pd
import time, os, gc, warnings
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score, r2_score, mean_absolute_error,
                             mean_squared_error, confusion_matrix, roc_curve,
                             precision_recall_curve)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
CSV_PATH = os.path.join(BASE_DIR, "waterborne_disease_dataset.csv")

# GPU Detection (L4 / CUDA)
def detect_gpu():
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  \U0001f680 GPU DETECTED — Using CUDA acceleration")
            for line in result.stdout.split("\n"):
                if "NVIDIA" in line and "MiB" in line:
                    print(f"    {line.strip()}")
            return True
    except Exception:
        pass
    print("  CPU mode — no GPU detected")
    return False

USE_GPU = detect_gpu()

# Device configs
XGB_DEVICE = {"device": "cuda", "tree_method": "hist"} if USE_GPU else {"tree_method": "hist"}
LGBM_DEVICE = {"device": "gpu"} if USE_GPU else {}

# Plot style
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]


def save_plot(fig, name):
    """Save plot and close figure."""
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    📊 Saved: plots/{name}")


# ============================================================
# DATASET-LEVEL VISUALIZATIONS
# ============================================================
def plot_dataset_overview(df, le_dict):
    print("\n  Generating dataset overview plots...")

    # 1. Disease Distribution
    disease_names = le_dict["disease"].classes_
    disease_counts = df["disease"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(disease_names, [disease_counts.get(i, 0) for i in range(len(disease_names))],
                  color=COLORS[:len(disease_names)], edgecolor="black", linewidth=0.5)
    ax.set_title("Disease Class Distribution (5.25M Records)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Disease", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    for bar, name in zip(bars, disease_names):
        pct = bar.get_height() / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_plot(fig, "01_disease_distribution.png")

    # 2. Feature Correlation Heatmap (numeric features)
    numeric_cols = ["water_quality_index", "ph", "turbidity_ntu", "dissolved_oxygen_mg_l",
                    "bod_mg_l", "fecal_coliform_per_100ml", "total_coliform_per_100ml",
                    "tds_mg_l", "open_defecation_rate", "sewage_treatment_pct",
                    "avg_temperature_c", "avg_rainfall_mm", "age"]
    sample = df[numeric_cols].sample(100000, random_state=42)
    corr = sample.corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "02_correlation_heatmap.png")

    # 3. Water Quality by Disease (Box Plot)
    sample2 = df[["water_quality_index", "disease"]].sample(200000, random_state=42)
    sample2["disease_name"] = sample2["disease"].map(dict(enumerate(disease_names)))
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=sample2, x="disease_name", y="water_quality_index", palette=COLORS, ax=ax)
    ax.set_title("Water Quality Index by Disease", fontsize=16, fontweight="bold")
    ax.set_xlabel("Disease", fontsize=12)
    ax.set_ylabel("Water Quality Index", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_plot(fig, "03_wqi_by_disease.png")

    # 4. Season vs Disease Heatmap
    season_names_map = dict(enumerate(le_dict["season"].classes_))
    cross = pd.crosstab(
        df["disease"].map(dict(enumerate(disease_names))),
        df["season"].map(season_names_map),
        normalize="index"
    ) * 100
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cross, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title("Disease Prevalence by Season (%)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Disease", fontsize=12)
    plt.tight_layout()
    save_plot(fig, "04_disease_by_season.png")

    # 5. Geographic Distribution
    region_names_map = dict(enumerate(le_dict["region"].classes_))
    region_disease = pd.crosstab(
        df["region"].map(region_names_map),
        df["disease"].map(dict(enumerate(disease_names)))
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    region_disease.plot(kind="bar", stacked=True, ax=ax, color=COLORS, edgecolor="black", linewidth=0.3)
    ax.set_title("Disease Cases by Region", fontsize=16, fontweight="bold")
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(title="Disease", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_plot(fig, "05_disease_by_region.png")


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================
def load_and_preprocess():
    print("=" * 60)
    print("  LOADING & PREPROCESSING DATA")
    print("=" * 60)
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {df.shape[0]:,} rows x {df.shape[1]} cols in {time.time()-t0:.1f}s")

    le_dict = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"  Encoding {len(cat_cols)} categorical columns...")
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        print(f"    {col}: {len(le.classes_)} classes")

    joblib.dump(le_dict, os.path.join(MODEL_DIR, "label_encoders.joblib"))
    print(f"  Done in {time.time()-t0:.1f}s\n")
    return df, le_dict


# ============================================================
# MODEL 1: DISEASE CLASSIFICATION
# ============================================================
def train_disease_classifier(df, le_dict):
    print("=" * 60)
    print("  MODEL 1: DISEASE CLASSIFICATION")
    print("=" * 60)

    X = df.drop("disease", axis=1)
    y = df["disease"]
    disease_names = le_dict["disease"].classes_
    print(f"  Classes: {len(disease_names)}, Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # XGBoost
    print("\n  [1/3] Training XGBoost...")
    t = time.time()
    xgb = XGBClassifier(
        n_estimators=500, max_depth=14, learning_rate=0.08,
        min_child_weight=5, gamma=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42,
        eval_metric="mlogloss", verbosity=0,
        **XGB_DEVICE
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"  XGBoost Accuracy: {xgb_acc*100:.2f}% ({time.time()-t:.1f}s)")

    # LightGBM
    print("  [2/3] Training LightGBM...")
    t = time.time()
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=14, learning_rate=0.08,
        min_child_samples=20, num_leaves=127,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbose=-1,
        **LGBM_DEVICE
    )
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_test)
    lgbm_acc = accuracy_score(y_test, lgbm_pred)
    print(f"  LightGBM Accuracy: {lgbm_acc*100:.2f}% ({time.time()-t:.1f}s)")

    # Ensemble
    print("  [3/3] Ensemble...")
    xgb_proba = xgb.predict_proba(X_test)
    lgbm_proba = lgbm.predict_proba(X_test)
    xgb_classes = xgb.classes_
    lgbm_classes = lgbm.classes_
    if not np.array_equal(xgb_classes, lgbm_classes):
        reorder = [np.where(lgbm_classes == c)[0][0] for c in xgb_classes]
        lgbm_proba = lgbm_proba[:, reorder]
    ensemble_proba = xgb_proba * 0.55 + lgbm_proba * 0.45
    ensemble_pred = xgb_classes[np.argmax(ensemble_proba, axis=1)]
    ensemble_acc = accuracy_score(y_test, ensemble_pred)

    print(f"\n  {'='*50}")
    print(f"  XGBoost   : {xgb_acc*100:.2f}%")
    print(f"  LightGBM  : {lgbm_acc*100:.2f}%")
    print(f"  ENSEMBLE  : {ensemble_acc*100:.2f}%")
    print(f"  {'='*50}\n")
    print(classification_report(y_test, ensemble_pred, target_names=disease_names))

    imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("  Top 10 features:")
    for feat, val in imp.head(10).items():
        print(f"    {feat:35s}: {val:.4f}")

    # --- PLOTS ---
    print("\n  Generating Model 1 plots...")

    # Confusion Matrix
    cm = confusion_matrix(y_test, ensemble_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", xticklabels=disease_names,
                yticklabels=disease_names, linewidths=0.5, ax=ax)
    ax.set_title(f"Disease Classification — Confusion Matrix (Acc: {ensemble_acc*100:.2f}%)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_plot(fig, "06_disease_confusion_matrix.png")

    # Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["XGBoost", "LightGBM", "Ensemble"]
    accs = [xgb_acc*100, lgbm_acc*100, ensemble_acc*100]
    bars = ax.bar(models, accs, color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                  edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylim(min(accs) - 2, max(accs) + 2)
    ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    plt.tight_layout()
    save_plot(fig, "07_model_comparison.png")

    # Feature Importance (Top 20)
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = imp.head(20)
    ax.barh(range(len(top20)), top20.values, color="#1f77b4", edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_title("Disease Classification — Top 20 Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=12)
    plt.tight_layout()
    save_plot(fig, "08_disease_feature_importance.png")

    # Per-Class Accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(disease_names, per_class_acc, color=COLORS[:len(disease_names)],
                  edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Per-Class Classification Accuracy", fontsize=16, fontweight="bold")
    ax.set_xlabel("Disease", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_plot(fig, "09_per_class_accuracy.png")

    joblib.dump(xgb, os.path.join(MODEL_DIR, "disease_xgboost.joblib"))
    joblib.dump(lgbm, os.path.join(MODEL_DIR, "disease_lightgbm.joblib"))
    print(f"  ✅ Models saved\n")
    return xgb_acc, lgbm_acc, ensemble_acc


# ============================================================
# MODEL 2: WATER QUALITY INDEX PREDICTION
# ============================================================
def train_wqi_predictor(df):
    print("=" * 60)
    print("  MODEL 2: WQI PREDICTION")
    print("=" * 60)

    feature_cols = [
        "ph", "turbidity_ntu", "dissolved_oxygen_mg_l", "bod_mg_l",
        "fecal_coliform_per_100ml", "total_coliform_per_100ml",
        "tds_mg_l", "nitrate_mg_l", "fluoride_mg_l", "arsenic_ug_l",
        "region", "is_urban", "month", "season",
        "avg_temperature_c", "avg_rainfall_mm", "avg_humidity_pct"
    ]

    X = df[feature_cols]
    y = df["water_quality_index"]
    print(f"  Features: {len(feature_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("  Training XGBoost Regressor...")
    t = time.time()
    model = XGBRegressor(
        n_estimators=400, max_depth=12, learning_rate=0.08,
        min_child_weight=5, gamma=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbosity=0,
        **XGB_DEVICE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n  {'='*50}")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : {mae:.2f}")
    print(f"  RMSE     : {rmse:.2f}")
    print(f"  Time     : {time.time()-t:.1f}s")
    print(f"  {'='*50}")

    # --- PLOTS ---
    print("\n  Generating Model 2 plots...")
    sample_idx = np.random.RandomState(42).choice(len(y_test), 50000, replace=False)
    yt_s = y_test.values[sample_idx]
    yp_s = y_pred[sample_idx]

    # Actual vs Predicted Scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(yt_s, yp_s, alpha=0.1, s=3, c="#1f77b4")
    ax.plot([0, 100], [0, 100], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_title(f"WQI — Actual vs Predicted (R²={r2:.4f})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual WQI", fontsize=12)
    ax.set_ylabel("Predicted WQI", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    save_plot(fig, "10_wqi_actual_vs_predicted.png")

    # Residual Distribution
    residuals = yt_s - yp_s
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=80, color="#1f77b4", edgecolor="black", linewidth=0.3, alpha=0.8)
    ax.axvline(0, color="red", linewidth=2, linestyle="--")
    ax.set_title("WQI Prediction — Residual Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Residual (Actual - Predicted)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.text(0.02, 0.95, f"MAE={mae:.2f}\nRMSE={rmse:.2f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat"))
    plt.tight_layout()
    save_plot(fig, "11_wqi_residuals.png")

    # Feature Importance
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(imp)), imp.values, color="#ff7f0e", edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_title("WQI Prediction — Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=12)
    plt.tight_layout()
    save_plot(fig, "12_wqi_feature_importance.png")

    joblib.dump(model, os.path.join(MODEL_DIR, "wqi_xgboost.joblib"))
    print(f"  ✅ Model saved\n")
    return r2, mae, rmse


# ============================================================
# MODEL 3: DISEASE OUTBREAK RISK
# ============================================================
def train_outbreak_predictor(df, le_dict):
    print("=" * 60)
    print("  MODEL 3: OUTBREAK RISK")
    print("=" * 60)

    no_disease_code = le_dict["disease"].transform(["No_Disease"])[0]
    disease_cases = df[df["disease"] != no_disease_code]
    print(f"  Disease cases: {disease_cases.shape[0]:,}")

    case_counts = disease_cases.groupby(["state", "district", "month"]).size().reset_index(name="case_count")
    state_stats = case_counts.groupby("state")["case_count"].agg(["mean", "std"]).reset_index()
    state_stats.columns = ["state", "state_mean", "state_std"]
    state_stats["state_std"] = state_stats["state_std"].fillna(0)
    case_counts = case_counts.merge(state_stats, on="state")
    case_counts["outbreak"] = (case_counts["case_count"] > (case_counts["state_mean"] + 0.5 * case_counts["state_std"])).astype(int)

    agg = df.groupby(["state", "district", "month"]).agg(
        region=("region", "first"), is_urban=("is_urban", "mean"),
        pop_density=("population_density", "mean"), wqi=("water_quality_index", "mean"),
        ph=("ph", "mean"), turbidity=("turbidity_ntu", "mean"),
        do_val=("dissolved_oxygen_mg_l", "mean"), bod=("bod_mg_l", "mean"),
        fc=("fecal_coliform_per_100ml", "mean"), tc=("total_coliform_per_100ml", "mean"),
        tds=("tds_mg_l", "mean"), nitrate=("nitrate_mg_l", "mean"),
        fluoride=("fluoride_mg_l", "mean"), arsenic=("arsenic_ug_l", "mean"),
        odr=("open_defecation_rate", "mean"), toilet=("toilet_access", "mean"),
        sewage=("sewage_treatment_pct", "mean"), temp=("avg_temperature_c", "mean"),
        rain=("avg_rainfall_mm", "mean"), humidity=("avg_humidity_pct", "mean"),
        flooding=("flooding", "mean"), season=("season", "first"),
    ).reset_index()

    merged = agg.merge(case_counts[["state", "district", "month", "outbreak"]],
                       on=["state", "district", "month"], how="left")
    merged["outbreak"] = merged["outbreak"].fillna(0).astype(int)

    print(f"  Samples: {merged.shape[0]:,} | Outbreak=1: {merged['outbreak'].sum():,} ({merged['outbreak'].mean()*100:.1f}%)")

    feat_cols = [c for c in merged.columns if c not in ["state", "district", "month", "outbreak"]]
    X = merged[feat_cols]
    y = merged["outbreak"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("  Training LightGBM...")
    t = time.time()
    model = LGBMClassifier(
        n_estimators=300, max_depth=10, learning_rate=0.08,
        num_leaves=63, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbose=-1, is_unbalance=True,
        **LGBM_DEVICE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = 0.0

    print(f"\n  {'='*50}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1       : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Time     : {time.time()-t:.1f}s")
    print(f"  {'='*50}")

    # --- PLOTS ---
    print("\n  Generating Model 3 plots...")

    # ROC Curve
    if auc > 0:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color="#d62728", linewidth=2.5, label=f"LightGBM (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.15, color="#d62728")
        ax.set_title("Outbreak Risk — ROC Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.legend(fontsize=12)
        plt.tight_layout()
        save_plot(fig, "13_outbreak_roc_curve.png")

    # Precision-Recall Curve
    if auc > 0:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(rec, prec, color="#9467bd", linewidth=2.5)
        ax.fill_between(rec, prec, alpha=0.15, color="#9467bd")
        ax.set_title("Outbreak Risk — Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        plt.tight_layout()
        save_plot(fig, "14_outbreak_precision_recall.png")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=["No Outbreak", "Outbreak"],
                yticklabels=["No Outbreak", "Outbreak"], linewidths=0.5, ax=ax)
    ax.set_title(f"Outbreak Risk — Confusion Matrix (Acc: {acc*100:.1f}%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    save_plot(fig, "15_outbreak_confusion_matrix.png")

    joblib.dump(model, os.path.join(MODEL_DIR, "outbreak_lightgbm.joblib"))
    print(f"  ✅ Model saved\n")
    del merged, agg, case_counts; gc.collect()
    return acc, f1, auc


# ============================================================
# MODEL 4: WATER SAFETY CLASSIFICATION
# ============================================================
def train_water_safety_classifier(df):
    print("=" * 60)
    print("  MODEL 4: WATER SAFETY CLASSIFICATION")
    print("=" * 60)

    safe_mask = ((df["water_quality_index"] >= 60) &
                 (df["fecal_coliform_per_100ml"] < 100) &
                 (df["turbidity_ntu"] < 10))
    moderate_mask = (~safe_mask &
                     (df["water_quality_index"] >= 30) &
                     (df["fecal_coliform_per_100ml"] < 500))
    water_safety = np.where(safe_mask, 0, np.where(moderate_mask, 1, 2))
    safety_names = ["Safe", "Moderate", "Dangerous"]

    for i, name in enumerate(safety_names):
        cnt = (water_safety == i).sum()
        print(f"    {name:12s}: {cnt:>10,} ({cnt/len(df)*100:.1f}%)")

    feature_cols = [
        "ph", "turbidity_ntu", "dissolved_oxygen_mg_l", "bod_mg_l",
        "fecal_coliform_per_100ml", "total_coliform_per_100ml",
        "tds_mg_l", "nitrate_mg_l", "fluoride_mg_l", "arsenic_ug_l",
        "water_source", "water_treatment"
    ]
    X = df[feature_cols]
    y = water_safety
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("  Training XGBoost...")
    t = time.time()
    model = XGBClassifier(
        n_estimators=400, max_depth=12, learning_rate=0.08,
        min_child_weight=5, gamma=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42,
        eval_metric="mlogloss", verbosity=0,
        **XGB_DEVICE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  {'='*50}")
    print(f"  ACCURACY: {acc*100:.2f}%")
    print(f"  Time    : {time.time()-t:.1f}s")
    print(f"  {'='*50}")
    print(classification_report(y_test, y_pred, target_names=safety_names))

    # --- PLOTS ---
    print("  Generating Model 4 plots...")
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Greens",
                xticklabels=safety_names, yticklabels=safety_names, linewidths=0.5, ax=ax)
    ax.set_title(f"Water Safety — Confusion Matrix (Acc: {acc*100:.2f}%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    save_plot(fig, "16_water_safety_confusion_matrix.png")

    # Safety Distribution Pie
    fig, ax = plt.subplots(figsize=(8, 8))
    counts = [np.sum(water_safety == i) for i in range(3)]
    colors_pie = ["#2ca02c", "#ff7f0e", "#d62728"]
    ax.pie(counts, labels=safety_names, colors=colors_pie, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 13, "fontweight": "bold"},
           wedgeprops={"edgecolor": "black", "linewidth": 0.5})
    ax.set_title("Water Safety Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "17_water_safety_distribution.png")

    joblib.dump(model, os.path.join(MODEL_DIR, "water_safety_xgboost.joblib"))
    print(f"  ✅ Model saved\n")
    return acc


# ============================================================
# MODEL 5: SANITATION RISK SCORE
# ============================================================
def train_sanitation_risk_predictor(df, le_dict):
    print("=" * 60)
    print("  MODEL 5: SANITATION RISK SCORE")
    print("=" * 60)

    hw_le = le_dict["handwashing_practice"]
    hw_risk_map = {}
    for cls in hw_le.classes_:
        enc = hw_le.transform([cls])[0]
        hw_risk_map[enc] = {"Always": 0.0, "Sometimes": 0.5}.get(cls, 1.0)

    hw_risk = df["handwashing_practice"].map(hw_risk_map).fillna(0.5)
    risk_score = np.clip(
        df["open_defecation_rate"] * 0.35 +
        (1 - df["toilet_access"]) * 25.0 +
        (100 - df["sewage_treatment_pct"]) * 0.25 +
        hw_risk * 15.0, 0, 100
    )
    print(f"  Risk: min={risk_score.min():.1f}, max={risk_score.max():.1f}, mean={risk_score.mean():.1f}")

    feature_cols = [
        "state", "district", "region", "is_urban", "population_density",
        "latitude", "longitude", "water_quality_index",
        "fecal_coliform_per_100ml", "total_coliform_per_100ml", "tds_mg_l",
        "avg_temperature_c", "avg_rainfall_mm", "avg_humidity_pct", "month", "season"
    ]
    X = df[feature_cols]
    y = risk_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("  Training XGBoost Regressor...")
    t = time.time()
    model = XGBRegressor(
        n_estimators=400, max_depth=12, learning_rate=0.08,
        min_child_weight=5, gamma=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbosity=0,
        **XGB_DEVICE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n  {'='*50}")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : {mae:.2f}")
    print(f"  RMSE     : {rmse:.2f}")
    print(f"  Time     : {time.time()-t:.1f}s")
    print(f"  {'='*50}")

    # --- PLOTS ---
    print("\n  Generating Model 5 plots...")
    sample_idx = np.random.RandomState(42).choice(len(y_test), 50000, replace=False)
    yt_s = y_test.values[sample_idx]
    yp_s = y_pred[sample_idx]

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(yt_s, yp_s, alpha=0.1, s=3, c="#2ca02c")
    ax.plot([0, 100], [0, 100], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_title(f"Sanitation Risk — Actual vs Predicted (R²={r2:.4f})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Risk Score", fontsize=12)
    ax.set_ylabel("Predicted Risk Score", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    save_plot(fig, "18_sanitation_actual_vs_predicted.png")

    # Feature Importance
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(imp)), imp.values, color="#2ca02c", edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_title("Sanitation Risk — Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=12)
    plt.tight_layout()
    save_plot(fig, "19_sanitation_feature_importance.png")

    joblib.dump(model, os.path.join(MODEL_DIR, "sanitation_risk_xgboost.joblib"))
    print(f"  ✅ Model saved\n")
    return r2, mae, rmse


# ============================================================
# FINAL SUMMARY PLOT
# ============================================================
def plot_final_summary(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Classification Accuracies
    ax = axes[0]
    names = ["Disease\n(Ensemble)", "Water\nSafety"]
    accs = [results["ens_acc"]*100, results["ws_acc"]*100]
    bars = ax.bar(names, accs, color=["#1f77b4", "#2ca02c"], edgecolor="black", linewidth=0.5, width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.2f}%", ha="center", fontsize=14, fontweight="bold")
    ax.set_title("Classification Accuracies", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)

    # Regression R² Scores
    ax = axes[1]
    names = ["WQI\nPrediction", "Sanitation\nRisk"]
    r2s = [results["wqi_r2"], results["san_r2"]]
    bars = ax.bar(names, r2s, color=["#ff7f0e", "#2ca02c"], edgecolor="black", linewidth=0.5, width=0.5)
    for bar, r2 in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{r2:.4f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_title("Regression R² Scores", fontsize=14, fontweight="bold")
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_ylim(0, 1.1)

    fig.suptitle("🏆 Overall Model Performance Summary", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot(fig, "20_final_summary.png")


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()
    df, le_dict = load_and_preprocess()

    # Dataset overview plots
    plot_dataset_overview(df, le_dict)

    # Train all models
    xgb_acc, lgbm_acc, ens_acc = train_disease_classifier(df, le_dict); gc.collect()
    wqi_r2, wqi_mae, wqi_rmse = train_wqi_predictor(df); gc.collect()
    out_acc, out_f1, out_auc = train_outbreak_predictor(df, le_dict); gc.collect()
    ws_acc = train_water_safety_classifier(df); gc.collect()
    san_r2, san_mae, san_rmse = train_sanitation_risk_predictor(df, le_dict); gc.collect()

    # Final summary plot
    plot_final_summary({
        "ens_acc": ens_acc, "ws_acc": ws_acc,
        "wqi_r2": wqi_r2, "san_r2": san_r2
    })

    # Print summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("  🏆 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f} min)\n")
    print(f"  {'#':<4} {'Model':<32} {'Metric':<12} {'Score':>10}")
    print(f"  {'-'*60}")
    print(f"  {'1.':<4} {'Disease (XGBoost)':<32} {'Accuracy':<12} {xgb_acc*100:>9.2f}%")
    print(f"  {'1.':<4} {'Disease (LightGBM)':<32} {'Accuracy':<12} {lgbm_acc*100:>9.2f}%")
    print(f"  {'1.':<4} {'Disease (ENSEMBLE)':<32} {'Accuracy':<12} {ens_acc*100:>9.2f}%")
    print(f"  {'2.':<4} {'WQI Prediction':<32} {'R²':<12} {wqi_r2:>10.4f}")
    print(f"  {'':<4} {'':<32} {'MAE':<12} {wqi_mae:>10.2f}")
    print(f"  {'3.':<4} {'Outbreak Risk':<32} {'AUC-ROC':<12} {out_auc:>10.4f}")
    print(f"  {'':<4} {'':<32} {'F1':<12} {out_f1:>10.4f}")
    print(f"  {'4.':<4} {'Water Safety':<32} {'Accuracy':<12} {ws_acc*100:>9.2f}%")
    print(f"  {'5.':<4} {'Sanitation Risk':<32} {'R²':<12} {san_r2:>10.4f}")
    print(f"  {'':<4} {'':<32} {'MAE':<12} {san_mae:>10.2f}")

    print(f"\n  📁 Models: {MODEL_DIR}")
    for f in sorted(os.listdir(MODEL_DIR)):
        sz = os.path.getsize(os.path.join(MODEL_DIR, f)) / (1024**2)
        print(f"    {f:<45} {sz:>8.1f} MB")

    print(f"\n  📊 Plots ({len(os.listdir(PLOT_DIR))} charts): {PLOT_DIR}")
    for f in sorted(os.listdir(PLOT_DIR)):
        print(f"    {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
