"""
Corpus Attribution Experiment.
Train classifier on 11 cognitive features to distinguish Moltbook/OASST/Reddit.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import PROCESSED_DIR, FIGURES_DIR

ALL_FEATURES = [
    # Dependency features (4)
    "mean_dep_distance", "std_dep_distance", "left_arc_ratio", "dep_types_entropy",
    # Lexical diversity (4)
    "hapax_ratio", "distinct_1", "distinct_2", "distinct_3",
    # Cognitive complexity (4)
    "ttr", "emotional_granularity", "syntactic_depth", "flesch_kincaid",
    # Burstiness (1)
    "burstiness",
    # Moral foundations (5)
    "mf_care", "mf_fairness", "mf_loyalty", "mf_authority", "mf_sanctity"
]

# Extended features with POS distributions (10 most discriminative)
EXTENDED_FEATURES = ALL_FEATURES + [
    "pos_VERB", "pos_NOUN", "pos_ADJ", "pos_PRON", "pos_PUNCT",
    "pos_ADP", "pos_ADV", "pos_AUX", "pos_DET", "pos_PROPN"
]

# Comprehensive features with ALL POS tags
COMPREHENSIVE_FEATURES = ALL_FEATURES + [
    "pos_VERB", "pos_NOUN", "pos_ADJ", "pos_PRON", "pos_PUNCT",
    "pos_ADP", "pos_ADV", "pos_AUX", "pos_DET", "pos_PROPN",
    "pos_CCONJ", "pos_SCONJ", "pos_PART", "pos_NUM", "pos_SYM",
    "pos_X", "pos_INTJ", "text_length"
]

# Original 7 features for comparison
FEATURES_7 = [
    "mean_dep_distance", "left_arc_ratio", "dep_types_entropy",
    "hapax_ratio", "distinct_1", "distinct_2", "distinct_3",
]


def load_and_merge_features():
    adv_path = os.path.join(PROCESSED_DIR, "advanced_metrics.csv")
    met_path = os.path.join(PROCESSED_DIR, "metrics_raw.csv")

    adv_df = pd.read_csv(adv_path)
    met_df = pd.read_csv(met_path)

    print(f"Advanced metrics: {len(adv_df)} rows, sources: {adv_df['source'].unique()}")
    print(f"Metrics raw: {len(met_df)} rows, sources: {met_df['source'].unique()}")

    # Merge on source and id
    merged = pd.merge(adv_df, met_df, on=['source', 'id'], how='inner', suffixes=('', '_met'))
    print(f"Merged: {len(merged)} rows")

    return merged


def prepare_three_way_data(df, feature_set="all"):
    if feature_set == "comprehensive":
        feature_list = COMPREHENSIVE_FEATURES
    elif feature_set == "extended":
        feature_list = EXTENDED_FEATURES
    elif feature_set == "all":
        feature_list = ALL_FEATURES
    else:  # "baseline"
        feature_list = FEATURES_7

    feature_cols = [c for c in feature_list if c in df.columns]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    df_clean = df.dropna(subset=feature_cols + ["source"])

    X = df_clean[feature_cols].values
    y = df_clean["source"].values

    return X, y, feature_cols


def run_classification(X, y, feature_names):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\n=== {name} ===")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y_enc, cv=cv, scoring='f1_macro')
        print(f"5-Fold CV Macro F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Test Macro F1: {test_f1:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "test_f1": test_f1,
            "confusion_matrix": cm,
            "model": model,
        }

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            importances = None

        if importances is not None:
            results[name]["feature_importance"] = dict(zip(feature_names, importances))

    return results, le


def plot_confusion_matrix(cm, classes, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Corpus Attribution: Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_feature_importance(importance_dict, filename):
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), values, color='steelblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title('Corpus Attribution: Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def main():
    print("Loading data...")
    df = load_and_merge_features()

    results_dict = {}

    # Run with COMPREHENSIVE features (18 + 17 POS + text_length = 36)
    print("\n" + "="*60)
    print("CORPUS ATTRIBUTION EXPERIMENT (COMPREHENSIVE: 36 FEATURES)")
    print("="*60)

    X_comp, y_comp, feature_names_comp = prepare_three_way_data(df, feature_set="comprehensive")
    print(f"Dataset: {X_comp.shape[0]} samples, {X_comp.shape[1]} features")
    print(f"Class distribution: {pd.Series(y_comp).value_counts().to_dict()}")

    results_comp, le = run_classification(X_comp, y_comp, feature_names_comp)
    best_model_comp = max(results_comp.items(), key=lambda x: x[1]["test_f1"])
    print(f"\n*** Best Model (Comprehensive): {best_model_comp[0]} (Test F1: {best_model_comp[1]['test_f1']:.3f}) ***")

    plot_confusion_matrix(
        best_model_comp[1]["confusion_matrix"],
        le.classes_,
        "attribution_confusion_matrix.pdf"
    )

    if "feature_importance" in best_model_comp[1]:
        plot_feature_importance(
            best_model_comp[1]["feature_importance"],
            "attribution_feature_importance.pdf"
        )

    results_dict["comprehensive_36"] = {
        "best_model": best_model_comp[0],
        "test_f1": best_model_comp[1]["test_f1"],
        "cv_f1_mean": best_model_comp[1]["cv_f1_mean"],
        "cv_f1_std": best_model_comp[1]["cv_f1_std"],
        "n_samples": X_comp.shape[0],
        "n_features": X_comp.shape[1],
        "features": feature_names_comp,
        "all_results": {k: {"cv_f1": v["cv_f1_mean"], "test_f1": v["test_f1"]} for k, v in results_comp.items()}
    }

    # Run with EXTENDED features (28)
    print("\n" + "="*60)
    print("CORPUS ATTRIBUTION EXPERIMENT (EXTENDED: 28 FEATURES)")
    print("="*60)

    X_ext, y_ext, feature_names_ext = prepare_three_way_data(df, feature_set="extended")
    print(f"Dataset: {X_ext.shape[0]} samples, {X_ext.shape[1]} features")

    results_ext, _ = run_classification(X_ext, y_ext, feature_names_ext)
    best_model_ext = max(results_ext.items(), key=lambda x: x[1]["test_f1"])
    print(f"\n*** Best Model (Extended): {best_model_ext[0]} (Test F1: {best_model_ext[1]['test_f1']:.3f}) ***")

    results_dict["extended_28"] = {
        "best_model": best_model_ext[0],
        "test_f1": best_model_ext[1]["test_f1"],
        "cv_f1_mean": best_model_ext[1]["cv_f1_mean"],
        "cv_f1_std": best_model_ext[1]["cv_f1_std"],
        "n_samples": X_ext.shape[0],
        "n_features": X_ext.shape[1],
        "features": feature_names_ext,
        "all_results": {k: {"cv_f1": v["cv_f1_mean"], "test_f1": v["test_f1"]} for k, v in results_ext.items()}
    }

    # Run with ALL features (18)
    print("\n" + "="*60)
    print("CORPUS ATTRIBUTION EXPERIMENT (ALL: 18 FEATURES)")
    print("="*60)

    X_all, y_all, feature_names_all = prepare_three_way_data(df, feature_set="all")
    print(f"Dataset: {X_all.shape[0]} samples, {X_all.shape[1]} features")

    results_all, _ = run_classification(X_all, y_all, feature_names_all)
    best_model_all = max(results_all.items(), key=lambda x: x[1]["test_f1"])
    print(f"\n*** Best Model (18 Features): {best_model_all[0]} (Test F1: {best_model_all[1]['test_f1']:.3f}) ***")

    results_dict["all_18"] = {
        "best_model": best_model_all[0],
        "test_f1": best_model_all[1]["test_f1"],
        "cv_f1_mean": best_model_all[1]["cv_f1_mean"],
        "cv_f1_std": best_model_all[1]["cv_f1_std"],
        "n_samples": X_all.shape[0],
        "n_features": X_all.shape[1],
        "features": feature_names_all,
        "all_results": {k: {"cv_f1": v["cv_f1_mean"], "test_f1": v["test_f1"]} for k, v in results_all.items()}
    }

    # Run with 7 features for comparison
    print("\n" + "="*60)
    print("CORPUS ATTRIBUTION EXPERIMENT (BASELINE: 7 FEATURES)")
    print("="*60)

    X_7, y_7, feature_names_7 = prepare_three_way_data(df, feature_set="baseline")
    print(f"Dataset: {X_7.shape[0]} samples, {X_7.shape[1]} features")

    results_7, _ = run_classification(X_7, y_7, feature_names_7)
    best_model_7 = max(results_7.items(), key=lambda x: x[1]["test_f1"])
    print(f"\n*** Best Model (7 Features): {best_model_7[0]} (Test F1: {best_model_7[1]['test_f1']:.3f}) ***")

    results_dict["baseline_7"] = {
        "best_model": best_model_7[0],
        "test_f1": best_model_7[1]["test_f1"],
        "cv_f1_mean": best_model_7[1]["cv_f1_mean"],
        "cv_f1_std": best_model_7[1]["cv_f1_std"],
        "n_samples": X_7.shape[0],
        "n_features": X_7.shape[1],
        "features": feature_names_7,
        "all_results": {k: {"cv_f1": v["cv_f1_mean"], "test_f1": v["test_f1"]} for k, v in results_7.items()}
    }

    # Print comparison
    print("\n" + "="*60)
    print("FEATURE SET COMPARISON")
    print("="*60)
    for name, res in results_dict.items():
        print(f"{name}: F1={res['test_f1']:.3f} ({res['n_features']} features)")

    summary_path = os.path.join(PROCESSED_DIR, "attribution_results.json")
    with open(summary_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved results to {summary_path}")


if __name__ == "__main__":
    main()
