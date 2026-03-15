"""
Telecom Customer Churn Prediction — Full ML Pipeline
Covers: preprocessing, SMOTE, 4 models, evaluation, SHAP, model saving
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import time
import json
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = '/Users/nithinkrishna145/Desktop/churn-prediction-project'
DATA   = f'{BASE}/data/raw/Telco-Customer-Churn.csv'
PROC   = f'{BASE}/data/processed'
VIZ    = f'{BASE}/visualizations'
MODELS = f'{BASE}/models'
ANAL   = f'{BASE}/analysis'

for p in [PROC, VIZ, MODELS, ANAL]:
    os.makedirs(p, exist_ok=True)

sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'savefig.bbox': 'tight',
                     'font.size': 11, 'axes.titlesize': 14})

print("=" * 70)
print("  TELECOM CHURN PREDICTION — ML PIPELINE")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] DATA PREPROCESSING")

df = pd.read_csv(DATA)
print(f"  Raw data shape: {df.shape}")

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing_tc = df['TotalCharges'].isnull().sum()
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
print(f"  Filled {missing_tc} missing TotalCharges with median")

# Drop customerID
df.drop(columns=['customerID'], inplace=True)

# Binary encode simple yes/no columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'Churn']
binary_map  = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
for col in binary_cols:
    df[col] = df[col].map(binary_map)

# One-hot encode remaining categoricals
cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

# Feature engineering
df['AvgMonthlyCharges']  = df['TotalCharges'] / (df['tenure'] + 1)
df['ChargePerTenure']    = df['MonthlyCharges'] / (df['tenure'] + 1)
df['HasStreaming']        = (
    df.get('StreamingTV_Yes', 0) | df.get('StreamingMovies_Yes', 0)
).astype(int)
df['HasSecurity']         = (
    df.get('OnlineSecurity_Yes', 0) | df.get('TechSupport_Yes', 0)
).astype(int)
df['NumAddonServices']    = (
    df.get('OnlineSecurity_Yes', 0) + df.get('OnlineBackup_Yes', 0) +
    df.get('DeviceProtection_Yes', 0) + df.get('TechSupport_Yes', 0) +
    df.get('StreamingTV_Yes', 0) + df.get('StreamingMovies_Yes', 0)
)

print(f"  Encoded shape: {df.shape}")

# Split
X = df.drop(columns=['Churn'])
y = df['Churn']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"  Train churn rate: {y_train.mean()*100:.1f}%")

# Scale numerical features
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'AvgMonthlyCharges', 'ChargePerTenure']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()
X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
X_test_scaled[num_features]  = scaler.transform(X_test[num_features])

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"  After SMOTE — 0: {(y_train_sm==0).sum()} | 1: {(y_train_sm==1).sum()}")

# Save processed data
pd.concat([X_train_scaled, y_train], axis=1).to_csv(f'{PROC}/train_processed.csv', index=False)
pd.concat([X_test_scaled,  y_test],  axis=1).to_csv(f'{PROC}/test_processed.csv',  index=False)
joblib.dump(scaler, f'{MODELS}/scaler.pkl')
print("  Saved processed splits and scaler")

# ── SMOTE visualisation ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, counts, title in [
    (axes[0], y_train.value_counts(),    'Before SMOTE (Train Set)'),
    (axes[1], pd.Series(y_train_sm).value_counts().sort_index(), 'After SMOTE (Train Set)')
]:
    bars = ax.bar(['No Churn (0)', 'Churn (1)'], [counts[0], counts[1]],
                  color=['#2ecc71', '#e74c3c'], edgecolor='white', linewidth=1.5, width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 30,
                f'{int(b.get_height()):,}', ha='center', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Count'); ax.set_ylim(0, max(counts[0], counts[1]) * 1.25)
plt.suptitle('Class Balance: Before vs After SMOTE', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{VIZ}/smote_balance.png')
plt.close()
print("  Saved: smote_balance.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] MODEL TRAINING")

models = {
    'Logistic Regression': LogisticRegression(
        C=1.0, max_iter=1000, random_state=42, n_jobs=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    ),
}

trained = {}
train_times = {}
cv_scores = {}

for name, model in models.items():
    print(f"  Training {name}...", end=' ', flush=True)
    t0 = time.time()
    model.fit(X_train_sm, y_train_sm)
    elapsed = time.time() - t0
    train_times[name] = elapsed

    # 5-fold CV on SMOTE data
    cv = cross_val_score(model, X_train_sm, y_train_sm,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring='roc_auc', n_jobs=-1)
    cv_scores[name] = cv
    trained[name] = model
    print(f"done in {elapsed:.1f}s  |  CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] MODEL EVALUATION")

results = {}
for name, model in trained.items():
    y_pred      = model.predict(X_test_scaled)
    y_proba     = model.predict_proba(X_test_scaled)[:, 1]
    results[name] = {
        'model':    model,
        'y_pred':   y_pred,
        'y_proba':  y_proba,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_test, y_proba),
        'cv_mean':   cv_scores[name].mean(),
        'cv_std':    cv_scores[name].std(),
        'train_time': train_times[name],
    }

# Print table
header = f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'CV AUC':>8}"
print(f"\n  {header}")
print("  " + "-" * 66)
for name, r in results.items():
    print(f"  {name:<22} {r['accuracy']:.4f} {r['precision']:.4f} "
          f"{r['recall']:.4f} {r['f1']:.4f} {r['roc_auc']:.4f} "
          f"{r['cv_mean']:.4f}±{r['cv_std']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CHARTS — Metrics Comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] GENERATING CHARTS")

model_names   = list(results.keys())
short_names   = ['LR', 'RF', 'XGB', 'LGBM']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
metrics_data  = {
    'Accuracy':  [results[n]['accuracy']  for n in model_names],
    'Precision': [results[n]['precision'] for n in model_names],
    'Recall':    [results[n]['recall']    for n in model_names],
    'F1 Score':  [results[n]['f1']        for n in model_names],
    'ROC AUC':   [results[n]['roc_auc']   for n in model_names],
}
palette = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

# ── 4a. Grouped bar — all metrics ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
x      = np.arange(len(metric_labels))
width  = 0.18
for i, (name, color) in enumerate(zip(model_names, palette)):
    vals   = [metrics_data[m][i] for m in metric_labels]
    bars   = ax.bar(x + i * width, vals, width, label=name, color=color,
                    edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_labels, fontsize=13)
ax.set_ylim(0, 1.08)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Model Performance Comparison — All Metrics', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.axhline(0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig(f'{VIZ}/model_comparison_metrics.png')
plt.close()
print("  Saved: model_comparison_metrics.png")

# ── 4b. ROC Curves ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
for name, color in zip(model_names, palette):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
    auc_val      = results[name]['roc_auc']
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f'{name}  (AUC = {auc_val:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — All Models', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(f'{VIZ}/roc_curves.png')
plt.close()
print("  Saved: roc_curves.png")

# ── 4c. Confusion Matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, name in zip(axes.flatten(), model_names):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name}\n'
                 f'Acc={results[name]["accuracy"]:.3f}  '
                 f'F1={results[name]["f1"]:.3f}  '
                 f'AUC={results[name]["roc_auc"]:.3f}',
                 fontsize=12, fontweight='bold')
    for text in ax.texts:
        text.set_fontsize(16)
plt.suptitle('Confusion Matrices — All Models', fontsize=17, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{VIZ}/confusion_matrices.png')
plt.close()
print("  Saved: confusion_matrices.png")

# ── 4d. CV Score Distributions ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot([cv_scores[n] for n in model_names], labels=model_names,
                patch_artist=True, notch=False, widths=0.5)
for patch, color in zip(bp['boxes'], palette):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('black'); median.set_linewidth(2)
for i, name in enumerate(model_names):
    ax.scatter([i+1]*5, cv_scores[name], color=palette[i], zorder=5, s=50, edgecolors='black')
ax.set_title('5-Fold CV ROC AUC Distribution', fontsize=16, fontweight='bold')
ax.set_ylabel('ROC AUC Score', fontsize=13)
ax.set_ylim(0.7, 1.0)
ax.axhline(np.mean([cv_scores[n].mean() for n in model_names]), color='red',
           linestyle='--', alpha=0.5, label='Overall mean')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{VIZ}/cv_scores_boxplot.png')
plt.close()
print("  Saved: cv_scores_boxplot.png")

# ── 4e. Radar chart ──────────────────────────────────────────────────────────
categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
for name, color in zip(model_names, palette):
    vals = [results[name]['accuracy'], results[name]['precision'],
            results[name]['recall'],   results[name]['f1'],
            results[name]['roc_auc']]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2.5, color=color, label=name)
    ax.fill(angles, vals, alpha=0.08, color=color)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=9)
ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=12)
plt.tight_layout()
plt.savefig(f'{VIZ}/radar_chart.png')
plt.close()
print("  Saved: radar_chart.png")

# ── 4f. Training time ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
times = [train_times[n] for n in model_names]
bars = ax.bar(model_names, times, color=palette, edgecolor='white', linewidth=1.5, width=0.5)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{t:.2f}s', ha='center', fontsize=13, fontweight='bold')
ax.set_title('Model Training Time Comparison', fontsize=16, fontweight='bold')
ax.set_ylabel('Time (seconds)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{VIZ}/training_time.png')
plt.close()
print("  Saved: training_time.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SELECT BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] SELECTING BEST MODEL")

# Score = weighted combo of F1 (0.4) + AUC (0.4) + Recall (0.2)
# We weight recall highly to minimise missed churners
scores = {
    n: 0.4 * results[n]['f1'] + 0.4 * results[n]['roc_auc'] + 0.2 * results[n]['recall']
    for n in model_names
}
best_name  = max(scores, key=scores.get)
best_model = trained[best_name]
best_r     = results[best_name]

print(f"\n  Composite scores:")
for n, s in sorted(scores.items(), key=lambda x: -x[1]):
    marker = " ← BEST" if n == best_name else ""
    print(f"    {n:<22} {s:.4f}{marker}")

print(f"\n  Best Model: {best_name}")
print(f"    Accuracy : {best_r['accuracy']:.4f}")
print(f"    Precision: {best_r['precision']:.4f}")
print(f"    Recall   : {best_r['recall']:.4f}")
print(f"    F1 Score : {best_r['f1']:.4f}")
print(f"    ROC AUC  : {best_r['roc_auc']:.4f}")
print(f"    CV AUC   : {best_r['cv_mean']:.4f} ± {best_r['cv_std']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] SAVING BEST MODEL")

model_filename = best_name.lower().replace(' ', '_') + '_best_model.pkl'
joblib.dump(best_model, f'{MODELS}/{model_filename}')
joblib.dump(feature_names, f'{MODELS}/feature_names.pkl')

model_meta = {
    'best_model': best_name,
    'model_file': model_filename,
    'metrics': {
        'accuracy':  round(best_r['accuracy'], 4),
        'precision': round(best_r['precision'], 4),
        'recall':    round(best_r['recall'], 4),
        'f1':        round(best_r['f1'], 4),
        'roc_auc':   round(best_r['roc_auc'], 4),
        'cv_auc_mean': round(best_r['cv_mean'], 4),
        'cv_auc_std':  round(best_r['cv_std'], 4),
    },
    'all_models': {
        n: {
            'accuracy':  round(results[n]['accuracy'], 4),
            'precision': round(results[n]['precision'], 4),
            'recall':    round(results[n]['recall'], 4),
            'f1':        round(results[n]['f1'], 4),
            'roc_auc':   round(results[n]['roc_auc'], 4),
        }
        for n in model_names
    },
    'n_features': len(feature_names),
    'test_set_size': len(y_test),
    'training_date': '2026-03-15',
}
with open(f'{MODELS}/model_metadata.json', 'w') as f:
    json.dump(model_meta, f, indent=2)

print(f"  Saved: models/{model_filename}")
print(f"  Saved: models/scaler.pkl")
print(f"  Saved: models/feature_names.pkl")
print(f"  Saved: models/model_metadata.json")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] SHAP ANALYSIS")

# Use a sample for SHAP speed
X_shap_bg   = X_train_scaled.sample(min(300, len(X_train_scaled)), random_state=42)
X_shap_test = X_test_scaled.sample(min(200, len(X_test_scaled)), random_state=42)

try:
    if best_name in ('XGBoost', 'LightGBM', 'Random Forest'):
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_shap_test)
        # For multi-output tree models shap_values may be a list
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
    else:
        explainer   = shap.LinearExplainer(best_model, X_shap_bg)
        shap_values = explainer.shap_values(X_shap_test)
        sv = shap_values

    sv = np.array(sv)
    print(f"  SHAP values computed — shape: {sv.shape}")
    # Handle 3D array (samples, features, classes) — take churn class (index 1)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
        print(f"  3D SHAP detected, sliced to churn class — shape: {sv.shape}")

    # ── 7a. SHAP Summary (beeswarm) ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(sv, X_shap_test, feature_names=feature_names,
                      show=False, plot_size=None, max_display=20)
    plt.title(f'SHAP Summary Plot — {best_name}', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{VIZ}/shap_summary_beeswarm.png')
    plt.close()
    print("  Saved: shap_summary_beeswarm.png")

    # ── 7b. SHAP Bar (mean |SHAP|) ───────────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    idx      = np.argsort(mean_abs)[-20:]
    top_features = [feature_names[i] for i in idx]
    top_vals     = mean_abs[idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors_shap = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_features)))
    bars = ax.barh(top_features, top_vals, color=colors_shap, edgecolor='white', linewidth=1)
    for bar, val in zip(bars, top_vals):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=13)
    ax.set_title(f'Top 20 Features — {best_name}\n(Mean Absolute SHAP Contribution)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{VIZ}/shap_feature_importance.png')
    plt.close()
    print("  Saved: shap_feature_importance.png")

    # ── 7c. SHAP Dependence plots for top 3 features ──────────────────────────
    top3_idx = np.argsort(mean_abs)[-3:][::-1]
    top3     = [feature_names[i] for i in top3_idx]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, feat_idx, feat_name in zip(axes, top3_idx, top3):
        ax.scatter(X_shap_test.iloc[:, feat_idx], sv[:, feat_idx],
                   alpha=0.4, s=20, c=sv[:, feat_idx], cmap='RdYlGn_r')
        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel(feat_name, fontsize=12)
        ax.set_ylabel('SHAP Value', fontsize=12)
        ax.set_title(f'SHAP Dependence: {feat_name}', fontsize=13, fontweight='bold')
    plt.suptitle(f'SHAP Dependence Plots — Top 3 Features ({best_name})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{VIZ}/shap_dependence_top3.png')
    plt.close()
    print("  Saved: shap_dependence_top3.png")

    # ── 7d. SHAP Waterfall for one prediction ────────────────────────────────
    # Pick a churner example
    churn_idx_list = np.where(y_test.values == 1)[0]
    if len(churn_idx_list):
        sample_idx = churn_idx_list[0]
        sample_sv  = sv[sample_idx] if sample_idx < len(sv) else sv[0]
        exp_val    = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            exp_val = exp_val[1] if len(exp_val) > 1 else exp_val[0]

        top_n = 15
        order  = np.argsort(np.abs(sample_sv))[-top_n:][::-1]
        feats  = [feature_names[i] for i in order]
        vals   = sample_sv[order]
        colors_wf = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals]

        fig, ax = plt.subplots(figsize=(12, 8))
        cumulative = exp_val
        y_pos = np.arange(top_n)
        ax.barh(y_pos, vals[::-1], color=colors_wf[::-1], edgecolor='white', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats[::-1], fontsize=10)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('SHAP Value (impact on model output)', fontsize=12)
        ax.set_title(f'SHAP Waterfall — Single Churner Prediction\n{best_name} | Base value: {exp_val:.3f}',
                     fontsize=14, fontweight='bold')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#e74c3c', label='Increases churn probability'),
                           Patch(facecolor='#2ecc71', label='Decreases churn probability')]
        ax.legend(handles=legend_elements, fontsize=11, loc='lower right')
        plt.tight_layout()
        plt.savefig(f'{VIZ}/shap_waterfall_churner.png')
        plt.close()
        print("  Saved: shap_waterfall_churner.png")

    # Save top-20 SHAP feature importances
    shap_df = pd.DataFrame({
        'feature': [feature_names[i] for i in np.argsort(mean_abs)[::-1]],
        'mean_abs_shap': np.sort(mean_abs)[::-1]
    })
    shap_df.to_csv(f'{ANAL}/shap_feature_importance.csv', index=False)
    print("  Saved: analysis/shap_feature_importance.csv")

except Exception as e:
    print(f"  SHAP warning: {e} — skipping SHAP plots")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
report_lines = [
    "=" * 70,
    "  CHURN PREDICTION ML PIPELINE — FINAL REPORT",
    "=" * 70,
    "",
    f"Dataset       : 7,043 customers | {len(feature_names)} features (after engineering)",
    f"Train / Test  : {len(y_train)} / {len(y_test)}",
    f"SMOTE balance : {(y_train_sm==1).sum()} positive examples (after)",
    "",
    "MODEL RESULTS (Test Set)",
    "-" * 66,
    f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}",
    "-" * 66,
]
for n in model_names:
    r = results[n]
    marker = " ★" if n == best_name else ""
    report_lines.append(
        f"{n:<22} {r['accuracy']:.4f} {r['precision']:.4f} "
        f"{r['recall']:.4f} {r['f1']:.4f} {r['roc_auc']:.4f}{marker}"
    )
report_lines += [
    "-" * 66,
    "",
    f"BEST MODEL    : {best_name}  (composite score = {scores[best_name]:.4f})",
    f"  Accuracy    : {best_r['accuracy']:.4f}",
    f"  Precision   : {best_r['precision']:.4f}",
    f"  Recall      : {best_r['recall']:.4f}",
    f"  F1 Score    : {best_r['f1']:.4f}",
    f"  ROC AUC     : {best_r['roc_auc']:.4f}",
    f"  CV AUC      : {best_r['cv_mean']:.4f} ± {best_r['cv_std']:.4f}",
    "",
    "SAVED ARTIFACTS",
    f"  models/{model_filename}",
    "  models/scaler.pkl",
    "  models/feature_names.pkl",
    "  models/model_metadata.json",
    "  data/processed/train_processed.csv",
    "  data/processed/test_processed.csv",
    "",
    "VISUALIZATIONS",
    "  smote_balance.png           | class balance before/after SMOTE",
    "  model_comparison_metrics.png| grouped bar chart all metrics",
    "  roc_curves.png              | ROC curves all 4 models",
    "  confusion_matrices.png      | 2×2 grid confusion matrices",
    "  cv_scores_boxplot.png       | 5-fold CV AUC distributions",
    "  radar_chart.png             | polar performance radar",
    "  training_time.png           | training time comparison",
    "  shap_summary_beeswarm.png   | SHAP beeswarm top 20 features",
    "  shap_feature_importance.png | mean |SHAP| bar chart",
    "  shap_dependence_top3.png    | SHAP dependence for top 3 features",
    "  shap_waterfall_churner.png  | waterfall for a single churner",
    "=" * 70,
]
report = "\n".join(report_lines)
print("\n" + report)

with open(f'{ANAL}/ml_pipeline_report.txt', 'w') as f:
    f.write(report)
print(f"\n  Full report saved: analysis/ml_pipeline_report.txt")
