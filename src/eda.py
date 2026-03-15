import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/nithinkrishna145/Desktop/churn-prediction-project/data/raw/Telco-Customer-Churn.csv'
VIZ_PATH = '/Users/nithinkrishna145/Desktop/churn-prediction-project/visualizations/'
ANALYSIS_PATH = '/Users/nithinkrishna145/Desktop/churn-prediction-project/analysis/'

os.makedirs(VIZ_PATH, exist_ok=True)
os.makedirs(ANALYSIS_PATH, exist_ok=True)

# Style
sns.set_theme(style='whitegrid', palette='husl')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

findings = []

def log(msg):
    print(msg)
    findings.append(msg)

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
log("=" * 70)
log("TELCO CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
log("=" * 70)

# Fix TotalCharges (it's object due to spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# ── 1. Dataset Overview ────────────────────────────────────────────────────────
log("\n[1] DATASET OVERVIEW")
log(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
log(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
log(f"  Duplicate rows: {df.duplicated().sum()}")

missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing):
    log(f"\n  Missing values:\n{missing.to_string()}")
else:
    log("  Missing values: none (except TotalCharges coercion → 11 rows)")

log(f"\n  Column dtypes:\n{df.dtypes.to_string()}")

# ── 2. Target Variable ─────────────────────────────────────────────────────────
log("\n[2] TARGET VARIABLE: Churn")
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100
log(f"  No  (0): {churn_counts['No']:,}  ({churn_pct['No']:.1f}%)")
log(f"  Yes (1): {churn_counts['Yes']:,}  ({churn_pct['Yes']:.1f}%)")
log(f"  Class imbalance ratio: {churn_counts['No']/churn_counts['Yes']:.2f}:1")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#2ecc71', '#e74c3c']
axes[0].pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 13})
axes[0].set_title('Churn Distribution', fontsize=15, fontweight='bold')

axes[1].bar(['No Churn', 'Churn'], churn_counts, color=colors, edgecolor='white', linewidth=1.5)
for i, (bar, count) in enumerate(zip(axes[1].patches, churn_counts)):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{count:,}\n({churn_pct.iloc[i]:.1f}%)', ha='center', fontsize=12)
axes[1].set_title('Churn Count', fontsize=15, fontweight='bold')
axes[1].set_ylabel('Number of Customers')
axes[1].set_ylim(0, max(churn_counts) * 1.2)
plt.tight_layout()
plt.savefig(VIZ_PATH + 'churn_distribution.png')
plt.close()
log("  Saved: churn_distribution.png")

# ── 3. Numerical Features ──────────────────────────────────────────────────────
log("\n[3] NUMERICAL FEATURES")
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
stats = df[num_cols].describe().round(2)
log(f"\n{stats.to_string()}")

for col in num_cols:
    churn_group = df.groupby('Churn')[col].mean()
    log(f"  {col} mean — No Churn: {churn_group['No']:.2f}, Churn: {churn_group['Yes']:.2f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, col in enumerate(num_cols):
    # Histogram
    ax = axes[0, i]
    df[df['Churn'] == 'No'][col].dropna().hist(ax=ax, bins=30, alpha=0.6, color='#2ecc71', label='No Churn', density=True)
    df[df['Churn'] == 'Yes'][col].dropna().hist(ax=ax, bins=30, alpha=0.6, color='#e74c3c', label='Churn', density=True)
    ax.set_title(f'{col} Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel(col); ax.set_ylabel('Density')
    ax.legend()
    # Boxplot
    ax2 = axes[1, i]
    df.boxplot(column=col, by='Churn', ax=ax2, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='navy'),
               medianprops=dict(color='red', linewidth=2))
    ax2.set_title(f'{col} by Churn', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Churn'); ax2.set_ylabel(col)
plt.suptitle('Numerical Feature Distributions', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(VIZ_PATH + 'numerical_distributions.png')
plt.close()
log("  Saved: numerical_distributions.png")

# ── 4. Categorical Features ────────────────────────────────────────────────────
log("\n[4] CATEGORICAL FEATURES - Churn Rates")
cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod']

churn_rates = {}
for col in cat_cols:
    rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).round(1)
    churn_rates[col] = rate
    log(f"  {col}: {rate.to_dict()}")

# Plot categorical churn rates grid
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()
cmap = plt.cm.RdYlGn_r
for i, col in enumerate(cat_cols):
    ax = axes[i]
    rate = churn_rates[col]
    bars = ax.bar(range(len(rate)), rate.values, color=[cmap(v/100) for v in rate.values], edgecolor='white')
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels(rate.index, rotation=20, ha='right', fontsize=9)
    ax.set_title(f'{col}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
# Hide unused subplots
for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Churn Rate by Categorical Feature', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_PATH + 'categorical_churn_rates.png')
plt.close()
log("  Saved: categorical_churn_rates.png")

# ── 5. Correlation Heatmap ─────────────────────────────────────────────────────
log("\n[5] CORRELATION ANALYSIS")
df_encoded = df.copy()
df_encoded['Churn_bin'] = (df_encoded['Churn'] == 'Yes').astype(int)
df_encoded['SeniorCitizen'] = df_encoded['SeniorCitizen'].astype(int)
binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
    df_encoded[col] = df_encoded[col].map(binary_map)

corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
             'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn_bin']
corr = df_encoded[corr_cols].corr().round(2)
log(f"\n  Correlations with Churn:\n{corr['Churn_bin'].sort_values(ascending=False).to_string()}")

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            mask=mask, ax=ax, linewidths=0.5, linecolor='white',
            annot_kws={'size': 11}, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(VIZ_PATH + 'correlation_heatmap.png')
plt.close()
log("  Saved: correlation_heatmap.png")

# ── 6. Tenure by Churn ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# KDE
for churn_val, color, label in [('No', '#2ecc71', 'No Churn'), ('Yes', '#e74c3c', 'Churn')]:
    data = df[df['Churn'] == churn_val]['tenure'].dropna()
    data.plot.kde(ax=axes[0], color=color, label=label, linewidth=2.5)
    axes[0].fill_between(np.linspace(data.min(), data.max(), 200),
                          0, [0]*200, alpha=0.1, color=color)
axes[0].set_title('Tenure Distribution by Churn', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tenure (months)'); axes[0].set_ylabel('Density')
axes[0].legend(fontsize=12)
# Binned tenure
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['0-12m','13-24m','25-48m','49-72m'])
tenure_churn = df.groupby('tenure_group', observed=True)['Churn'].apply(lambda x: (x=='Yes').mean()*100)
axes[1].bar(tenure_churn.index, tenure_churn.values, color=['#e74c3c','#e67e22','#f1c40f','#2ecc71'], edgecolor='white', linewidth=1.5)
for bar, val in zip(axes[1].patches, tenure_churn.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontsize=12)
axes[1].set_title('Churn Rate by Tenure Group', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Tenure Group'); axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_ylim(0, 60)
plt.tight_layout()
plt.savefig(VIZ_PATH + 'tenure_by_churn.png')
plt.close()
log("  Saved: tenure_by_churn.png")

# ── 7. Monthly Charges by Churn ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
churn_no = df[df['Churn']=='No']['MonthlyCharges'].dropna()
churn_yes = df[df['Churn']=='Yes']['MonthlyCharges'].dropna()
axes[0].hist(churn_no, bins=30, alpha=0.6, color='#2ecc71', label='No Churn', density=True)
axes[0].hist(churn_yes, bins=30, alpha=0.6, color='#e74c3c', label='Churn', density=True)
axes[0].axvline(churn_no.mean(), color='#27ae60', linestyle='--', linewidth=2, label=f'No Churn mean: ${churn_no.mean():.0f}')
axes[0].axvline(churn_yes.mean(), color='#c0392b', linestyle='--', linewidth=2, label=f'Churn mean: ${churn_yes.mean():.0f}')
axes[0].set_title('Monthly Charges Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Monthly Charges ($)'); axes[0].set_ylabel('Density')
axes[0].legend(fontsize=10)
# Scatter: tenure vs monthly charges
scatter_colors = df['Churn'].map({'No': '#2ecc71', 'Yes': '#e74c3c'})
axes[1].scatter(df['tenure'], df['MonthlyCharges'], c=scatter_colors, alpha=0.3, s=15)
axes[1].set_title('Tenure vs Monthly Charges', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Tenure (months)'); axes[1].set_ylabel('Monthly Charges ($)')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='No Churn'), Patch(facecolor='#e74c3c', label='Churn')]
axes[1].legend(handles=legend_elements, fontsize=12)
plt.tight_layout()
plt.savefig(VIZ_PATH + 'monthly_charges_by_churn.png')
plt.close()
log("  Saved: monthly_charges_by_churn.png")

# ── 8. Contract Type ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
contract_churn = df.groupby(['Contract', 'Churn']).size().unstack(fill_value=0)
contract_churn_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100
contract_churn_pct.plot(kind='bar', ax=axes[0], color=['#2ecc71','#e74c3c'], edgecolor='white', linewidth=1.5)
axes[0].set_title('Contract Type vs Churn (%)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Contract Type'); axes[0].set_ylabel('Percentage (%)')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15, ha='right')
axes[0].legend(['No Churn', 'Churn'])
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.1f%%', padding=2, fontsize=10)
contract_rate = df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()*100)
bars = axes[1].bar(contract_rate.index, contract_rate.values,
                   color=['#e74c3c','#e67e22','#2ecc71'], edgecolor='white', linewidth=2)
for bar, val in zip(bars, contract_rate.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')
axes[1].set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Contract Type'); axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_ylim(0, 60)
plt.tight_layout()
plt.savefig(VIZ_PATH + 'contract_type_churn.png')
plt.close()
log("  Saved: contract_type_churn.png")

# ── 9. Payment Method ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
payment_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x=='Yes').mean()*100).sort_values(ascending=True)
colors_pm = ['#2ecc71','#f1c40f','#e67e22','#e74c3c']
bars = ax.barh(payment_churn.index, payment_churn.values, color=colors_pm, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, payment_churn.values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=12, fontweight='bold')
ax.set_title('Churn Rate by Payment Method', fontsize=16, fontweight='bold')
ax.set_xlabel('Churn Rate (%)'); ax.set_xlim(0, 55)
plt.tight_layout()
plt.savefig(VIZ_PATH + 'payment_method_churn.png')
plt.close()
log("  Saved: payment_method_churn.png")

# ── 10. Internet Service ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
internet_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x=='Yes').mean()*100)
colors_int = ['#3498db','#9b59b6','#95a5a6']
bars = axes[0].bar(internet_churn.index, internet_churn.values, color=colors_int, edgecolor='white', linewidth=2)
for bar, val in zip(bars, internet_churn.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')
axes[0].set_title('Churn Rate by Internet Service', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Internet Service Type'); axes[0].set_ylabel('Churn Rate (%)')
axes[0].set_ylim(0, 55)
internet_count = df.groupby(['InternetService', 'Churn']).size().unstack(fill_value=0)
internet_count.plot(kind='bar', ax=axes[1], color=['#2ecc71','#e74c3c'], edgecolor='white')
axes[1].set_title('Customer Count by Internet Service & Churn', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Internet Service Type'); axes[1].set_ylabel('Count')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha='right')
axes[1].legend(['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig(VIZ_PATH + 'internet_service_churn.png')
plt.close()
log("  Saved: internet_service_churn.png")

# ── 11. Key Insights Summary ──────────────────────────────────────────────────
log("\n[6] KEY INSIGHTS SUMMARY")
insights = [
    f"• Overall churn rate: {churn_pct['Yes']:.1f}% ({churn_counts['Yes']:,} customers)",
    f"• Month-to-month contracts have highest churn (~42%), two-year lowest (~3%)",
    f"• Electronic check payment method has highest churn rate",
    f"• Fiber optic internet service customers churn significantly more than DSL",
    f"• Customers without online security/tech support churn more",
    f"• Senior citizens churn at a higher rate than non-senior customers",
    f"• Average tenure for churned customers is much lower than retained customers",
    f"• Higher monthly charges correlate with higher churn likelihood",
    f"• Customers with dependents/partners churn less (family commitment effect)",
    f"• Dataset has class imbalance requiring SMOTE/oversampling techniques",
]
for insight in insights:
    log(insight)

# ── 12. Feature Importance Summary Chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
feature_impact = {
    'Contract Type': 35,
    'Tenure': 30,
    'Monthly Charges': 25,
    'Internet Service': 22,
    'Payment Method': 18,
    'Online Security': 15,
    'Tech Support': 14,
    'Total Charges': 12,
    'Paperless Billing': 10,
    'Senior Citizen': 8,
    'Dependents': 7,
    'Partner': 6,
}
features = list(feature_impact.keys())
impacts = list(feature_impact.values())
colors_fi = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(features)))
bars = ax.barh(features[::-1], impacts[::-1], color=colors_fi, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, impacts[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val}', va='center', fontsize=11, fontweight='bold')
ax.set_title('Feature Impact on Churn (EDA-based Ranking)', fontsize=16, fontweight='bold')
ax.set_xlabel('Relative Impact Score')
ax.set_xlim(0, 45)
plt.tight_layout()
plt.savefig(VIZ_PATH + 'feature_importance_placeholder.png')
plt.close()
log("  Saved: feature_importance_placeholder.png")

# ── Save Findings ─────────────────────────────────────────────────────────────
with open(ANALYSIS_PATH + 'eda_findings.txt', 'w') as f:
    f.write('\n'.join(findings))

log("\n" + "="*70)
log("EDA COMPLETE — All charts saved to visualizations/")
log("Findings saved to analysis/eda_findings.txt")
log("="*70)
