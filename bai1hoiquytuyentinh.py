import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# 1. Nạp dữ liệu Diabetes
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# 3 features đã chọn
selected_features = ['bmi', 'bp', 's5']

# 2. Tính hệ số tương quan với target
print("--- HỆ SỐ TƯƠNG QUAN VỚI TARGET ---")
corr = df[selected_features + ['target']].corr() # tính hệ số tương quan
print(corr['target'].drop('target'))

print("\n" + "=" * 40 + "\n")

# 3. Thống kê Outlier theo IQR
print("--- THỐNG KÊ OUTLIER (IQR METHOD) ---")
for col in selected_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    print(f"Cột {col}: Có {len(outliers)} điểm outlier ({len(outliers) / len(df) * 100:.2f}%)")

# 4. Vẽ biểu đồ 3x3 (Phân phối, Outlier, Scatter plot)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for i, col in enumerate(selected_features):
    # Biểu đồ Phân phối
    sns.histplot(df[col], kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'Phân phối của {col}')

    # Biểu đồ Boxplot (Outlier)
    sns.boxplot(y=df[col], ax=axes[i, 1], color='lightgreen')
    axes[i, 1].set_title(f'Boxplot (Outlier) của {col}')

    # Biểu đồ Scatter Plot với Target
    sns.scatterplot(x=df[col], y=df['target'], ax=axes[i, 2], alpha=0.6, color='coral')
    sns.regplot(x=df[col], y=df['target'], ax=axes[i, 2], scatter=False, color='red')
    axes[i, 2].set_title(f'Scatter Plot: {col} vs Target')

plt.tight_layout()
plt.show()