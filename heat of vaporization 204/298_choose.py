import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# === 1. 读取数据 ===
df = pd.read_csv("describe word_cleaned_204_normal.csv")  # 或 .xlsx 替换成 .read_excel

# === 2. 提取特征和目标 ===
X = df.iloc[:, :-1]  # 所有列除了最后一列（假设为描述符）
y = df.iloc[:, -1]   # 最后一列为目标变量（如 ln(Vapor_Pressure)）

# === 3. 构建线性回归模型 ===
lr = LinearRegression()

# === 4. 前向特征选择：选择前 25 个特征 ===
sfs = SFS(lr,
          k_features=25,
          forward=True,
          floating=False,
          scoring='r2',
          cv=5,
          n_jobs=-1)

sfs = sfs.fit(X.values, y.values)

# === 5. 获取特征名并输出 ===
selected_features = list(X.columns[list(sfs.k_feature_idx_)])
print("✅ Top 25 descriptors selected:")
for feat in selected_features:
    print(" -", feat)

# === 6. 保存选中特征 + 目标列 ===
df_selected = df[selected_features + [df.columns[-1]]]
df_selected.to_excel("selected_25_descriptors_data_298.xlsx", index=False)
print("📁 已保存文件: selected_25_descriptors_data_298.xlsx")
