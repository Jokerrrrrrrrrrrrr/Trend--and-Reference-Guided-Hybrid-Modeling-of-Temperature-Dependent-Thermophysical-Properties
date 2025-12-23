import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# === 1. 读取数据 ===
df = pd.read_excel('data_normal.xlsx')

# === 2. 划分特征与目标 ===
# 最后 20 列是温度（10列）+ 热容（10列），我们只取最后10列（热容）
X = df.iloc[:, :-1]  # 所有描述符
y = df.iloc[:, -1:]
# === 3. 构建线性回归模型 ===
lr = LinearRegression()

# === 4. 前向选择 25 个最优特征 ===
sfs = SFS(lr,
          k_features=25,
          forward=True,
          floating=False,
          scoring='r2',
          cv=5,
          n_jobs=-1)

sfs = sfs.fit(X, y)

# === 5. 获取选出的列名 ===
selected_features = list(sfs.k_feature_names_)
print("Top 25 descriptors selected:")
print(selected_features)

# === 6. 保存结果到 Excel（可选） ===
df_selected = df[selected_features + list(df.columns[-20:])]  # 合并描述符 + 原目标数据
df_selected.to_excel('selected_25_descriptors_normal.xlsx', index=False)
