# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
#
# # 1. 读取数据
# file_path = "heat capacity 207.xlsx"
# df = pd.read_excel(file_path, sheet_name="Sheet1")
# df = df.dropna(subset=[df.columns[0]]).reset_index(drop=True)  # 保证行索引连续，与你的 X_poly_all 对齐
# df[df.columns[0]] = df[df.columns[0]].astype(int)
#
# # 2. 列定义
# group_cols = df.columns[11:30]   # 你代码中的“基团列”切片
# temp_cols  = df.columns[30:40]   # 10个温度点
# cp_cols    = df.columns[40:50]   # 10个 Cp 列（但你训练 Cp1/Cp2 用的是下面两个绝对列索引）
# target_column_T1 = 'ASPEN Half Critical T'
# Tc0 = 138.0
#
# # === 新增：第57/59列的位置（0-based 索引）===
# CP0_COL_0BASED = 56  # BE列
# CP3_COL_0BASED = 58  # BG列
#
# # 3. 子模型训练：用于估算 T1, Cp0, Cp1, Cp2, Cp3 → 计算三段斜率
# X_groups = df[group_cols]
# valid_mask = ~df[target_column_T1].isna()
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X_groups[valid_mask])
# y_exp_T1 = np.exp(df.loc[valid_mask, target_column_T1] / Tc0)
#
# T1_model  = HuberRegressor(max_iter=9000).fit(X_poly, y_exp_T1)
#
# # 保持你原本的两列目标：Cp1 ← df.iloc[:, 9]；Cp2 ← df.iloc[:, 50]
# Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 9])
# Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 50])
#
# # === 新增：Cp0/Cp3 模型（第57/59列）===
# Cp0_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, CP0_COL_0BASED])
# Cp3_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, CP3_COL_0BASED])
#
# # 4. 构建训练数据（把 slope*T 换成“三段门控”特征）
# X_total, y_total, material_ids, temperatures = [], [], [], []
# X_poly_all = poly.transform(X_groups)
# eps = 1e-8  # 防除零
#
# for i, row in df.iterrows():
#     material_id = row.iloc[0]
#     Nk    = row[group_cols].values.astype(float)
#     temps = row[temp_cols].values.astype(float)
#     cps   = row[cp_cols].values.astype(float)
#
#     Nk_df   = pd.DataFrame([Nk], columns=group_cols)
#     Nk_poly = X_poly_all[i:i+1]
#
#     try:
#         # — 4.1 预测 T1（由 T1_model ），并构造 T0/T2/T3
#         T1_exp = T1_model.predict(Nk_poly)[0]
#         if (T1_exp <= 0) or (not np.isfinite(T1_exp)):
#             continue
#         T1 = Tc0 * np.log(T1_exp)
#         T2 = T1 * 1.5
#         T0 = T1 - 50.0         # 你的设定
#         T3 = T2 + 30.0         # 你的设定
#
#         # — 4.2 预测四个参考点的 Cp
#         C0 = float(Cp0_model.predict(Nk_df)[0])  # 第57列（BE）
#         C1 = float(Cp1_model.predict(Nk_df)[0])  # 你原来用的第 10 列（0-based 9）
#         C2 = float(Cp2_model.predict(Nk_df)[0])  # 你原来用的第 51 列（0-based 50）
#         C3 = float(Cp3_model.predict(Nk_df)[0])  # 第59列（BG）
#
#         # — 4.3 三段斜率
#         s01 = (C1 - C0) / max(T1 - T0, eps)
#         s12 = (C2 - C1) / max(T2 - T1, eps)
#         s23 = (C3 - C2) / max(T3 - T2, eps)
#
#     except Exception:
#         continue
#
#     # — 4.4 遍历该物质的温度点，构造三段“门控”特征（只激活所在分段）
#     for T, Cp in zip(temps, cps):
#         if np.isnan(T) or np.isnan(Cp):
#             continue
#
#         # 推荐：斜率 × 到折点的“距离”，更贴物理；边界归中段
#         if T < T1:
#             zL, zM, zR = s01 * (T - T1), 0.0, 0.0
#         elif T >= T2:
#             zL, zM, zR = 0.0, s12 * (T - T1), 0.0
#         else:
#             zL, zM, zR = 0.0, 0.0, s23 * (T - T2)
#
#         features = np.concatenate([
#             Nk,           # 基团
#             [T],          # 当前温度
#             [zL, zM, zR]  # 三段门控特征（替代原来的 slope*T）
#         ])
#
#         X_total.append(features)
#         y_total.append(Cp)
#         material_ids.append(material_id)
#         temperatures.append(T)
#
# X_total = np.array(X_total, dtype=float)
# y_total = np.array(y_total, dtype=float)
#
# # 5. 拟合机器学习模型（随机森林）
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_total, y_total)
#
# # 6. 评估模型
# y_pred = model.predict(X_total)
# mse = mean_squared_error(y_total, y_pred)
# r2  = r2_score(y_total, y_pred)
# ard = np.mean(np.abs((y_total - y_pred) / (np.abs(y_total) + eps))) * 100  # 加 eps 更稳
#
# # === 误差范围统计 ===
# relative_error = np.abs((y_total - y_pred) / (np.abs(y_total) + eps)) * 100
# within_1pct  = np.sum(relative_error <= 1)
# within_5pct  = np.sum(relative_error <= 5)
# within_10pct = np.sum(relative_error <= 10)
#
# print("\n📊 模型评估（分段门控特征）：")
# print(f"R²  = {r2:.4f}")
# print(f"MSE = {mse:.2f}")
# print(f"ARD = {ard:.2f}%")
# print(f"✅ 误差 ≤ 1% 的点数:  {within_1pct}")
# print(f"✅ 误差 ≤ 5% 的点数:  {within_5pct}")
# print(f"✅ 误差 ≤ 10% 的点数: {within_10pct}")
#
# # 7. 保存预测结果
# results = pd.DataFrame({
#     "Material_ID": material_ids,
#     "Temperature (K)": temperatures,
#     "Cp_measured": y_total,
#     "Cp_predicted": y_pred
# })
# results.to_excel("Cp预测结果_分段门控_RF模型.xlsx", index=False)
# print("✅ 已保存预测结果为: Cp预测结果_分段门控_RF模型.xlsx")


import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. 读取数据
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.dropna(subset=[df.columns[0]]).reset_index(drop=True)  # 保证行索引连续
df[df.columns[0]] = df[df.columns[0]].astype(int)

# 2. 列定义
group_cols = df.columns[11:30]   # 基团列
temp_cols  = df.columns[30:40]   # 10个温度点
cp_cols    = df.columns[40:50]   # 10个 Cp 列（用于 y_total）
target_column_T1 = 'ASPEN Half Critical T'
Tc0 = 138.0

# 第57/59列（Excel 1-based）→ 0-based 索引
CP0_COL_0BASED = 56  # BE
CP3_COL_0BASED = 58  # BG

# 3. 子模型训练：T1, Cp0, Cp1, Cp2, Cp3
X_groups = df[group_cols]
valid_mask = ~df[target_column_T1].isna()
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_groups[valid_mask])
y_exp_T1 = np.exp(df.loc[valid_mask, target_column_T1] / Tc0)
T1_model = HuberRegressor(max_iter=9000).fit(X_poly, y_exp_T1)

# —— 通用：带数值化与掩码过滤的拟合函数（避免 y 中 NaN）——
def fit_huber_on_col(col_idx: int) -> HuberRegressor:
    y_raw = df.iloc[:, col_idx]
    y_num = pd.to_numeric(y_raw, errors="coerce")                # 文本/#N/A/空白 → NaN
    mask = y_num.notna() & X_groups.notna().all(axis=1)          # 过滤 y 为 NaN 或特征缺失的行
    if mask.sum() == 0:
        raise ValueError(f"列 {col_idx} 过滤后无有效样本，无法训练。")
    return HuberRegressor(max_iter=9000).fit(X_groups.loc[mask], y_num.loc[mask])

# 你原来用的两列：第10列(0-based 9)、第51列(0-based 50)
Cp1_model = fit_huber_on_col(9)
Cp2_model = fit_huber_on_col(50)

# 新增两列：第57/59列（0-based 56/58）
Cp0_model = fit_huber_on_col(CP0_COL_0BASED)
Cp3_model = fit_huber_on_col(CP3_COL_0BASED)

# 4. 构建训练数据（把 slope*T 换成“三段门控”特征）
X_total, y_total, material_ids, temperatures = [], [], [], []
X_poly_all = poly.transform(X_groups)
eps = 1e-8  # 数值保护

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk    = row[group_cols].values.astype(float)
    temps = row[temp_cols].values.astype(float)
    cps   = row[cp_cols].values.astype(float)

    Nk_df   = pd.DataFrame([Nk], columns=group_cols)
    Nk_poly = X_poly_all[i:i+1]

    try:
        # — 4.1 预测 T1，并构造 T0/T2/T3
        T1_exp = T1_model.predict(Nk_poly)[0]
        if (T1_exp <= 0) or (not np.isfinite(T1_exp)):
            continue
        T1 = Tc0 * np.log(T1_exp)
        T2 = 1.5 * T1
        T0 = T1 - 50.0
        T3 = T2 + 30.0

        # — 4.2 预测四个参考点的 Cp
        C0 = float(Cp0_model.predict(Nk_df)[0])  # 第57列（BE）
        C1 = float(Cp1_model.predict(Nk_df)[0])  # 第10列（0-based 9）
        C2 = float(Cp2_model.predict(Nk_df)[0])  # 第51列（0-based 50）
        C3 = float(Cp3_model.predict(Nk_df)[0])  # 第59列（BG）

        # — 4.3 三段斜率
        s01 = (C1 - C0) / max(T1 - T0, eps)
        s12 = (C2 - C1) / max(T2 - T1, eps)
        s23 = (C3 - C2) / max(T3 - T2, eps)

    except Exception:
        continue

    # — 4.4 三段“门控”特征（只激活所在分段）
    for T, Cp in zip(temps, cps):
        if not (np.isfinite(T) and np.isfinite(Cp)):
            continue

        # 正确的分段：左段 T<T1；中段 T1≤T≤T2；右段 T>T2
        if T < T1:
            zL, zM, zR = s01 * (T - T1), 0.0, 0.0
        elif T <= T2:
            zL, zM, zR = 0.0, s12 * (T - T1), 0.0
        else:
            zL, zM, zR = 0.0, 0.0, s23 * (T - T2)

        features = np.concatenate([Nk, [T, zL, zM, zR]])
        X_total.append(features)
        y_total.append(Cp)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total, dtype=float)
y_total = np.array(y_total, dtype=float)

# 5. 拟合机器学习模型（随机森林）
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_total, y_total)

# 6. 评估模型（训练集上）
y_pred = model.predict(X_total)
mse = mean_squared_error(y_total, y_pred)
r2  = r2_score(y_total, y_pred)
rel_err = np.abs((y_total - y_pred) / (np.abs(y_total) + eps)) * 100
ard = rel_err.mean()

within_1pct  = (rel_err <= 1).sum()
within_5pct  = (rel_err <= 5).sum()
within_10pct = (rel_err <= 10).sum()

print("\n📊 模型评估（分段门控特征）：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")
print(f"✅ 误差 ≤ 1% 的点数:  {within_1pct}")
print(f"✅ 误差 ≤ 5% 的点数:  {within_5pct}")
print(f"✅ 误差 ≤ 10% 的点数: {within_10pct}")

# 7. 保存预测结果
results = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Cp_measured": y_total,
    "Cp_predicted": y_pred
})
results.to_excel("Cp预测结果_分段门控_RF模型.xlsx", index=False)
print("✅ 已保存预测结果为: Cp预测结果_分段门控_RF模型.xlsx")
