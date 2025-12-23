# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ========== 读取数据 ==========
# file_path = "vp209.xlsx"
# df = pd.read_excel(file_path, sheet_name='Sheet1')
#
# # 特征提取
# Nk = df.iloc[:, 12:31].values  # 19个基团
# T = df.iloc[:, 31:41].values
# P_vp = df.iloc[:, 41:51].values
#
# # 清理非法行（所有10个蒸汽压均有效）
# valid_mask = np.isfinite(P_vp) & (P_vp > 0)
# valid_mask = valid_mask.all(axis=1)
#
# Nk = Nk[valid_mask]
# T = T[valid_mask]
# P_vp = P_vp[valid_mask]
#
# # ========== 构造训练数据 ==========
# X = np.hstack([
#     Nk.repeat(10, axis=0),            # 12 个基团
#     T.flatten().reshape(-1, 1)        # 温度
# ])
# y = np.log(P_vp).flatten()           # 目标值为 ln(P)
#
# # 清理 NaN
# finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
# X = X[finite_mask]
# y = y[finite_mask]
#
# # ========== 训练模型 ==========
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X, y)
#
# # ========== 模型评估 ==========
# y_pred = model.predict(X)
# mse_lnP = mean_squared_error(y, y_pred)
# r2_lnP = r2_score(y, y_pred)
#
# print("\n📈 机器学习模型对 ln(P) 拟合结果：")
# print(f"R² (lnP) = {r2_lnP:.6f}")
# print(f"MSE (lnP) = {mse_lnP:.6f}")
#
# # 还原对数后的 P 值
# P_true = np.exp(y)
# P_pred = np.exp(y_pred)
#
# mse_P = mean_squared_error(P_true, P_pred)
# r2_P = r2_score(P_true, P_pred)
# ard_P = np.mean(np.abs((P_pred - P_true) / P_true)) * 100
#
# print("\n📈 实际蒸汽压 P 评估：")
# print(f"R² (P)  = {r2_P:.6f}")
# print(f"MSE (P) = {mse_P:.4f}")
# print(f"ARD (P) = {ard_P:.2f}%")
#
# # ========== 保存结果 ==========
# compare_df = pd.DataFrame({
#     "Temperature_K": T.flatten()[finite_mask],
#     "ln(P)_true": y,
#     "ln(P)_pred": y_pred,
#     "Absolute_Error_lnP": np.abs(y - y_pred),
#     "Relative_Error_lnP (%)": 100 * np.abs((y - y_pred) / y),
#     "P_true": P_true,
#     "P_pred": P_pred,
#     "Absolute_Error_P": np.abs(P_true - P_pred),
#     "Relative_Error_P (%)": 100 * np.abs((P_true - P_pred) / P_true)
# })
# compare_df.to_excel("VaporPrediction_ML_lnP_RF.xlsx", index=False)
# print("✅ 已保存预测结果为 VaporPrediction_ML_lnP_RF.xlsx")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ========== 读取数据 ===========
file_path = "vp209.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 特征提取
Nk = df.iloc[:, 12:31].values  # 19个基团
T = df.iloc[:, 31:41].values
P_vp = df.iloc[:, 41:51].values

# 清理非法行（所有10个蒸汽压均有效）
valid_mask = np.isfinite(P_vp) & (P_vp > 0)
valid_mask = valid_mask.all(axis=1)

Nk = Nk[valid_mask]
T = T[valid_mask]
P_vp = P_vp[valid_mask]

# ========== 构造训练数据 ==========
X = np.hstack([
    Nk.repeat(10, axis=0),            # 12 个基团
    T.flatten().reshape(-1, 1)        # 温度
])
y = np.log(P_vp).flatten()           # 目标值为 ln(P)

# 清理 NaN
finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
X = X[finite_mask]
y = y[finite_mask]

# ========== 训练模型 ==========
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ========== 模型评估 ==========
y_pred = model.predict(X)
mse_lnP = mean_squared_error(y, y_pred)
r2_lnP = r2_score(y, y_pred)

print("\n📈 机器学习模型对 ln(P) 拟合结果：")
print(f"R² (lnP) = {r2_lnP:.6f}")
print(f"MSE (lnP) = {mse_lnP:.6f}")

# 还原对数后的 P 值
P_true = np.exp(y)
P_pred = np.exp(y_pred)

mse_P = mean_squared_error(P_true, P_pred)
r2_P = r2_score(P_true, P_pred)
ard_P = np.mean(np.abs((P_pred - P_true) / P_true)) * 100

print("\n📈 实际蒸汽压 P 评估：")
print(f"R² (P)  = {r2_P:.6f}")
print(f"MSE (P) = {mse_P:.4f}")
print(f"ARD (P) = {ard_P:.2f}%")

# ========== 误差统计 ==========
relative_error = np.abs((P_pred - P_true) / P_true) * 100

# 统计误差小于 1%、5%、10% 的点数
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print(f"\n✅ 误差 ≤ 1% 的点数: {within_1pct}")
print(f"✅ 误差 ≤ 5% 的点数: {within_5pct}")
print(f"✅ 误差 ≤ 10% 的点数: {within_10pct}")

# ========== 保存结果 ==========
compare_df = pd.DataFrame({
    "Temperature_K": T.flatten()[finite_mask],
    "ln(P)_true": y,
    "ln(P)_pred": y_pred,
    "Absolute_Error_lnP": np.abs(y - y_pred),
    "Relative_Error_lnP (%)": 100 * np.abs((y - y_pred) / y),
    "P_true": P_true,
    "P_pred": P_pred,
    "Absolute_Error_P": np.abs(P_true - P_pred),
    "Relative_Error_P (%)": 100 * np.abs((P_true - P_pred) / P_true)
})
compare_df.to_excel("VaporPrediction_ML_lnP_RF.xlsx", index=False)
print("✅ 已保存预测结果为 VaporPrediction_ML_lnP_RF.xlsx")
