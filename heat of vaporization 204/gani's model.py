# import numpy as np
# import pandas as pd
# from scipy.optimize import least_squares
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ==================== 读取数据 ====================
# df = pd.read_excel("heat of vaporization 204.xlsx", sheet_name='Sheet1')
#
# Nk = df.iloc[:, 13:32].values           # 19个基团
# MW = df.iloc[:, 4].values.reshape(-1, 1)
# Nc = df.iloc[:, 10].values.reshape(-1, 1)
# T = df.iloc[:, 32:42].values            # 温度 (10列)
# Hvap = df.iloc[:, 42:52].values         # ΔHvap (单位 J/mol)
#
# # ==================== 清洗非法值 ====================
# valid_mask = np.isfinite(Hvap) & (Hvap > 0)
# valid_mask = valid_mask.all(axis=1)
#
# Nk = Nk[valid_mask]
# MW = MW[valid_mask]
# Nc = Nc[valid_mask]
# T = T[valid_mask]
# Hvap = Hvap[valid_mask]
#
# X = np.hstack([
#     Nk.repeat(10, axis=0),
#     MW.repeat(10, axis=0),
#     Nc.repeat(10, axis=0),
#     T.flatten().reshape(-1, 1)
# ])
# y = Hvap.flatten()
#
# mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
# X = X[mask]
# y = y[mask]
#
# # ==================== 残差函数（适配 19 个基团） ====================
# def residuals(params, X, y):
#     Nk = X[:, :19]
#     MW = X[:, 19].reshape(-1, 1)
#     Nc = X[:, 20].reshape(-1, 1)
#     T = np.clip(X[:, 21].reshape(-1, 1), 1e-6, None)
#
#     B1k = params[0:19]
#     B2k = params[19:38]
#     C1k = params[38:57]
#     C2k = params[57:76]
#     D1k = params[76:95]
#     D2k = params[95:114]
#     β, γ, δ = params[114:117]
#     f0, f1 = params[117:119]
#
#     R = 8.3144
#
#     Bi = np.sum(Nk * (B1k + MW * B2k), axis=1, keepdims=True) + β * (f0 + Nc * f1)
#     Ci = np.sum(Nk * (C1k + MW * C2k), axis=1, keepdims=True) + γ * (f0 + Nc * f1)
#     Di = np.sum(Nk * (D1k + MW * D2k), axis=1, keepdims=True) + δ * (f0 + Nc * f1)
#
#     y_pred = -R * ((1.5 * Bi) / np.sqrt(T) + Ci * T + Di * T**2)
#     return (y_pred.flatten() - y)
#
# # ==================== 参数初始化（121个参数） ====================
# params_init = np.zeros(121)
#
# # ==================== 最小二乘拟合 ====================
# print("🚀 拟合中，请稍候...")
# result = least_squares(residuals, x0=params_init, args=(X, y), max_nfev=10000)
#
# # ==================== 输出参数 ====================
# param_names = (
#     [f"B1_{i}" for i in range(19)] +
#     [f"B2_{i}" for i in range(19)] +
#     [f"C1_{i}" for i in range(19)] +
#     [f"C2_{i}" for i in range(19)] +
#     [f"D1_{i}" for i in range(19)] +
#     [f"D2_{i}" for i in range(19)] +
#     ["β", "γ", "δ", "f0", "f1"]
# )
#
# print("\n🔧 参数拟合结果：")
# for name, val in zip(param_names, result.x):
#     print(f"{name:10s}: {val:.6f}")
#
# # ==================== 模型评估 ====================
# y_pred = y - residuals(result.x, X, y)
# mse = mean_squared_error(y, y_pred)
# r2 = r2_score(y, y_pred)
# ard = np.mean(np.abs((y_pred - y) / y)) * 100
#
# print(f"\n📈 模型性能：")
# print(f"R²  = {r2:.6f}")
# print(f"MSE = {mse:.4f}")
# print(f"ARD = {ard:.2f}%")
#
# # ==================== 保存结果 ====================
# compound_ids = np.repeat(df.iloc[valid_mask, 0].values, 10)[mask]
# T_used = T.flatten()[mask]
#
# df_result = pd.DataFrame({
#     "Compound_ID": compound_ids,
#     "Temperature (K)": T_used,
#     "Hvap_true (J/mol)": y,
#     "Hvap_pred (J/mol)": y_pred,
#     "Absolute Error": np.abs(y - y_pred),
#     "Relative Error (%)": 100 * np.abs((y - y_pred) / y)
# })
# df_result.to_excel("Hvap_prediction_results_19group.xlsx", index=False)
# print("\n✅ 已保存预测结果为 Hvap_prediction_results_19group.xlsx")


import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score

# ==================== 读取数据 ====================
df = pd.read_excel("heat of vaporization 204.xlsx", sheet_name='Sheet1')

Nk = df.iloc[:, 13:32].values           # 19个基团
MW = df.iloc[:, 4].values.reshape(-1, 1)
Nc = df.iloc[:, 10].values.reshape(-1, 1)
T = df.iloc[:, 32:42].values            # 温度 (10列)
Hvap = df.iloc[:, 42:52].values         # ΔHvap (单位 J/mol)

# ==================== 清洗非法值 ====================
valid_mask = np.isfinite(Hvap) & (Hvap > 0)
valid_mask = valid_mask.all(axis=1)

Nk = Nk[valid_mask]
MW = MW[valid_mask]
Nc = Nc[valid_mask]
T = T[valid_mask]
Hvap = Hvap[valid_mask]

X = np.hstack([
    Nk.repeat(10, axis=0),
    MW.repeat(10, axis=0),
    Nc.repeat(10, axis=0),
    T.flatten().reshape(-1, 1)
])
y = Hvap.flatten()

mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
X = X[mask]
y = y[mask]

# ==================== 残差函数（适配 19 个基团） ====================
def residuals(params, X, y):
    Nk = X[:, :19]
    MW = X[:, 19].reshape(-1, 1)
    Nc = X[:, 20].reshape(-1, 1)
    T = np.clip(X[:, 21].reshape(-1, 1), 1e-6, None)

    B1k = params[0:19]
    B2k = params[19:38]
    C1k = params[38:57]
    C2k = params[57:76]
    D1k = params[76:95]
    D2k = params[95:114]
    β, γ, δ = params[114:117]
    f0, f1 = params[117:119]

    R = 8.3144

    Bi = np.sum(Nk * (B1k + MW * B2k), axis=1, keepdims=True) + β * (f0 + Nc * f1)
    Ci = np.sum(Nk * (C1k + MW * C2k), axis=1, keepdims=True) + γ * (f0 + Nc * f1)
    Di = np.sum(Nk * (D1k + MW * D2k), axis=1, keepdims=True) + δ * (f0 + Nc * f1)

    y_pred = -R * ((1.5 * Bi) / np.sqrt(T) + Ci * T + Di * T**2)
    return (y_pred.flatten() - y)

# ==================== 参数初始化（121个参数） ====================
params_init = np.zeros(121)

# ==================== 最小二乘拟合 ====================
print("🚀 拟合中，请稍候...")
result = least_squares(residuals, x0=params_init, args=(X, y), max_nfev=10000)

# ==================== 输出参数 ====================
param_names = (
    [f"B1_{i}" for i in range(19)] +
    [f"B2_{i}" for i in range(19)] +
    [f"C1_{i}" for i in range(19)] +
    [f"C2_{i}" for i in range(19)] +
    [f"D1_{i}" for i in range(19)] +
    [f"D2_{i}" for i in range(19)] +
    ["β", "γ", "δ", "f0", "f1"]
)

print("\n🔧 参数拟合结果：")
for name, val in zip(param_names, result.x):
    print(f"{name:10s}: {val:.6f}")

# ==================== 模型评估 ====================
y_pred = y - residuals(result.x, X, y)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
ard = np.mean(np.abs((y_pred - y) / y)) * 100

print(f"\n📈 模型性能：")
print(f"R²  = {r2:.6f}")
print(f"MSE = {mse:.4f}")
print(f"ARD = {ard:.2f}%")

# ==================== 实际热蒸汽化焓评估 ====================
P_true = y
P_pred = y_pred
mse_P = mean_squared_error(P_true, P_pred)
r2_P = r2_score(P_true, P_pred)
ard_P = np.mean(np.abs((P_pred - P_true) / P_true)) * 100

print(f"\n📈 实际热蒸汽化焓评估: ")
print(f"R² (P)  = {r2_P:.6f}")
print(f"MSE (P) = {mse_P:.4f}")
print(f"ARD (P) = {ard_P:.2f}%")

# ==================== 误差统计（1%、5%、10%） ====================
relative_error = np.abs((P_pred - P_true) / P_true) * 100

# 统计误差小于 1%、5%、10% 的数据点数量
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

# 输出误差统计
print(f"✅ 误差 ≤ 1% 的点数: {within_1pct}")
print(f"✅ 误差 ≤ 5% 的点数: {within_5pct}")
print(f"✅ 误差 ≤ 10% 的点数: {within_10pct}")

# ==================== 保存结果 ====================
compound_ids = np.repeat(df.iloc[valid_mask, 0].values, 10)[mask]
T_used = T.flatten()[mask]

df_result = pd.DataFrame({
    "Compound_ID": compound_ids,
    "Temperature (K)": T_used,
    "Hvap_true (J/mol)": y,
    "Hvap_pred (J/mol)": y_pred,
    "Absolute Error": np.abs(y - y_pred),
    "Relative Error (%)": 100 * np.abs((y - y_pred) / y)
})
df_result.to_excel("Hvap_prediction_results_with_error_statistics.xlsx", index=False)
print("\n✅ 已保存预测结果为 Hvap_prediction_results_with_error_statistics.xlsx")
