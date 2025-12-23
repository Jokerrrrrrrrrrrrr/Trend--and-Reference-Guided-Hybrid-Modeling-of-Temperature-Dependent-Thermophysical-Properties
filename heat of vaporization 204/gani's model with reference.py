# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures
# from scipy.optimize import least_squares
#
# # ==== 常数与路径 ====
# HV0, HVB, Tb0 = 9612.7, 15419.9, 222.543
# T_ref = 298.15
#
# # ==== 读取数据 ====
# df_main = pd.read_excel("heat of vaporization.xlsx", sheet_name="Sheet1")
# Nk_all = df_main.iloc[:, 13:25].apply(pd.to_numeric, errors='coerce')
# poly = PolynomialFeatures(degree=2, include_bias=False)
# Nk_poly = poly.fit_transform(Nk_all)
#
# # ==== Tb 模型 ====
# Tb_raw = df_main.iloc[:, 5].values
# mask_tb = ~np.isnan(Tb_raw)
# Nk_valid = Nk_all[mask_tb]
# Nk_poly_valid = poly.transform(Nk_valid)
# model_Tb = HuberRegressor(max_iter=5000).fit(Nk_poly_valid, np.exp(Tb_raw[mask_tb] / Tb0))
# Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly_valid), 1e-6, None))
#
# # ==== 预测 HVap_298 和 HVap_Tb ====
# df_298 = pd.read_excel("selected_25_descriptors_data_298.xlsx")
# X_298 = df_298.drop(columns=["Heat of vaporization at normal temperature"])
# rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["Heat of vaporization at normal temperature"])
# HVap_298_all = rf_298.predict(X_298)
#
# df_Tb = pd.read_excel("selected_25_descriptors_data_boiling_point.xlsx")
# X_Tb = df_Tb.drop(columns=["Heat of vaporization at boiling temperature"])
# rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["Heat of vaporization at boiling temperature"])
# HVap_Tb_all = rf_Tb.predict(X_Tb)
#
# # ==== slope × T 特征 ====
# slope_pred = ((HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)).reshape(-1, 1)
#
# # ==== 多温度点 ΔHvap ====
# T = df_main.iloc[:, 25:35].values[mask_tb]
# Hvap = df_main.iloc[:, 35:45].values[mask_tb]
# MW = df_main.iloc[:, 4].values[mask_tb].reshape(-1, 1)
# Nc = df_main.iloc[:, 10].values[mask_tb].reshape(-1, 1)
#
# # ==== 清洗有效样本 ====
# valid_row_mask = np.isfinite(Hvap).all(axis=1)
# Nk_valid = Nk_valid[valid_row_mask].values
# MW = MW[valid_row_mask]
# Nc = Nc[valid_row_mask]
# T = T[valid_row_mask]
# Hvap = Hvap[valid_row_mask]
# slope_pred = slope_pred[valid_row_mask]
#
# # ==== 构造训练数据 ====
# X = np.hstack([
#     Nk_valid.repeat(10, axis=0),
#     MW.repeat(10, axis=0),
#     Nc.repeat(10, axis=0),
#     T.flatten().reshape(-1, 1),
#     slope_pred.repeat(10, axis=0)
# ])
# y = Hvap.flatten()
#
# # ==== 清除非法值 ====
# mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
# X = X[mask]
# y = y[mask]
# T_valid = T.flatten()[mask]
#
# # ==== 拟合函数 ====
# def residuals(params, X, y):
#     Nk = X[:, :12]
#     MW = X[:, 12].reshape(-1, 1)
#     Nc = X[:, 13].reshape(-1, 1)
#     T = np.clip(X[:, 14].reshape(-1, 1), 1e-6, None)
#     slope = X[:, 15].reshape(-1, 1)
#
#     B1k = params[0:12]
#     B2k = params[12:24]
#     C1k = params[24:36]
#     C2k = params[36:48]
#     D1k = params[48:60]
#     D2k = params[60:72]
#     β, γ, δ = params[72:75]
#     f0, f1 = params[75:77]
#     γ_slope = params[77]
#     intercept = params[78]
#
#     R = 8.3144
#     Bi = np.sum(Nk * (B1k + MW * B2k), axis=1, keepdims=True) + β * (f0 + Nc * f1)
#     Ci = np.sum(Nk * (C1k + MW * C2k), axis=1, keepdims=True) + γ * (f0 + Nc * f1)
#     Di = np.sum(Nk * (D1k + MW * D2k), axis=1, keepdims=True) + δ * (f0 + Nc * f1)
#
#     y_pred = -R * ((1.5 * Bi) / np.sqrt(T) + Ci * T + Di * T**2) + γ_slope * slope * T + intercept
#     return y_pred.flatten() - y
#
# # ==== 模型拟合 ====
# params_init = np.zeros(79)
# result = least_squares(residuals, x0=params_init, args=(X, y), max_nfev=10000)
#
# # ==== 评估指标 ====
# y_pred = y - residuals(result.x, X, y)
# r2 = r2_score(y, y_pred)
# mse = mean_squared_error(y, y_pred)
# ard = np.mean(np.abs((y_pred - y) / y)) * 100
#
# print("\n📈 主模型评估（含 slope × T 和截距项）:")
# print(f"R²  = {r2:.6f}")
# print(f"MSE = {mse:.2f}")
# print(f"ARD = {ard:.2f}%")
#
# # ==== 输出参数 ====
# param_names = (
#     [f"B1_{i}" for i in range(12)] + [f"B2_{i}" for i in range(12)] +
#     [f"C1_{i}" for i in range(12)] + [f"C2_{i}" for i in range(12)] +
#     [f"D1_{i}" for i in range(12)] + [f"D2_{i}" for i in range(12)] +
#     ["beta", "gamma", "delta", "f0", "f1", "gamma_slope", "intercept"]
# )
# print("\n🔧 参数拟合结果:")
# for name, val in zip(param_names, result.x):
#     print(f"{name:14s}: {val:.6f}")
#
# # ==== 保存结果 ====
# compound_ids = np.repeat(df_main.iloc[mask_tb, 0].values[valid_row_mask], 10)[mask]
# df_result = pd.DataFrame({
#     "Compound_ID": compound_ids,
#     "Temperature (K)": T_valid,
#     "Hvap_true (J/mol)": y,
#     "Hvap_pred (J/mol)": y_pred,
#     "Absolute Error": np.abs(y - y_pred),
#     "Relative Error (%)": 100 * np.abs((y - y_pred) / y)
# })
# df_result.to_excel("Hvap_prediction_with_slopeT_and_intercept.xlsx", index=False)
# print("✅ 预测结果已保存为 Hvap_prediction_with_slopeT_and_intercept.xlsx")

import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares

# ==== 常数与路径 ====
HV0, HVB, Tb0 = 9612.7, 15419.9, 222.543
T_ref = 298.15

# ==== 读取数据 ====
df_main = pd.read_excel("heat of vaporization 204.xlsx", sheet_name="Sheet1")
Nk_all = df_main.iloc[:, 13:32].apply(pd.to_numeric, errors='coerce')  # 19基团
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)

# ==== Tb 模型 ====
Tb_raw = df_main.iloc[:, 5].values
mask_tb = ~np.isnan(Tb_raw)
Nk_valid = Nk_all[mask_tb]
Nk_poly_valid = poly.transform(Nk_valid)
model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly_valid, np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly_valid), 1e-6, None))

# ==== 预测 HVap_298 和 HVap_Tb ====
df_298 = pd.read_excel("selected_25_descriptors_data_298.xlsx")
X_298 = df_298.drop(columns=["Heat of vaporization at normal temperature"])
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["Heat of vaporization at normal temperature"])
HVap_298_all = rf_298.predict(X_298)

df_Tb = pd.read_excel("selected_25_descriptors_data_boiling_point.xlsx")
X_Tb = df_Tb.drop(columns=["Heat of vaporization at boiling temperature"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["Heat of vaporization at boiling temperature"])
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== slope × T 特征 ====
slope_pred = ((HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)).reshape(-1, 1)

# ==== 多温度点 ΔHvap ====
T = df_main.iloc[:, 32:42].values[mask_tb]
Hvap = df_main.iloc[:, 42:52].values[mask_tb]
MW = df_main.iloc[:, 4].values[mask_tb].reshape(-1, 1)
Nc = df_main.iloc[:, 10].values[mask_tb].reshape(-1, 1)

# ==== 清洗有效样本 ====
valid_row_mask = np.isfinite(Hvap).all(axis=1)
Nk_valid = Nk_valid[valid_row_mask].values
MW = MW[valid_row_mask]
Nc = Nc[valid_row_mask]
T = T[valid_row_mask]
Hvap = Hvap[valid_row_mask]
slope_pred = slope_pred[valid_row_mask]

# ==== 构造训练数据 ====
X = np.hstack([
    Nk_valid.repeat(10, axis=0),
    MW.repeat(10, axis=0),
    Nc.repeat(10, axis=0),
    T.flatten().reshape(-1, 1),
    slope_pred.repeat(10, axis=0)
])
y = Hvap.flatten()

# ==== 清除非法值 ====
mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
X = X[mask]
y = y[mask]
T_valid = T.flatten()[mask]

# ==== 拟合函数（支持 19 基团） ====
def residuals(params, X, y):
    Nk = X[:, :19]
    MW = X[:, 19].reshape(-1, 1)
    Nc = X[:, 20].reshape(-1, 1)
    T = np.clip(X[:, 21].reshape(-1, 1), 1e-6, None)
    slope = X[:, 22].reshape(-1, 1)

    B1k = params[0:19]
    B2k = params[19:38]
    C1k = params[38:57]
    C2k = params[57:76]
    D1k = params[76:95]
    D2k = params[95:114]
    β, γ, δ = params[114:117]
    f0, f1 = params[117:119]
    γ_slope = params[119]
    intercept = params[120]

    R = 8.3144
    Bi = np.sum(Nk * (B1k + MW * B2k), axis=1, keepdims=True) + β * (f0 + Nc * f1)
    Ci = np.sum(Nk * (C1k + MW * C2k), axis=1, keepdims=True) + γ * (f0 + Nc * f1)
    Di = np.sum(Nk * (D1k + MW * D2k), axis=1, keepdims=True) + δ * (f0 + Nc * f1)

    y_pred = -R * ((1.5 * Bi) / np.sqrt(T) + Ci * T + Di * T**2) + γ_slope * slope * T + intercept
    return y_pred.flatten() - y

# ==== 模型拟合 ====
params_init = np.zeros(121)
result = least_squares(residuals, x0=params_init, args=(X, y), max_nfev=10000)

# ==== 评估指标 ====
y_pred = y - residuals(result.x, X, y)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
ard = np.mean(np.abs((y_pred - y) / y)) * 100

print("\n📈 主模型评估（含 slope × T 和截距项）:")
print(f"R²  = {r2:.6f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")
# ==== 误差统计（1%、5%、10%） ====
relative_error = np.abs((y_pred - y) / y) * 100
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print(f"\n✅ 相对误差 ≤ 1% 的点数: {within_1pct}")
print(f"✅ 相对误差 ≤ 5% 的点数: {within_5pct}")
print(f"✅ 相对误差 ≤ 10% 的点数: {within_10pct}")

# ==== 输出参数 ====
param_names = (
    [f"B1_{i}" for i in range(19)] + [f"B2_{i}" for i in range(19)] +
    [f"C1_{i}" for i in range(19)] + [f"C2_{i}" for i in range(19)] +
    [f"D1_{i}" for i in range(19)] + [f"D2_{i}" for i in range(19)] +
    ["beta", "gamma", "delta", "f0", "f1", "gamma_slope", "intercept"]
)
print("\n🔧 参数拟合结果:")
for name, val in zip(param_names, result.x):
    print(f"{name:14s}: {val:.6f}")

# ==== 保存结果 ====
compound_ids = np.repeat(df_main.iloc[mask_tb, 0].values[valid_row_mask], 10)[mask]
df_result = pd.DataFrame({
    "Compound_ID": compound_ids,
    "Temperature (K)": T_valid,
    "Hvap_true (J/mol)": y,
    "Hvap_pred (J/mol)": y_pred,
    "Absolute Error": np.abs(y - y_pred),
    "Relative Error (%)": 100 * np.abs((y - y_pred) / y)
})
df_result.to_excel("Hvap_prediction_with_slopeT_and_intercept_19group.xlsx", index=False)
print("✅ 预测结果已保存为 Hvap_prediction_with_slopeT_and_intercept_19group.xlsx")
