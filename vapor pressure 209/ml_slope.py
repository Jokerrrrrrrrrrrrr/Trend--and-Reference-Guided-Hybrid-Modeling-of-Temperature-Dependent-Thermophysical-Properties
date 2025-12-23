# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import HuberRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from scipy.optimize import least_squares
#
# # ========== 读取数据 ========== #
# df = pd.read_excel("vp209.xlsx", sheet_name='Sheet1')
#
# # 特征提取
# MW = df.iloc[:, 4].values.reshape(-1, 1)
# Nc = df.iloc[:, 10].values.reshape(-1, 1)
# Ncs = df.iloc[:, 9].values.reshape(-1, 1)
# Nk = df.iloc[:, 12:31].values  # 19个基团
# T = df.iloc[:, 31:41].values
# P_vp = df.iloc[:, 41:51].values
#
# # ========= 清理非法样本 ========= #
# valid_mask = np.isfinite(P_vp) & (P_vp > 0)
# valid_mask = valid_mask.all(axis=1)
# MW, Nc, Ncs, Nk, T, P_vp = [x[valid_mask] for x in [MW, Nc, Ncs, Nk, T, P_vp]]
#
# # ========= 构建 Nk_poly ========= #
# poly = PolynomialFeatures(degree=2, include_bias=False)
# Nk_poly = poly.fit_transform(Nk)
#
# # ========= Tb 模型 ========= #
# Tb0 = 222.543
# Tb = df.iloc[:, 5].values[valid_mask]
# model_tb = HuberRegressor(max_iter=10000).fit(Nk_poly, np.exp(Tb / Tb0))
# Tb_pred = Tb0 * np.log(np.clip(model_tb.predict(Nk_poly), 1e-6, None))
#
# # ========= Tc 模型 (Gradient Boosting) ========= #
# Tc_half = df['ASPEN Half Critical T'].values[valid_mask]
# gb_model_tc = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0)
# gb_model_tc.fit(Nk_poly, Tc_half)
# Tc_pred = gb_model_tc.predict(Nk_poly)
# Tc_pred_full = Tc_pred * 2
#
# # ========= Pc 模型 ========= #
# Pc_bar = df.iloc[:, 51].values[valid_mask]
# MW_flat = MW.flatten()
# Pc_poly = poly.fit_transform(Nk)
#
# def residual_pc(params, X, MW, Pc_true):
#     beta = params[:-1]
#     beta3 = params[-1]
#     y_pred = X @ beta
#     x_pred = y_pred + 0.108998
#     Pc_pred = 5.9827 + (1 / x_pred) ** 2 + beta3 * np.exp(1 / MW)
#     return Pc_pred - Pc_true
#
# params_init_pc = np.zeros(Pc_poly.shape[1] + 1)
# result_pc = least_squares(residual_pc, x0=params_init_pc, args=(Pc_poly, MW_flat, Pc_bar), max_nfev=5000)
# x_fit = Pc_poly @ result_pc.x[:-1] + 0.108998
# Pc_pred = (5.9827 + (1 / x_fit) ** 2 + result_pc.x[-1] * np.exp(1 / MW_flat)) * 1e5  # Pa
#
# # ========= slope × T 特征构建 ========= #
# Pb = 101325  # 标准大气压 Pa
# slope_all = (np.log(Pc_pred) - np.log(Pb)) / (Tc_pred_full - Tb_pred)
# slope_all = slope_all.reshape(-1, 1)
# slope_T = slope_all.repeat(10, axis=0) * T.flatten().reshape(-1, 1)
#
# # ========= 构造训练数据 ========= #
# X = np.hstack([
#     Nk.repeat(10, axis=0),
#     T.flatten().reshape(-1, 1),
#     slope_T
# ])
# y = np.log(P_vp).flatten()
#
# finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
# X, y = X[finite_mask], y[finite_mask]
# T_valid = T.flatten()[finite_mask]
#
# # ========= 模型训练 ========= #
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X, y)
#
# # ========= 模型评估 ========= #
# y_pred = model.predict(X)
# print("\n📈 蒸汽压模型对 ln(P) 拟合结果：")
# print(f"R² (lnP) = {r2_score(y, y_pred):.6f}")
# print(f"MSE (lnP) = {mean_squared_error(y, y_pred):.6f}")
#
# P_true = np.exp(y)
# P_pred = np.exp(y_pred)
# mse_P = mean_squared_error(P_true, P_pred)
# r2_P = r2_score(P_true, P_pred)
# ard_P = np.mean(np.abs((P_pred - P_true) / P_true)) * 100
#
# print("\n📈 实际蒸汽压 P 评估：")
# print(f"R² (P)  = {r2_P:.6f}")
# print(f"MSE (P) = {mse_P:.4f}")
# print(f"ARD (P) = {ard_P:.2f}%")
#
# # ========= 保存预测结果 ========= #
# compare_df = pd.DataFrame({
#     "Temperature_K": T_valid,
#     "ln(P)_true": y,
#     "ln(P)_pred": y_pred,
#     "Absolute_Error_lnP": np.abs(y - y_pred),
#     "Relative_Error_lnP (%)": 100 * np.abs((y - y_pred) / y),
#     "P_true": P_true,
#     "P_pred": P_pred,
#     "Absolute_Error_P": np.abs(P_true - P_pred),
#     "Relative_Error_P (%)": 100 * np.abs((P_true - P_pred) / P_true)
# })
# compare_df.to_excel("VaporPrediction_ML_lnP_RF_with_slopeT.xlsx", index=False)
# print("\n✅ 结果已保存为 VaporPrediction_ML_lnP_RF_with_slopeT.xlsx")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import least_squares

# ========== 读取数据 ========== #
df = pd.read_excel("vp209.xlsx", sheet_name='Sheet1')

# 特征提取
MW = df.iloc[:, 4].values.reshape(-1, 1)
Nc = df.iloc[:, 10].values.reshape(-1, 1)
Ncs = df.iloc[:, 9].values.reshape(-1, 1)
Nk = df.iloc[:, 12:31].values  # 19个基团
T = df.iloc[:, 31:41].values
P_vp = df.iloc[:, 41:51].values

# ========= 清理非法样本 ========= #
valid_mask = np.isfinite(P_vp) & (P_vp > 0)
valid_mask = valid_mask.all(axis=1)
MW, Nc, Ncs, Nk, T, P_vp = [x[valid_mask] for x in [MW, Nc, Ncs, Nk, T, P_vp]]

# ========= 构建 Nk_poly ========= #
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk)

# ========= Tb 模型 ========= #
Tb0 = 222.543
Tb = df.iloc[:, 5].values[valid_mask]
model_tb = HuberRegressor(max_iter=10000).fit(Nk_poly, np.exp(Tb / Tb0))
Tb_pred = Tb0 * np.log(np.clip(model_tb.predict(Nk_poly), 1e-6, None))

# ========= Tc 模型 (Gradient Boosting) ========= #
Tc_half = df['ASPEN Half Critical T'].values[valid_mask]
gb_model_tc = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0)
gb_model_tc.fit(Nk_poly, Tc_half)
Tc_pred = gb_model_tc.predict(Nk_poly)
Tc_pred_full = Tc_pred * 2

# ========= Pc 模型 ========= #
Pc_bar = df.iloc[:, 51].values[valid_mask]
MW_flat = MW.flatten()
Pc_poly = poly.fit_transform(Nk)

def residual_pc(params, X, MW, Pc_true):
    beta = params[:-1]
    beta3 = params[-1]
    y_pred = X @ beta
    x_pred = y_pred + 0.108998
    Pc_pred = 5.9827 + (1 / x_pred) ** 2 + beta3 * np.exp(1 / MW)
    return Pc_pred - Pc_true

params_init_pc = np.zeros(Pc_poly.shape[1] + 1)
result_pc = least_squares(residual_pc, x0=params_init_pc, args=(Pc_poly, MW_flat, Pc_bar), max_nfev=5000)
x_fit = Pc_poly @ result_pc.x[:-1] + 0.108998
Pc_pred = (5.9827 + (1 / x_fit) ** 2 + result_pc.x[-1] * np.exp(1 / MW_flat)) * 1e5  # Pa

# ========= slope × T 特征构建 ========= #
Pb = 101325  # 标准大气压 Pa
slope_all = (np.log(Pc_pred) - np.log(Pb)) / (Tc_pred_full - Tb_pred)
slope_all = slope_all.reshape(-1, 1)
slope_T = slope_all.repeat(10, axis=0) * T.flatten().reshape(-1, 1)

# ========= 构造训练数据 ========= #
X = np.hstack([
    Nk.repeat(10, axis=0),
    T.flatten().reshape(-1, 1),
    slope_T
])
y = np.log(P_vp).flatten()

finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
X, y = X[finite_mask], y[finite_mask]
T_valid = T.flatten()[finite_mask]

# ========= 模型训练 ========= #
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ========= 模型评估 ========= #
y_pred = model.predict(X)
print("\n📈 蒸汽压模型对 ln(P) 拟合结果：")
print(f"R² (lnP) = {r2_score(y, y_pred):.6f}")
print(f"MSE (lnP) = {mean_squared_error(y, y_pred):.6f}")

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

# ========= 保存预测结果 ========= #
compare_df = pd.DataFrame({
    "Temperature_K": T_valid,
    "ln(P)_true": y,
    "ln(P)_pred": y_pred,
    "Absolute_Error_lnP": np.abs(y - y_pred),
    "Relative_Error_lnP (%)": 100 * np.abs((y - y_pred) / y),
    "P_true": P_true,
    "P_pred": P_pred,
    "Absolute_Error_P": np.abs(P_true - P_pred),
    "Relative_Error_P (%)": 100 * np.abs((P_true - P_pred) / P_true)
})
compare_df.to_excel("VaporPrediction_ML_lnP_RF_with_slopeT.xlsx", index=False)
print("\n✅ 结果已保存为 VaporPrediction_ML_lnP_RF_with_slopeT.xlsx")

#
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import HuberRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from scipy.optimize import least_squares
#
# # ========== 读取数据 ========== #
# df = pd.read_excel("Vapor pressure.xlsx", sheet_name='Sheet7')
#
# # 特征提取
# MW = df.iloc[:, 4].values.reshape(-1, 1)
# Nc = df.iloc[:, 10].values.reshape(-1, 1)
# Ncs = df.iloc[:, 9].values.reshape(-1, 1)
# Nk = df.iloc[:, 12:24].values
# T = df.iloc[:, 24:34].values
# P_vp = df.iloc[:, 34:44].values
#
# # ========= 清理非法样本 ========= #
# valid_mask = np.isfinite(P_vp) & (P_vp > 0)
# valid_mask = valid_mask.all(axis=1)
# MW, Nc, Ncs, Nk, T, P_vp = [x[valid_mask] for x in [MW, Nc, Ncs, Nk, T, P_vp]]
#
# # ========= 构建 Nk_poly ========= #
# poly = PolynomialFeatures(degree=2, include_bias=False)
# Nk_poly = poly.fit_transform(Nk)
#
# # ========= Tb 模型 ========= #
# Tb0 = 222.543
# Tb = df.iloc[:, 5].values[valid_mask]
# model_tb = HuberRegressor(max_iter=5000).fit(Nk_poly, np.exp(Tb / Tb0))
# Tb_pred = Tb0 * np.log(np.clip(model_tb.predict(Nk_poly), 1e-6, None))
# r2_tb = r2_score(Tb, Tb_pred)
# mse_tb = mean_squared_error(Tb, Tb_pred)
#
# # ========= Tc 模型 ========= #
# Tc_half = df['ASPEN Half Critical T'].values[valid_mask]
# gb_model_tc = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0)
# gb_model_tc.fit(Nk_poly, Tc_half)
# Tc_pred_half = gb_model_tc.predict(Nk_poly)
# Tc_pred = Tc_pred_half * 2
# r2_tc = r2_score(Tc_half * 2, Tc_pred)
# mse_tc = mean_squared_error(Tc_half * 2, Tc_pred)
#
# # ========= Pc 模型 ========= #
# Pc_bar = df.iloc[:, 44].values[valid_mask]
# MW_flat = MW.flatten()
# Pc_poly = poly.fit_transform(Nk)
#
# def residual_pc(params, X, MW, Pc_true):
#     beta = params[:-1]
#     beta3 = params[-1]
#     y_pred = X @ beta
#     x_pred = y_pred + 0.108998
#     Pc_pred = 5.9827 + (1 / x_pred) ** 2 + beta3 * np.exp(1 / MW)
#     return Pc_pred - Pc_true
#
# params_init_pc = np.zeros(Pc_poly.shape[1] + 1)
# result_pc = least_squares(residual_pc, x0=params_init_pc, args=(Pc_poly, MW_flat, Pc_bar), max_nfev=5000)
# x_fit = Pc_poly @ result_pc.x[:-1] + 0.108998
# Pc_pred = (5.9827 + (1 / x_fit) ** 2 + result_pc.x[-1] * np.exp(1 / MW_flat)) * 1e5  # Pa
# r2_pc = r2_score(Pc_bar, Pc_pred / 1e5)
# mse_pc = mean_squared_error(Pc_bar, Pc_pred / 1e5)
#
# # ========= slope × T 特征构建 ========= #
# Pb = 101325
# slope_all = (np.log(Pc_pred) - np.log(Pb)) / (Tc_pred - Tb_pred)
# slope_all = slope_all.reshape(-1, 1)
# slope_T = slope_all.repeat(10, axis=0) * T.flatten().reshape(-1, 1)
#
# # ========= 构造训练数据 ========= #
# X = np.hstack([
#     Nk.repeat(10, axis=0),
#     MW.repeat(10, axis=0),
#     Nc.repeat(10, axis=0),
#     Ncs.repeat(10, axis=0),
#     T.flatten().reshape(-1, 1),
#     slope_T
# ])
# y = np.log(P_vp).flatten()
# finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
# X, y = X[finite_mask], y[finite_mask]
# T_valid = T.flatten()[finite_mask]
#
# # ========= 主模型训练 ========= #
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X, y)
# y_pred = model.predict(X)
#
# # ========= 蒸汽压模型评估 ========= #
# print("\n📈 蒸汽压模型（ln(P)）评估：")
# print(f"R² (lnP) = {r2_score(y, y_pred):.6f}")
# print(f"MSE (lnP) = {mean_squared_error(y, y_pred):.6f}")
#
# P_true = np.exp(y)
# P_pred = np.exp(y_pred)
# print("\n📈 实际蒸汽压 P 评估：")
# print(f"R² (P) = {r2_score(P_true, P_pred):.6f}")
# print(f"MSE (P) = {mean_squared_error(P_true, P_pred):.4f}")
#
# # ========= 各子模型评估 ========= #
# print("\n🔍 子模型预测性能评估：")
# print(f"📊 Tb 模型：R² = {r2_tb:.6f}, MSE = {mse_tb:.4f} K²")
# print(f"📊 Tc 模型：R² = {r2_tc:.6f}, MSE = {mse_tc:.4f} K²")
# print(f"📊 Pc 模型：R² = {r2_pc:.6f}, MSE = {mse_pc:.4f} bar²")
#
# # ========= 保存结果 ========= #
# compare_df = pd.DataFrame({
#     "Temperature_K": T_valid,
#     "ln(P)_true": y,
#     "ln(P)_pred": y_pred,
#     "Absolute_Error_lnP": np.abs(y - y_pred),
#     "Relative_Error_lnP (%)": 100 * np.abs((y - y_pred) / y),
#     "P_true": P_true,
#     "P_pred": P_pred,
#     "Absolute_Error_P": np.abs(P_true - P_pred),
#     "Relative_Error_P (%)": 100 * np.abs((P_true - P_pred) / P_true)
# })
# compare_df.to_excel("VaporPrediction_ML_lnP_RF_with_slopeT.xlsx", index=False)
# print("✅ 结果已保存为 VaporPrediction_ML_lnP_RF_with_slopeT.xlsx")
