# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import HuberRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from scipy.optimize import least_squares
#
# # === Step 1: 读取数据 === #
# df = pd.read_excel("vp209.xlsx", sheet_name="Sheet1")
#
# # 提取基础特征
# MW = df.iloc[:, 4].values.reshape(-1, 1)
# Nc = df.iloc[:, 10].values.reshape(-1, 1)
# Ncs = df.iloc[:, 9].values.reshape(-1, 1)
# Nk = df.iloc[:, 12:31].values  # 19个基团
# T = df.iloc[:, 31:41].values
# P_vp = df.iloc[:, 41:51].values
#
# # 过滤掉非法样本
# valid_mask = np.isfinite(P_vp) & (P_vp > 0)
# valid_mask = valid_mask.all(axis=1)
# MW, Nc, Ncs, Nk, T, P_vp = [x[valid_mask] for x in [MW, Nc, Ncs, Nk, T, P_vp]]
# df_valid = df[valid_mask].reset_index(drop=True)
#
# # 构造 Nk 多项式特征
# poly = PolynomialFeatures(degree=2, include_bias=False)
# Nk_poly = poly.fit_transform(Nk)
#
# # ====== Tb 模型 ====== #
# Tb0 = 222.543
# Tb = df_valid.iloc[:, 5].values
# model_tb = HuberRegressor(max_iter=10000).fit(Nk_poly, np.exp(Tb / Tb0))
# Tb_pred = Tb0 * np.log(np.clip(model_tb.predict(Nk_poly), 1e-6, None))
#
# # ====== Tc 模型（使用 GBDT） ====== #
# Tc_half = df_valid['ASPEN Half Critical T'].values
# gb_model_tc = GradientBoostingRegressor(
#     n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
# )
# gb_model_tc.fit(Nk_poly, Tc_half)
# Tc_pred_full = gb_model_tc.predict(Nk_poly) * 2
#
# # ====== Pc 模型 ====== #
# Pc_bar = df_valid.iloc[:, 51].values
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
# result_pc = least_squares(
#     residual_pc, x0=params_init_pc, args=(Pc_poly, MW_flat, Pc_bar), max_nfev=5000
# )
# x_fit = Pc_poly @ result_pc.x[:-1] + 0.108998
# Pc_pred = (5.9827 + (1 / x_fit) ** 2 + result_pc.x[-1] * np.exp(1 / MW_flat)) * 1e5  # Pa
#
# # ====== 计算 slope ====== #
# Pb = 101325  # 标准大气压
# slope_all = (np.log(Pc_pred) - np.log(Pb)) / (Tc_pred_full - Tb_pred)
# slope_all = slope_all.reshape(-1, 1)
#
# # 保存为 CSV 文件
# slope_df = pd.DataFrame({
#     "Index": df_valid.index + 1,  # 每行编号从1开始
#     "slope": slope_all.flatten()
# })
# slope_df.to_csv("vp_slope_values.csv", index=False)
# print("✅ slope 值已保存为 vp_slope_values.csv")

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import least_squares

# === Step 1: 读取数据 === #
df = pd.read_excel("vp209.xlsx", sheet_name="Sheet1")

# 提取基础特征
MW = df.iloc[:, 4].values.reshape(-1, 1)
Nc = df.iloc[:, 10].values.reshape(-1, 1)
Ncs = df.iloc[:, 9].values.reshape(-1, 1)
Nk = df.iloc[:, 12:31].values  # 19个基团
T = df.iloc[:, 31:41].values
P_vp = df.iloc[:, 41:51].values

# 过滤掉非法样本
valid_mask = np.isfinite(P_vp) & (P_vp > 0)
valid_mask = valid_mask.all(axis=1)
MW, Nc, Ncs, Nk, T, P_vp = [x[valid_mask] for x in [MW, Nc, Ncs, Nk, T, P_vp]]
df_valid = df[valid_mask].reset_index(drop=True)

# 构造 Nk 多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk)

# ====== Tb 模型 ====== #
Tb0 = 222.543
Tb = df_valid.iloc[:, 5].values
model_tb = HuberRegressor(max_iter=10000).fit(Nk_poly, np.exp(Tb / Tb0))
Tb_pred = Tb0 * np.log(np.clip(model_tb.predict(Nk_poly), 1e-6, None))

# ====== Tb 模型评估 ====== #
Tb_pred_eval = Tb_pred
Tb_true_eval = Tb
r2_tb = np.corrcoef(Tb_true_eval, Tb_pred_eval)[0, 1] ** 2
mse_tb = np.mean((Tb_true_eval - Tb_pred_eval) ** 2)
print(f"\n📌 Tb 模型评估:")
print(f"R²  = {r2_tb:.4f}")
print(f"MSE = {mse_tb:.2f}")

# ====== Tc 模型（使用 GBDT） ====== #
Tc_half = df_valid['ASPEN Half Critical T'].values
gb_model_tc = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
)
gb_model_tc.fit(Nk_poly, Tc_half)
Tc_pred_full = gb_model_tc.predict(Nk_poly) * 2

# ====== Tc 模型评估（仅对 Tc_half） ====== #
r2_tc = np.corrcoef(Tc_half, gb_model_tc.predict(Nk_poly))[0, 1] ** 2
mse_tc = np.mean((Tc_half - gb_model_tc.predict(Nk_poly)) ** 2)
print(f"\n📌 Tc_half 模型评估:")
print(f"R²  = {r2_tc:.4f}")
print(f"MSE = {mse_tc:.2f}")

# ====== Pc 模型 ====== #
Pc_bar = df_valid.iloc[:, 51].values
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
result_pc = least_squares(
    residual_pc, x0=params_init_pc, args=(Pc_poly, MW_flat, Pc_bar), max_nfev=5000
)
x_fit = Pc_poly @ result_pc.x[:-1] + 0.108998
Pc_pred = (5.9827 + (1 / x_fit) ** 2 + result_pc.x[-1] * np.exp(1 / MW_flat)) * 1e5  # Pa

# ====== Pc 模型评估 ====== #
Pc_true = Pc_bar * 1e5  # bar 转 Pa
r2_pc = np.corrcoef(Pc_true, Pc_pred)[0, 1] ** 2
mse_pc = np.mean((Pc_true - Pc_pred) ** 2)
print(f"\n📌 Pc 模型评估:")
print(f"R²  = {r2_pc:.4f}")
print(f"MSE = {mse_pc:.2e}")

# ====== 计算 slope ====== #
Pb = 101325  # 标准大气压
slope_all = (np.log(Pc_pred) - np.log(Pb)) / (Tc_pred_full - Tb_pred)
slope_all = slope_all.reshape(-1, 1)

# 保存为 CSV 文件
slope_df = pd.DataFrame({
    "Index": df_valid.index + 1,  # 每行编号从1开始
    "slope": slope_all.flatten()
})
slope_df.to_csv("vp_slope_values.csv", index=False)
print("\n✅ slope 值已保存为 vp_slope_values.csv")
