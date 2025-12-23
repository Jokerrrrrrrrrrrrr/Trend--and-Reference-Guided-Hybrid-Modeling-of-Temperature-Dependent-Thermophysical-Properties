# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
#
# # 1. 读取数据
# file_path = "vp209.xlsx"
# sheet1_df = pd.read_excel(file_path, sheet_name='Sheet1')
#
# # 2. 提取数据列（适配19个基团）
# MW = sheet1_df.iloc[:, 4].values.reshape(-1, 1)
# Ncs = sheet1_df.iloc[:, 9].values.reshape(-1, 1)
# Nc = sheet1_df.iloc[:, 10].values.reshape(-1, 1)
# Nk = sheet1_df.iloc[:, 12:31].values  # 19个基团
# temperatures = sheet1_df.iloc[:, 31:41].values
# vapor_pressures = sheet1_df.iloc[:, 41:51].values
#
# # 3. 计算变换后的变量
# ln_vapor_pressures = np.log(vapor_pressures)
# inv_temperatures = 1 / temperatures
# ln_temperatures = np.log(temperatures)
#
# # 4. 扩展特征以匹配温度点数量
# def tile_features(feature, n_times=10):
#     return np.tile(feature, (n_times, 1)).T.reshape(-1, 1)
#
# MW_tile = tile_features(MW)
# Nc_tile = tile_features(Nc)
# Ncs_tile = tile_features(Ncs)
# Nk_tile = np.repeat(Nk, 10, axis=0)
#
# # 5. 构造特征矩阵
# features = []
#
# # === A 部分 ===
# for k in range(19):
#     features.append(Nk_tile[:, k].reshape(-1, 1))                      # A1k
#     features.append(Nk_tile[:, k].reshape(-1, 1) * MW_tile)           # A2k
# features.append(np.ones((len(MW_tile), 1)))                           # s0
# features.append(Ncs_tile)                                             # s1 × Ncs
# features.append(Nc_tile)                                              # α × (f0 + Nc × f1)
#
# # === B 部分 ===
# for k in range(19):
#     features.append(Nk_tile[:, k].reshape(-1, 1) * inv_temperatures.flatten().reshape(-1, 1))  # B1k
#     features.append(Nk_tile[:, k].reshape(-1, 1) * MW_tile * inv_temperatures.flatten().reshape(-1, 1))  # B2k
# features.append(Nc_tile * inv_temperatures.flatten().reshape(-1, 1))  # β × (f0 + Nc × f1)
#
# # === C 部分 ===
# for k in range(19):
#     features.append(Nk_tile[:, k].reshape(-1, 1) * ln_temperatures.flatten().reshape(-1, 1))  # C1k
#     features.append(Nk_tile[:, k].reshape(-1, 1) * MW_tile * ln_temperatures.flatten().reshape(-1, 1))  # C2k
#
# X = np.hstack(features)
# y = ln_vapor_pressures.flatten()
#
# # 6. 模型拟合
# model = LinearRegression(fit_intercept=False)
# model.fit(X, y)
#
# # 7. 参数映射表
# param_names = []
# for k in range(19):
#     param_names.append(f"A1_{k} (N_{k})")
#     param_names.append(f"A2_{k} (N_{k}*M)")
# param_names.extend(["s0 (截距)", "s1 (Ncs)", "α*(f0 + Nc*f1)"])
# for k in range(19):
#     param_names.append(f"B1_{k} (N_{k}/T)")
#     param_names.append(f"B2_{k} (N_{k}*M/T)")
# param_names.append("β*(f0 + Nc*f1)")
# for k in range(19):
#     param_names.append(f"C1_{k} (N_{k}*ln(T))")
#     param_names.append(f"C2_{k} (N_{k}*M*ln(T))")
#
# # 8. 模型评估
# y_pred = model.predict(X)
# mse = mean_squared_error(y, y_pred)
# r2 = r2_score(y, y_pred)
#
# print(f"\n📊 模型性能:")
# print(f"MSE: {mse:.4f}")
# print(f"R² : {r2:.4f}")
#
# # 9. 保存参数系数
# coefficients_df = pd.DataFrame({
#     "Parameter": param_names,
#     "Coefficient": model.coef_
# })
# coefficients_df.to_csv("regression_coefficients_19groups.csv", index=False)
#
# # 10. 输出预测结果对比表
# comparison_df = pd.DataFrame({
#     "化合物编号": np.repeat(sheet1_df.iloc[:, 0].values, 10),
#     "温度 (K)": temperatures.flatten(),
#     "实际 ln(P_vp)": y,
#     "预测 ln(P_vp)": y_pred,
#     "误差": y_pred - y
# })
# comparison_df.to_excel("lnPv_预测与实际对比_19group.xlsx", index=False)
#
# print("✅ 参数系数已保存为 regression_coefficients_19groups.csv")
# print("✅ 预测结果已保存为 lnPv_预测与实际对比_19group.xlsx")
#
# import numpy as np
# import pandas as pd
# from scipy.optimize import least_squares
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ==================== 数据加载 ====================
# file_path = "vp209.xlsx"
# df = pd.read_excel(file_path, sheet_name='Sheet1')
#
# # 基本属性提取（适配19个基团）
# MW = df.iloc[:, 4].values.reshape(-1, 1)
# Nc = df.iloc[:, 10].values.reshape(-1, 1)
# Ncs = df.iloc[:, 9].values.reshape(-1, 1)
# Nk = df.iloc[:, 12:31].values  # 19个基团
# T = df.iloc[:, 31:41].values
# P_vp = df.iloc[:, 41:51].values
#
# # ==================== 清洗非法值 ====================
# valid_mask = np.isfinite(P_vp) & (P_vp > 0)
# valid_mask = valid_mask.all(axis=1)
#
# MW = MW[valid_mask]
# Nc = Nc[valid_mask]
# Ncs = Ncs[valid_mask]
# Nk = Nk[valid_mask]
# T = T[valid_mask]
# P_vp = P_vp[valid_mask]
#
# # 构造 y 和 X
# y = np.log(P_vp).flatten()
# X = np.hstack([
#     Nk.repeat(10, axis=0),
#     MW.repeat(10, axis=0),
#     Nc.repeat(10, axis=0),
#     Ncs.repeat(10, axis=0),
#     T.flatten().reshape(-1, 1)
# ])
#
# finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
# y = y[finite_mask]
# X = X[finite_mask, :]
#
# # ==================== 残差函数（适配19个基团） ====================
# def residuals(params, X, y):
#     Nk = X[:, :19]
#     MW = X[:, 19].reshape(-1, 1)
#     Nc = X[:, 20].reshape(-1, 1)
#     Ncs = X[:, 21].reshape(-1, 1)
#     T = np.clip(X[:, 22].reshape(-1, 1), 1e-6, None)
#
#     A1k = params[:19]
#     A2k = params[19:38]
#     s0, s1 = params[38], params[39]
#     alpha, f0, f1 = params[40], params[41], params[42]
#     B1k = params[43:62]
#     B2k = params[62:81]
#     beta = params[81]
#     C1k = params[82:101]
#     C2k = params[101:120]
#
#     term_A = np.sum(Nk * (A1k + MW * A2k), axis=1) + (s0 + Ncs.flatten() * s1) + alpha * (f0 + Nc.flatten() * f1)
#     term_B = np.sum(Nk * (B1k + MW * B2k), axis=1) + beta * (f0 + Nc.flatten() * f1)
#     term_C = np.sum(Nk * (C1k + MW * C2k), axis=1)
#
#     y_pred = term_A + term_B / T.flatten() + term_C * np.log(T.flatten())
#     return y - y_pred
#
# # ==================== 初始参数（适配19个基团） ====================
# params_init = np.zeros(120)
#
# # A1k (前12个真实值 + 后7个设为0)
# params_init[:12] = [17.3683, 7.7102, -15.2653, 11.2186, 9.3464, -13.1484,
#                     88.5031, 89.9539, 44.3037, 7.3199, 22.4376, -30.5308]
#
# # A2k
# params_init[19:31] = [0.0273, -0.0085, 0.0922, 0.0067, -0.0218, 0.0423,
#                       0.0988, 0.0580, 0.2499, -0.0100, 0.0131, 0.1849]
#
# # s0, s1, alpha, f0, f1
# params_init[38:43] = [18.3710, 0.0484, 0.0484, 0.0, 1.0]
#
# # B1k
# params_init[43:55] = [-1336.22, -1119.38, 770.74, -1231.84, -1157.32, -148.52,
#                       -8148.63, -9960.01, -5144.31, -1049.42, -2463.69, 1541.48]
#
# # B2k
# params_init[62:74] = [-0.6448, 0.6124, -9.6981, -0.7783, 1.6822, -2.0678,
#                       -5.6822, -3.0283, -21.1782, 1.0146, -1.5566, -19.3962]
#
# # beta
# params_init[81] = -40.7015
#
# # C1k
# params_init[82:94] = [-2.4928, -0.9628, 2.2021, -1.4518, -1.2337, 1.9651,
#                       -11.8996, -11.8098, -5.7519, -0.9450, -2.9035, 4.4042]
#
# # C2k
# params_init[101:113] = [-0.0043, 0.0012, -0.0120, -0.0008, 0.0029, -0.0062,
#                         -0.0141, -0.0084, -0.0335, 0.0013, -0.0017, -0.0239]
#
# # ==================== 拟合 ====================
# print("🚀 拟合中，请稍候...")
# result = least_squares(residuals, x0=params_init, args=(X, y), max_nfev=10000)
#
# # ==================== 输出参数 ====================
# param_names = (
#     [f"A1_{i}" for i in range(19)] +
#     [f"A2_{i}" for i in range(19)] +
#     ["s0", "s1", "alpha", "f0", "f1"] +
#     [f"B1_{i}" for i in range(19)] +
#     [f"B2_{i}" for i in range(19)] +
#     ["beta"] +
#     [f"C1_{i}" for i in range(19)] +
#     [f"C2_{i}" for i in range(19)]
# )
#
# print("\n🔧 拟合后的参数：")
# for name, val in zip(param_names, result.x):
#     print(f"{name:10s}: {val:.6f}")
#
# # ==================== 模型评估 ====================
# y_pred = y - residuals(result.x, X, y)
# mse = mean_squared_error(y, y_pred)
# r2 = r2_score(y, y_pred)
# print(f"\n📈 模型性能:")
# print(f"R²  = {r2:.6f}")
# print(f"MSE = {mse:.6f}")
#
# # ==================== 实际蒸汽压评估（含 ARD） ====================
# P_true = np.exp(y)
# P_pred = np.exp(y_pred)
# mse_real = mean_squared_error(P_true, P_pred)
# r2_real = r2_score(P_true, P_pred)
# ard_real = np.mean(np.abs((P_pred - P_true) / P_true)) * 100
#
# print(f"\n📈 实际蒸汽压评估:")
# print(f"R² (P)  = {r2_real:.6f}")
# print(f"MSE (P) = {mse_real:.4f}")
# print(f"ARD (P) = {ard_real:.2f}%")
#
# # ==================== 保存结果 ====================
# compare_df = pd.DataFrame({
#     "Compound_ID": np.repeat(df.iloc[valid_mask, 0].values, 10)[finite_mask],
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
# compare_df.to_excel("Gani_lnP_prediction_results_cleaned_19group.xlsx", index=False)
# print("\n✅ 已保存预测结果为 Gani_lnP_prediction_results_cleaned_19group.xlsx")
# import numpy as np
# import pandas as pd
# from scipy.optimize import least_squares
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ==================== 数据加载 ====================
# file_path = "vp209.xlsx"
# df = pd.read_excel(file_path, sheet_name='Sheet1')
#
# # 基本属性提取（适配19个基团）
# MW = df.iloc[:, 4].values.reshape(-1, 1)
# Nc = df.iloc[:, 10].values.reshape(-1, 1)
# Ncs = df.iloc[:, 9].values.reshape(-1, 1)
# Nk = df.iloc[:, 12:31].values  # 19个基团
# T = df.iloc[:, 31:41].values
# P_vp = df.iloc[:, 41:51].values
#
# # ==================== 清洗非法值 ====================
# valid_mask = np.isfinite(P_vp) & (P_vp > 0)
# valid_mask = valid_mask.all(axis=1)
#
# MW = MW[valid_mask]
# Nc = Nc[valid_mask]
# Ncs = Ncs[valid_mask]
# Nk = Nk[valid_mask]
# T = T[valid_mask]
# P_vp = P_vp[valid_mask]
#
# # 构造 y 和 X
# y = np.log(P_vp).flatten()
# X = np.hstack([
#     Nk.repeat(10, axis=0),
#     MW.repeat(10, axis=0),
#     Nc.repeat(10, axis=0),
#     Ncs.repeat(10, axis=0),
#     T.flatten().reshape(-1, 1)
# ])
#
# finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
# y = y[finite_mask]
# X = X[finite_mask, :]
#
# # ==================== 残差函数（适配19个基团） ====================
# def residuals(params, X, y):
#     Nk = X[:, :19]
#     MW = X[:, 19].reshape(-1, 1)
#     Nc = X[:, 20].reshape(-1, 1)
#     Ncs = X[:, 21].reshape(-1, 1)
#     T = np.clip(X[:, 22].reshape(-1, 1), 1e-6, None)
#
#     A1k = params[:19]
#     A2k = params[19:38]
#     s0, s1 = params[38], params[39]
#     alpha, f0, f1 = params[40], params[41], params[42]
#     B1k = params[43:62]
#     B2k = params[62:81]
#     beta = params[81]
#     C1k = params[82:101]
#     C2k = params[101:120]
#
#     term_A = np.sum(Nk * (A1k + MW * A2k), axis=1) + (s0 + Ncs.flatten() * s1) + alpha * (f0 + Nc.flatten() * f1)
#     term_B = np.sum(Nk * (B1k + MW * B2k), axis=1) + beta * (f0 + Nc.flatten() * f1)
#     term_C = np.sum(Nk * (C1k + MW * C2k), axis=1)
#
#     y_pred = term_A + term_B / T.flatten() + term_C * np.log(T.flatten())
#     return y - y_pred
#
# # ==================== 初始参数设置 ====================
# params_init = np.zeros(120)
#
# # A1k (N_0 到 N_18)
# params_init[:19] = [
#     13.65853808, 3.28418546, -659.6444719, 12.37483133, 4.81265536,
#     2.91551829, 97.31954706, 87.70370771, 95.98266611, 3.887261236,
#     27.43160868, 207.1319101, 47.22447225, 4687.002401, 3.637088127,
#     1523.380387, 3162.746842, 12062.07738, -8900.847866
# ]
#
# # A2k (N_0*M 到 N_18*M)
# params_init[19:38] = [
#     -0.015716978, 0.009075383, 11.48620132, -21.10261532, -0.011767963,
#     0.002675368, -0.109835685, -0.010236179, -0.171652319, 0.005908914,
#     10.467947, -5.994107293, -0.112649727, -17.43861742, 0.001820612,
#     -12.29192011, -5.831333421, -30.99113155, 26.51752291
# ]
#
# # s0 (鎴窛), s1 (Ncs), 伪*(f0 + Nc*f1)
# params_init[38:43] = [17.60905342, -0.000738906, 0.018089414, 0.0, 1.0]
#
# # B1k (N_0/T 到 N_18/T)
# params_init[43:62] = [
#     -1346.02436, -683.1104648, 67218.65971, -1384.512471, -884.3388538,
#     -1241.799972, -8807.96886, -9868.206835, -9972.171472, -764.4721254,
#     -2768.98, -22960.24319, -4496.012972, -507785.7608, -2221.349576,
#     -157397.6395, -350388.1207, -1307700.942, 957312.8216
# ]
#
# # B2k (N_0*M/T 到 N_18*M/T)
# params_init[62:81] = [
#     1.451298512, -0.736859315, -584.0308556, 3.123573902, 0.887401846,
#     0.122658761, 8.501979442, 0.898999866, 15.05201845, -0.396917177,
#     6.455487385, 318.8958283, 9.649044453, 2010.74563, 0.550921963,
#     1486.747823, 523.9930512, 3372.517851, -2848.526234
# ]
#
# # 尾*(f0 + Nc*f1)
# params_init[81] = -6.750229278
#
# # C1k (N_0*ln(T) 到 N_18*ln(T))
# params_init[82:101] = [
#     -1.846676986, -0.38538898, 85.74714557, -1.76399843, -0.569402352,
#     -0.250943128, -13.054703, -11.40790845, -12.58276815, -0.468789896,
#     -3.52337599, -26.44154671, -6.353423865, -606.0715674, -0.130106514,
#     -198.3318276, -407.7121286, -1560.004645, 1152.427648
# ]
#
# # C2k (N_0*M*ln(T) 到 N_18*M*ln(T))
# params_init[101:120] = [
#     0.002016846, -0.001221385, 7.344413404, 0.894155383, 0.001594902,
#     -0.000468558, 0.01491123, 0.001327088, 0.022906548, -0.0008161,
#     -0.43609896, -3.639773727, 0.015093667, 11.71908672, -0.000385519,
#     -3.450680198, -14.53413618, 9.827970088, 2.387602073
# ]
#
# # ==================== 拟合 ====================
# print("🚀 拟合中，请稍候...")
# result = least_squares(residuals, x0=params_init, args=(X, y), max_nfev=10000)
#
# # ==================== 输出参数 ====================
# param_names = (
#     [f"A1_{i}" for i in range(19)] +
#     [f"A2_{i}" for i in range(19)] +
#     ["s0 (鎴窛)", "s1 (Ncs)", "alpha", "f0", "f1"] +
#     [f"B1_{i}" for i in range(19)] +
#     [f"B2_{i}" for i in range(19)] +
#     ["beta"] +
#     [f"C1_{i}" for i in range(19)] +
#     [f"C2_{i}" for i in range(19)]
# )
#
# print("\n🔧 拟合后的参数：")
# for name, val in zip(param_names, result.x):
#     print(f"{name:15s}: {val:.8f}")
#
# # ==================== 模型评估 ====================
# y_pred = y - residuals(result.x, X, y)
# mse = mean_squared_error(y, y_pred)
# r2 = r2_score(y, y_pred)
# print(f"\n📈 模型性能:")
# print(f"R²  = {r2:.6f}")
# print(f"MSE = {mse:.6f}")
#
# # ==================== 实际蒸汽压评估（含 ARD） ====================
# P_true = np.exp(y)
# P_pred = np.exp(y_pred)
# mse_real = mean_squared_error(P_true, P_pred)
# r2_real = r2_score(P_true, P_pred)
# ard_real = np.mean(np.abs((P_pred - P_true) / P_true)) * 100
#
# print(f"\n📈 实际蒸汽压评估:")
# print(f"R² (P)  = {r2_real:.6f}")
# print(f"MSE (P) = {mse_real:.4f}")
# print(f"ARD (P) = {ard_real:.2f}%")
#
# # ==================== 保存结果 ====================
# compare_df = pd.DataFrame({
#     "Compound_ID": np.repeat(df.iloc[valid_mask, 0].values, 10)[finite_mask],
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
#
# output_filename = "Gani_lnP_prediction_results_19group_final.xlsx"
# compare_df.to_excel(output_filename, index=False)
# print(f"\n✅ 已保存预测结果为 {output_filename}")

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score

# ==================== 数据加载 ====================
file_path = "vp209.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 基本属性提取（适配19个基团）
MW = df.iloc[:, 4].values.reshape(-1, 1)
Nc = df.iloc[:, 10].values.reshape(-1, 1)
Ncs = df.iloc[:, 9].values.reshape(-1, 1)
Nk = df.iloc[:, 12:31].values  # 19个基团
T = df.iloc[:, 31:41].values
P_vp = df.iloc[:, 41:51].values

# ==================== 清洗非法值 ====================
valid_mask = np.isfinite(P_vp) & (P_vp > 0)
valid_mask = valid_mask.all(axis=1)

MW = MW[valid_mask]
Nc = Nc[valid_mask]
Ncs = Ncs[valid_mask]
Nk = Nk[valid_mask]
T = T[valid_mask]
P_vp = P_vp[valid_mask]

# 构造 y 和 X
y = np.log(P_vp).flatten()
X = np.hstack([
    Nk.repeat(10, axis=0),
    MW.repeat(10, axis=0),
    Nc.repeat(10, axis=0),
    Ncs.repeat(10, axis=0),
    T.flatten().reshape(-1, 1)
])

finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
y = y[finite_mask]
X = X[finite_mask, :]

# ==================== 残差函数（适配19个基团） ====================
def residuals(params, X, y):
    Nk = X[:, :19]
    MW = X[:, 19].reshape(-1, 1)
    Nc = X[:, 20].reshape(-1, 1)
    Ncs = X[:, 21].reshape(-1, 1)
    T = np.clip(X[:, 22].reshape(-1, 1), 1e-6, None)

    A1k = params[:19]
    A2k = params[19:38]
    s0, s1 = params[38], params[39]
    alpha, f0, f1 = params[40], params[41], params[42]
    B1k = params[43:62]
    B2k = params[62:81]
    beta = params[81]
    C1k = params[82:101]
    C2k = params[101:120]

    term_A = np.sum(Nk * (A1k + MW * A2k), axis=1) + (s0 + Ncs.flatten() * s1) + alpha * (f0 + Nc.flatten() * f1)
    term_B = np.sum(Nk * (B1k + MW * B2k), axis=1) + beta * (f0 + Nc.flatten() * f1)
    term_C = np.sum(Nk * (C1k + MW * C2k), axis=1)

    y_pred = term_A + term_B / T.flatten() + term_C * np.log(T.flatten())
    return y - y_pred

# ==================== 初始参数设置 ====================
params_init = np.zeros(120)

# A1k (N_0 到 N_18)
params_init[:19] = [
    13.65853808, 3.28418546, -659.6444719, 12.37483133, 4.81265536,
    2.91551829, 97.31954706, 87.70370771, 95.98266611, 3.887261236,
    27.43160868, 207.1319101, 47.22447225, 4687.002401, 3.637088127,
    1523.380387, 3162.746842, 12062.07738, -8900.847866
]

# A2k (N_0*M 到 N_18*M)
params_init[19:38] = [
    -0.015716978, 0.009075383, 11.48620132, -21.10261532, -0.011767963,
    0.002675368, -0.109835685, -0.010236179, -0.171652319, 0.005908914,
    10.467947, -5.994107293, -0.112649727, -17.43861742, 0.001820612,
    -12.29192011, -5.831333421, -30.99113155, 26.51752291
]

# s0 (鎴窛), s1 (Ncs), 伪*(f0 + Nc*f1)
params_init[38:43] = [17.60905342, -0.000738906, 0.018089414, 0.0, 1.0]

# B1k (N_0/T 到 N_18/T)
params_init[43:62] = [
    -1346.02436, -683.1104648, 67218.65971, -1384.512471, -884.3388538,
    -1241.799972, -8807.96886, -9868.206835, -9972.171472, -764.4721254,
    -2768.98, -22960.24319, -4496.012972, -507785.7608, -2221.349576,
    -157397.6395, -350388.1207, -1307700.942, 957312.8216
]

# B2k (N_0*M/T 到 N_18*M/T)
params_init[62:81] = [
    1.451298512, -0.736859315, -584.0308556, 3.123573902, 0.887401846,
    0.122658761, 8.501979442, 0.898999866, 15.05201845, -0.396917177,
    6.455487385, 318.8958283, 9.649044453, 2010.74563, 0.550921963,
    1486.747823, 523.9930512, 3372.517851, -2848.526234
]

# 尾*(f0 + Nc*f1)
params_init[81] = -6.750229278

# C1k (N_0*ln(T) 到 N_18*ln(T))
params_init[82:101] = [
    -1.846676986, -0.38538898, 85.74714557, -1.76399843, -0.569402352,
    -0.250943128, -13.054703, -11.40790845, -12.58276815, -0.468789896,
    -3.52337599, -26.44154671, -6.353423865, -606.0715674, -0.130106514,
    -198.3318276, -407.7121286, -1560.004645, 1152.427648
]

# C2k (N_0*M*ln(T) 到 N_18*M*ln(T))
params_init[101:120] = [
    0.002016846, -0.001221385, 7.344413404, 0.894155383, 0.001594902,
    -0.000468558, 0.01491123, 0.001327088, 0.022906548, -0.0008161,
    -0.43609896, -3.639773727, 0.015093667, 11.71908672, -0.000385519,
    -3.450680198, -14.53413618, 9.827970088, 2.387602073
]

# ==================== 拟合 ====================
print("🚀 拟合中，请稍候...")
result = least_squares(residuals, x0=params_init, args=(X, y), max_nfev=10000)

# ==================== 模型评估 ====================
y_pred = y - residuals(result.x, X, y)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"\n📈 模型性能:")
print(f"R²  = {r2:.6f}")
print(f"MSE = {mse:.6f}")

# ==================== 实际蒸汽压评估（含 ARD） ====================
P_true = np.exp(y)
P_pred = np.exp(y_pred)
mse_real = mean_squared_error(P_true, P_pred)
r2_real = r2_score(P_true, P_pred)
ard_real = np.mean(np.abs((P_pred - P_true) / P_true)) * 100

print(f"\n📈 实际蒸汽压评估:")
print(f"R² (P)  = {r2_real:.6f}")
print(f"MSE (P) = {mse_real:.4f}")
print(f"ARD (P) = {ard_real:.2f}%")

# ==================== 误差统计 ====================
relative_error = np.abs((P_pred - P_true) / P_true) * 100
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print(f"\n✅ 误差 ≤ 1% 的点数: {within_1pct}")
print(f"✅ 误差 ≤ 5% 的点数: {within_5pct}")
print(f"✅ 误差 ≤ 10% 的点数: {within_10pct}")

# ==================== 保存结果 ====================
compare_df = pd.DataFrame({
    "Compound_ID": np.repeat(df.iloc[valid_mask, 0].values, 10)[finite_mask],
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

output_filename = "Gani_lnP_prediction_results_19group_final.xlsx"
compare_df.to_excel(output_filename, index=False)
print(f"\n✅ 已保存预测结果为 {output_filename}")
