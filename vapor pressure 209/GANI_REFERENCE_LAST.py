# import numpy as np
# import pandas as pd
# from scipy.optimize import least_squares
# from sklearn.linear_model import HuberRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import GradientBoostingRegressor
#
# # ========= 1. Load Data =========
# df = pd.read_excel("vp209", sheet_name='Sheet1')
# Nk = df.iloc[:, 12:31].values  # 19个基团
# poly = PolynomialFeatures(degree=2, include_bias=False)
# Nk_poly = poly.fit_transform(Nk)
#
# # ========= 2. Boiling Point (Tb) Model =========
# Tb0 = 222.543
# Tb = df.iloc[:, 5].values
# model_tb = HuberRegressor(max_iter=5000).fit(Nk_poly, np.exp(Tb / Tb0))
# Tb_pred = Tb0 * np.log(np.clip(model_tb.predict(Nk_poly), 1e-6, None))
#
# # ========= 3. Critical Temperature (Tc) Model =========
# Tc_half = df['ASPEN Half Critical T'].values
# gb_model_tc = GradientBoostingRegressor(
#     n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
# )
# gb_model_tc.fit(Nk_poly, Tc_half)
# Tc_pred = gb_model_tc.predict(Nk_poly)
#
# # ========= 4. Critical Pressure (Pc) Model =========
# Pc_bar = df.iloc[:, 51].values
# MW = df.iloc[:, 4].values.flatten()
#
# def residual_pc(params, X, MW, Pc_true):
#     beta = params[:-1]
#     beta3 = params[-1]
#     y_pred = X @ beta
#     x_pred = y_pred + 0.108998
#     Pc_pred = 5.9827 + (1 / x_pred) ** 2 + beta3 * np.exp(1 / MW)
#     return Pc_pred - Pc_true
#
# params_init_pc = np.zeros(Nk_poly.shape[1] + 1)
# result_pc = least_squares(residual_pc, x0=params_init_pc, args=(Nk_poly, MW, Pc_bar), max_nfev=5000)
# x_fit = Nk_poly @ result_pc.x[:-1] + 0.108998
# Pc_pred = (5.9827 + (1 / x_fit)**2 + result_pc.x[-1] * np.exp(1 / MW)) * 1e5
#
# # ========= 5. Slope Feature Construction =========
# Pb = 101325
# slope_all = (np.log(Pc_pred) - np.log(Pb)) / (Tc_pred*2 - Tb_pred)
# slope_all = slope_all.reshape(-1, 1)
#
# # ========= 6. Prepare Dataset =========
# T = df.iloc[:, 31:41].values
# P_vp = df.iloc[:, 41:51].values
# MW = MW.reshape(-1, 1)
# Nc = df.iloc[:, 10].values.reshape(-1, 1)
# Ncs = df.iloc[:, 9].values.reshape(-1, 1)
#
# valid_mask = np.isfinite(P_vp) & (P_vp > 0)
# valid_mask = valid_mask.all(axis=1)
# Nk = Nk[valid_mask]
# T = T[valid_mask]
# P_vp = P_vp[valid_mask]
# MW = MW[valid_mask]
# Nc = Nc[valid_mask]
# Ncs = Ncs[valid_mask]
# slope_all = slope_all[valid_mask]
#
# # ========= 7. Build Training Set =========
# y = np.log(P_vp).flatten()
# X = np.hstack([
#     Nk.repeat(10, axis=0),
#     MW.repeat(10, axis=0),
#     Nc.repeat(10, axis=0),
#     Ncs.repeat(10, axis=0),
#     T.flatten().reshape(-1, 1),
#     slope_all.repeat(10, axis=0) * T.flatten().reshape(-1, 1)
# ])
#
# finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
# X = X[finite_mask]
# y = y[finite_mask]
#
# # ========= 8. Residual Function =========
# def residuals(params, X, y):
#     Nk = X[:, :12]
#     MW = X[:, 12].reshape(-1, 1)
#     Nc = X[:, 13].reshape(-1, 1)
#     Ncs = X[:, 14].reshape(-1, 1)
#     T = np.clip(X[:, 15].reshape(-1, 1), 1e-6, None)
#     slope_T = X[:, 16].reshape(-1, 1)
#
#     A1k = params[:12]
#     A2k = params[12:24]
#     s0, s1 = params[24], params[25]
#     alpha, f0, f1 = params[26], params[27], params[28]
#     B1k = params[29:41]
#     B2k = params[41:53]
#     beta = params[53]
#     C1k = params[54:66]
#     C2k = params[66:78]
#     gamma = params[78]
#
#     term_A = np.sum(Nk * (A1k + MW * A2k), axis=1) + s0 + s1 * Ncs.flatten() + alpha * (f0 + f1 * Nc.flatten())
#     term_B = np.sum(Nk * (B1k + MW * B2k), axis=1) + beta * (f0 + f1 * Nc.flatten())
#     term_C = np.sum(Nk * (C1k + MW * C2k), axis=1)
#
#     y_pred = term_A + term_B / T.flatten() + term_C * np.log(T.flatten()) + gamma * slope_T.flatten()
#     return y - y_pred
#
# # ========= 9. Initial Parameters =========
# params_init = np.array([
#     17.3683, 7.7102, -15.2653, 11.2186, 9.3464, -13.1484,
#     88.5031, 89.9539, 44.3037, 7.3199, 22.4376, -30.5308,
#     0.0273, -0.0085, 0.0922, 0.0067, -0.0218, 0.0423,
#     0.0988, 0.0580, 0.2499, -0.0100, 0.0131, 0.1849,
#     18.3710, 0.0484, 0.0484, 0.0, 1.0,
#     -1336.22, -1119.38, 770.74, -1231.84, -1157.32, -148.52,
#     -8148.63, -9960.01, -5144.31, -1049.42, -2463.69, 1541.48,
#     -0.6448, 0.6124, -9.6981, -0.7783, 1.6822, -2.0678,
#     -5.6822, -3.0283, -21.1782, 1.0146, -1.5566, -19.3962,
#     -40.7015,
#     -2.4928, -0.9628, 2.2021, -1.4518, -1.2337, 1.9651,
#     -11.8996, -11.8098, -5.7519, -0.9450, -2.9035, 4.4042,
#     -0.0043, 0.0012, -0.0120, -0.0008, 0.0029, -0.0062,
#     -0.0141, -0.0084, -0.0335, 0.0013, -0.0017, -0.0239,
#     1  # gamma
# ])
#
# # ========= 10. Fit Model =========
# print("\n🚀 Fitting model...")
# result = least_squares(residuals, x0=params_init, args=(X, y), max_nfev=8000)
#
# # ========= 11. Evaluation =========
# y_pred = y - residuals(result.x, X, y)
# P_true = np.exp(y)
# P_pred = np.exp(y_pred)
#
# r2_lnP = r2_score(y, y_pred)
# mse_lnP = mean_squared_error(y, y_pred)
# r2_P = r2_score(P_true, P_pred)
# mse_P = mean_squared_error(P_true, P_pred)
# ard_P = np.mean(np.abs((P_pred - P_true) / P_true)) * 100
#
# print("\n📊 ln(P) Evaluation:")
# print(f"R² = {r2_lnP:.6f}")
# print(f"MSE = {mse_lnP:.6f}")
#
# print("\n📊 P Evaluation:")
# print(f"R² = {r2_P:.6f}")
# print(f"MSE = {mse_P:.2f}")
# print(f"ARD = {ard_P:.2f}%")
#
# # ========= 12. Save Results =========
# compare_df = pd.DataFrame({
#     "ln(P)_true": y,
#     "ln(P)_pred": y_pred,
#     "P_true": P_true,
#     "P_pred": P_pred,
#     "Absolute_Error_P": np.abs(P_pred - P_true),
#     "Relative_Error_P (%)": 100 * np.abs((P_pred - P_true) / P_true)
# })
# compare_df.to_excel("Vapor_Pressure_Predictions_with_slopeT.xlsx", index=False)
# print("\n✅ Results saved to Vapor_Pressure_Predictions_with_slopeT.xlsx")
#
# import numpy as np
# import pandas as pd
# from scipy.optimize import least_squares
# from sklearn.linear_model import HuberRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import GradientBoostingRegressor
#
# # ========= 1. Load Data =========
# df = pd.read_excel("vp209.xlsx", sheet_name='Sheet1')
# Nk = df.iloc[:, 12:31].values  # 19个基团
# poly = PolynomialFeatures(degree=2, include_bias=False)
# Nk_poly = poly.fit_transform(Nk)
#
# # ========= 2. Boiling Point (Tb) Model =========
# Tb0 = 222.543
# Tb = df.iloc[:, 5].values
# model_tb = HuberRegressor(max_iter=10000).fit(Nk_poly, np.exp(Tb / Tb0))
# Tb_pred = Tb0 * np.log(np.clip(model_tb.predict(Nk_poly), 1e-6, None))
#
# # ========= 3. Critical Temperature (Tc) Model =========
# Tc_half = df['ASPEN Half Critical T'].values
# gb_model_tc = GradientBoostingRegressor(
#     n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
# )
# gb_model_tc.fit(Nk_poly, Tc_half)
# Tc_pred = gb_model_tc.predict(Nk_poly)
#
# # ========= 4. Critical Pressure (Pc) Model =========
# Pc_bar = df.iloc[:, 51].values
# MW = df.iloc[:, 4].values.flatten()
#
# def residual_pc(params, X, MW, Pc_true):
#     beta = params[:-1]
#     beta3 = params[-1]
#     y_pred = X @ beta
#     x_pred = y_pred + 0.108998
#     Pc_pred = 5.9827 + (1 / x_pred) ** 2 + beta3 * np.exp(1 / MW)
#     return Pc_pred - Pc_true
#
# params_init_pc = np.zeros(Nk_poly.shape[1] + 1)
# result_pc = least_squares(residual_pc, x0=params_init_pc, args=(Nk_poly, MW, Pc_bar), max_nfev=5000)
# x_fit = Nk_poly @ result_pc.x[:-1] + 0.108998
# Pc_pred = (5.9827 + (1 / x_fit)**2 + result_pc.x[-1] * np.exp(1 / MW)) * 1e5
#
# # ========= 5. Slope Feature Construction =========
# Pb = 101325
# slope_all = (np.log(Pc_pred) - np.log(Pb)) / (Tc_pred*2 - Tb_pred)
# slope_all = slope_all.reshape(-1, 1)
#
# # ========= 6. Prepare Dataset =========
# T = df.iloc[:, 31:41].values
# P_vp = df.iloc[:, 41:51].values
# MW = MW.reshape(-1, 1)
# Nc = df.iloc[:, 10].values.reshape(-1, 1)
# Ncs = df.iloc[:, 9].values.reshape(-1, 1)
#
# valid_mask = np.isfinite(P_vp) & (P_vp > 0)
# valid_mask = valid_mask.all(axis=1)
# Nk = Nk[valid_mask]
# T = T[valid_mask]
# P_vp = P_vp[valid_mask]
# MW = MW[valid_mask]
# Nc = Nc[valid_mask]
# Ncs = Ncs[valid_mask]
# slope_all = slope_all[valid_mask]
#
# # ========= 7. Build Training Set =========
# y = np.log(P_vp).flatten()
# X = np.hstack([
#     Nk.repeat(10, axis=0),
#     MW.repeat(10, axis=0),
#     Nc.repeat(10, axis=0),
#     Ncs.repeat(10, axis=0),
#     T.flatten().reshape(-1, 1),
#     slope_all.repeat(10, axis=0) * T.flatten().reshape(-1, 1)
# ])
#
# finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
# X = X[finite_mask]
# y = y[finite_mask]
#
# # ========= 8. Residual Function for 19 groups =========
# def residuals(params, X, y):
#     Nk = X[:, :19]
#     MW = X[:, 19].reshape(-1, 1)
#     Nc = X[:, 20].reshape(-1, 1)
#     Ncs = X[:, 21].reshape(-1, 1)
#     T = np.clip(X[:, 22].reshape(-1, 1), 1e-6, None)
#     slope_T = X[:, 23].reshape(-1, 1)
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
#     gamma = params[120]
#
#     term_A = np.sum(Nk * (A1k + MW * A2k), axis=1) + s0 + s1 * Ncs.flatten() + alpha * (f0 + f1 * Nc.flatten())
#     term_B = np.sum(Nk * (B1k + MW * B2k), axis=1) + beta * (f0 + f1 * Nc.flatten())
#     term_C = np.sum(Nk * (C1k + MW * C2k), axis=1)
#
#     y_pred = term_A + term_B / T.flatten() + term_C * np.log(T.flatten()) + gamma * slope_T.flatten()
#     return y - y_pred
#
# # ========= 9. Initial Parameters (19 groups × 6 sets + 6 structural + gamma) =========
# params_init = np.zeros(121)
#
# # 原来的12组初始值填入前12位，其余设为0或复制最后一项
# params_init[:12] = [17.3683, 7.7102, -15.2653, 11.2186, 9.3464, -13.1484,
#                     88.5031, 89.9539, 44.3037, 7.3199, 22.4376, -30.5308]
# params_init[19:31] = [0.0273, -0.0085, 0.0922, 0.0067, -0.0218, 0.0423,
#                       0.0988, 0.0580, 0.2499, -0.0100, 0.0131, 0.1849]
# params_init[38:43] = [18.3710, 0.0484, 0.0484, 0.0, 1.0]
# params_init[43:55] = [-1336.22, -1119.38, 770.74, -1231.84, -1157.32, -148.52,
#                       -8148.63, -9960.01, -5144.31, -1049.42, -2463.69, 1541.48]
# params_init[62:74] = [-0.6448, 0.6124, -9.6981, -0.7783, 1.6822, -2.0678,
#                       -5.6822, -3.0283, -21.1782, 1.0146, -1.5566, -19.3962]
# params_init[81] = -40.7015
# params_init[82:94] = [-2.4928, -0.9628, 2.2021, -1.4518, -1.2337, 1.9651,
#                       -11.8996, -11.8098, -5.7519, -0.9450, -2.9035, 4.4042]
# params_init[101:113] = [-0.0043, 0.0012, -0.0120, -0.0008, 0.0029, -0.0062,
#                         -0.0141, -0.0084, -0.0335, 0.0013, -0.0017, -0.0239]
# params_init[120] = 1  # gamma
#
# # ========= 10. Fit Model =========
# print("\n🚀 Fitting model...")
# result = least_squares(residuals, x0=params_init, args=(X, y), max_nfev=8000)
#
# # ========= 11. Evaluation =========
# y_pred = y - residuals(result.x, X, y)
# P_true = np.exp(y)
# P_pred = np.exp(y_pred)
#
# r2_lnP = r2_score(y, y_pred)
# mse_lnP = mean_squared_error(y, y_pred)
# r2_P = r2_score(P_true, P_pred)
# mse_P = mean_squared_error(P_true, P_pred)
# ard_P = np.mean(np.abs((P_pred - P_true) / P_true)) * 100
#
# print("\n📊 ln(P) Evaluation:")
# print(f"R² = {r2_lnP:.6f}")
# print(f"MSE = {mse_lnP:.6f}")
#
# print("\n📊 P Evaluation:")
# print(f"R² = {r2_P:.6f}")
# print(f"MSE = {mse_P:.2f}")
# print(f"ARD = {ard_P:.2f}%")
#
# # ========= 12. Save Results =========
# compare_df = pd.DataFrame({
#     "ln(P)_true": y,
#     "ln(P)_pred": y_pred,
#     "P_true": P_true,
#     "P_pred": P_pred,
#     "Absolute_Error_P": np.abs(P_pred - P_true),
#     "Relative_Error_P (%)": 100 * np.abs((P_pred - P_true) / P_true)
# })
# compare_df.to_excel("Vapor_Pressure_Predictions_with_slopeT_19group.xlsx", index=False)
# print("\n✅ Results saved to Vapor_Pressure_Predictions_with_slopeT_19group.xlsx")
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# ========= 1. 数据加载 =========
df = pd.read_excel("vp209.xlsx", sheet_name='Sheet1')
Nk = df.iloc[:, 12:31].values  # 19个基团
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk)

# ========= 2. 沸点(Tb)模型 =========
Tb0 = 222.543
Tb = df.iloc[:, 5].values
model_tb = HuberRegressor(max_iter=10000).fit(Nk_poly, np.exp(Tb / Tb0))
Tb_pred = Tb0 * np.log(np.clip(model_tb.predict(Nk_poly), 1e-6, None))

# ========= 3. 临界温度(Tc)模型 =========
Tc_half = df['ASPEN Half Critical T'].values
gb_model_tc = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
)
gb_model_tc.fit(Nk_poly, Tc_half)
Tc_pred = gb_model_tc.predict(Nk_poly)

# ========= 4. 临界压力(Pc)模型 =========
Pc_bar = df.iloc[:, 51].values
MW = df.iloc[:, 4].values.flatten()

def residual_pc(params, X, MW, Pc_true):
    beta = params[:-1]
    beta3 = params[-1]
    y_pred = X @ beta
    x_pred = y_pred + 0.108998
    Pc_pred = 5.9827 + (1 / x_pred) ** 2 + beta3 * np.exp(1 / MW)
    return Pc_pred - Pc_true

params_init_pc = np.zeros(Nk_poly.shape[1] + 1)
result_pc = least_squares(residual_pc, x0=params_init_pc, args=(Nk_poly, MW, Pc_bar), max_nfev=5000)
x_fit = Nk_poly @ result_pc.x[:-1] + 0.108998
Pc_pred = (5.9827 + (1 / x_fit)**2 + result_pc.x[-1] * np.exp(1 / MW)) * 1e5

# ========= 5. 斜率特征构建 =========
Pb = 101325
slope_all = (np.log(Pc_pred) - np.log(Pb)) / (Tc_pred*2 - Tb_pred)
slope_all = slope_all.reshape(-1, 1)

# ========= 6. 准备数据集 =========
T = df.iloc[:, 31:41].values
P_vp = df.iloc[:, 41:51].values
MW = MW.reshape(-1, 1)
Nc = df.iloc[:, 10].values.reshape(-1, 1)
Ncs = df.iloc[:, 9].values.reshape(-1, 1)

valid_mask = np.isfinite(P_vp) & (P_vp > 0)
valid_mask = valid_mask.all(axis=1)
Nk = Nk[valid_mask]
T = T[valid_mask]
P_vp = P_vp[valid_mask]
MW = MW[valid_mask]
Nc = Nc[valid_mask]
Ncs = Ncs[valid_mask]
slope_all = slope_all[valid_mask]

# ========= 7. 构建训练集 =========
y = np.log(P_vp).flatten()
X = np.hstack([
    Nk.repeat(10, axis=0),
    MW.repeat(10, axis=0),
    Nc.repeat(10, axis=0),
    Ncs.repeat(10, axis=0),
    T.flatten().reshape(-1, 1),
    slope_all.repeat(10, axis=0) * T.flatten().reshape(-1, 1)
])

finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
X = X[finite_mask]
y = y[finite_mask]

# ========= 8. 残差函数（19基团+斜率项） =========
def residuals(params, X, y):
    Nk = X[:, :19]
    MW = X[:, 19].reshape(-1, 1)
    Nc = X[:, 20].reshape(-1, 1)
    Ncs = X[:, 21].reshape(-1, 1)
    T = np.clip(X[:, 22].reshape(-1, 1), 1e-6, None)
    slope_T = X[:, 23].reshape(-1, 1)

    A1k = params[:19]
    A2k = params[19:38]
    s0, s1 = params[38], params[39]
    alpha, f0, f1 = params[40], params[41], params[42]
    B1k = params[43:62]
    B2k = params[62:81]
    beta = params[81]
    C1k = params[82:101]
    C2k = params[101:120]
    gamma = params[120]

    term_A = np.sum(Nk * (A1k + MW * A2k), axis=1) + s0 + s1 * Ncs.flatten() + alpha * (f0 + f1 * Nc.flatten())
    term_B = np.sum(Nk * (B1k + MW * B2k), axis=1) + beta * (f0 + f1 * Nc.flatten())
    term_C = np.sum(Nk * (C1k + MW * C2k), axis=1)

    y_pred = term_A + term_B / T.flatten() + term_C * np.log(T.flatten()) + gamma * slope_T.flatten()
    return y - y_pred

# ========= 9. 初始参数设置 =========
params_init = np.zeros(121)

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

# 斜率系数gamma
params_init[120] = 1.0

# ========= 10. 模型拟合 =========
print("\n🚀 模型拟合中，请稍候...")
result = least_squares(residuals, x0=params_init, args=(X, y), max_nfev=10000)

# ========= 11. 模型评估 =========
y_pred = y - residuals(result.x, X, y)
P_true = np.exp(y)
P_pred = np.exp(y_pred)

r2_lnP = r2_score(y, y_pred)
mse_lnP = mean_squared_error(y, y_pred)
r2_P = r2_score(P_true, P_pred)
mse_P = mean_squared_error(P_true, P_pred)
ard_P = np.mean(np.abs((P_pred - P_true) / P_true)) * 100

print("\n📊 ln(P) 评估结果:")
print(f"R² = {r2_lnP:.6f}")
print(f"MSE = {mse_lnP:.6f}")

print("\n📊 P 评估结果:")
print(f"R² = {r2_P:.6f}")
print(f"MSE = {mse_P:.2f}")
print(f"平均相对偏差(ARD) = {ard_P:.2f}%")
relative_error = np.abs((P_pred - P_true) / P_true) * 100

# 统计误差小于 1%、5%、10% 的点数
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print(f"\n✅ 误差 ≤ 1% 的点数: {within_1pct}")
print(f"✅ 误差 ≤ 5% 的点数: {within_5pct}")
print(f"✅ 误差 ≤ 10% 的点数: {within_10pct}")
# ========= 12. 参数输出 =========
param_names = (
    [f"A1_{i}" for i in range(19)] +
    [f"A2_{i}" for i in range(19)] +
    ["s0 (鎴窛)", "s1 (Ncs)", "alpha", "f0", "f1"] +
    [f"B1_{i}" for i in range(19)] +
    [f"B2_{i}" for i in range(19)] +
    ["beta"] +
    [f"C1_{i}" for i in range(19)] +
    [f"C2_{i}" for i in range(19)] +
    ["gamma (斜率系数)"]
)

print("\n🔧 拟合后的参数：")
for name, val in zip(param_names, result.x):
    print(f"{name:20s}: {val:.8f}")

# ========= 13. 结果保存 =========
compare_df = pd.DataFrame({
    "ln(P)_true": y,
    "ln(P)_pred": y_pred,
    "P_true": P_true,
    "P_pred": P_pred,
    "Absolute_Error_P": np.abs(P_pred - P_true),
    "Relative_Error_P (%)": 100 * np.abs((P_pred - P_true) / P_true),
    "Temperature_K": X[:, 22],
    "Molecular_Weight": X[:, 19],
    "Carbon_Number": X[:, 20],
    "Slope_Term": X[:, 23]
})

output_filename = "Vapor_Pressure_Predictions_Final_19group.xlsx"
compare_df.to_excel(output_filename, index=False)
print(f"\n✅ 结果已保存至 {output_filename}")