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

# 只对每个物质的特征进行一次预测
P_vp_Tb_pred = np.log(np.clip(model_tb.predict(Nk_poly), 1e-6, None))  # 直接对 Nk_poly 进行预测

# 计算 TB 残差
P_vp_Tb_true = np.log(P_vp[:, 0])  # 对应第一列的蒸汽压 P_vp_Tb_true
residual_Tb = P_vp_Tb_true - P_vp_Tb_pred  # 对数残差

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

# 直接把 TB 和 TC 残差添加为新特征
X_with_residual = np.hstack([X, residual_Tb.reshape(-1, 1)])

y = np.log(P_vp).flatten()

finite_mask = np.isfinite(y) & np.isfinite(X_with_residual).all(axis=1)
X_with_residual, y = X_with_residual[finite_mask], y[finite_mask]
T_valid = T.flatten()[finite_mask]

# ========= 模型训练 ========= #
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_with_residual, y)

# ========= 模型评估 ========= #
y_pred = model.predict(X_with_residual)
print("\n📈 蒸汽压模型对 ln(P) 拟合结果：")
print(f"R² (lnP) = {r2_score(y, y_pred):.6f}")
print(f"MSE (lnP) = {mean_squared_error(y, y_pred):.6f}")

P_true = np.exp(y)
P_pred = np.exp(y_pred)
mse_P = mean_squared_error(P_true, P_pred)
r2_P = r2_score(P_true, P_pred)
ard_P = np.mean_
