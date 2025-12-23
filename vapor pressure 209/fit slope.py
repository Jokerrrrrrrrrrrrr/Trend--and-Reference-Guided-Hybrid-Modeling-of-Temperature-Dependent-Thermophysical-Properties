import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score

# ========== 读取数据 ========== #
df = pd.read_excel("vp209.xlsx", sheet_name='Sheet1')

# ========== 定义列 ========== #
group_cols = df.columns[12:31]   # 第13~31列：基团
temp_cols = df.columns[31:41]    # 第32~41列：温度
v_cols = df.columns[41:51]       # 第42~51列：蒸汽压

# ========== 数据预处理 ========== #
# 确保数值列正确转换
for col in temp_cols.tolist() + v_cols.tolist():
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 基团数据
Nk = df.iloc[:, 12:31].values  # 19个基团
T = df.iloc[:, 31:41].values
P_vp = df.iloc[:, 41:51].values

# ========== 创建有效掩码 ========== #
# 使用你提供的valid_mask
valid_mask = np.isfinite(P_vp) & (P_vp > 0)
valid_mask = valid_mask.all(axis=1)

# ========== 构建 Nk_poly ========== #
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk)

# ========== Tb 模型 ========== #
Tb0 = 222.543
Tb_raw = df.iloc[:, 5].values

# 使用valid_mask筛选有效数据
model_tb = HuberRegressor(max_iter=10000).fit(Nk_poly[valid_mask], np.exp(Tb_raw[valid_mask] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_tb.predict(Nk_poly), 1e-6, None))

# ========== Pc 模型 ========== #
Pc_bar = df.iloc[:, 51].values[valid_mask]
MW = df.iloc[:, 4].values.reshape(-1, 1)  # 假设第5列是分子量
MW_flat = MW[valid_mask].flatten()

Pc_poly = poly.fit_transform(Nk[valid_mask])

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
Pc_pred_all = (5.9827 + (1 / x_fit) ** 2 + result_pc.x[-1] * np.exp(1 / MW_flat)) * 1e5  # Pa

# ========== 蒸汽压主模型 ========== #
# 使用原始19个基团
G = Nk  # (n, 19) 原始基团数据
X_rows, y_rows = [], []
temp_eval = []  # 保存温度点信息用于评估

# 构建训练集 - 只使用有效数据
for i in np.where(valid_mask)[0]:  # 只遍历有效物质
    for j, (tcol, vcol) in enumerate(zip(temp_cols, v_cols)):
        Tj = df.at[i, tcol]  # 温度值
        Vj = df.at[i, vcol]  # 蒸汽压值

        # 跳过NaN值
        if np.isnan(Tj) or np.isnan(Vj) or Vj <= 0:
            continue

        Tb_i = Tb_pred_all[i]  # 物质i的参考温度
        # 注意：这里需要参考蒸汽压值，但原代码中没有提供
        # 假设我们使用Antoine方程或其他方法计算参考蒸汽压
        # 这里使用一个简化的参考值
        V_ref = 101325  # 标准大气压，Pa

        # 特征：(T - T_ref) × G (使用原始19个基团)
        Xj = (Tj - Tb_i) * G[i]  # 形状: (19,)

        # 目标：ln(V) - ln(V_ref)
        # 对蒸汽压取对数，因为蒸汽压通常用对数形式建模
        yj = np.log(Vj) - np.log(V_ref)

        X_rows.append(Xj)
        y_rows.append(yj)
        temp_eval.append((tcol, vcol, i, j))

X_A = np.array(X_rows)  # (n_samples, 19)
y_A = np.array(y_rows)  # (n_samples,)

# 训练 A_k 系数模型
A_solver = HuberRegressor(fit_intercept=False, max_iter=5000)
A_solver.fit(X_A, y_A)
A_vec = A_solver.coef_  # 长度19，对应19个基团

# ========== 生成蒸汽压预测 ========== #
V_pred = pd.DataFrame(index=df.index, columns=v_cols, dtype=float)

for i in range(len(df)):  # 遍历所有物质
    Tb_i = Tb_pred_all[i]  # 物质i的参考温度
    V_ref = 101325  # 参考蒸汽压值，Pa

    for j, (tcol, vcol) in enumerate(zip(temp_cols, v_cols)):
        Tj = df.at[i, tcol]  # 温度值

        if np.isnan(Tj):
            V_pred.at[i, vcol] = np.nan
            continue

        # 特征：(T - T_ref) × G (使用原始19个基团)
        Xj = (Tj - Tb_i) * G[i]

        # 预测：ln(V_ref) + A_k × (T - T_ref) × G，然后取指数
        ln_V_pred_j = np.log(V_ref) + Xj @ A_vec
        V_pred_j = np.exp(ln_V_pred_j)
        V_pred.at[i, vcol] = V_pred_j

# ========== 评估模型性能 ========== #
# 只使用有效数据进行评估
y_true_all, y_pred_all = [], []
for vcol in v_cols:
    # 只考虑有效掩码为True的数据
    m = valid_mask & (~df[vcol].isna()) & (~V_pred[vcol].isna())
    if m.any():
        y_true_all.append(df.loc[m, vcol].to_numpy())
        y_pred_all.append(V_pred.loc[m, vcol].to_numpy())

if y_true_all and y_pred_all:
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    # 使用对数尺度评估，因为蒸汽压通常用对数形式
    mse_v = mean_squared_error(y_true_all, y_pred_all)
    r2_v = r2_score(y_true_all, y_pred_all)
    print(f"MSE (on ln(Vapor Pressure)): {mse_v:.6f}")
    print(f"R2  (on ln(Vapor Pressure)): {r2_v:.6f}")
else:
    print("没有有效数据用于评估")

# ========== 分温度点评估 ========== #
print("\n分温度点评估:")
for tcol, vcol in zip(temp_cols, v_cols):
    # 只考虑有效掩码为True的数据
    m = valid_mask & (~df[tcol].isna()) & (~df[vcol].isna()) & (~V_pred[vcol].isna())
    if m.any():
        v_true = df.loc[m, vcol].to_numpy()
        v_pred = V_pred.loc[m, vcol].to_numpy()
        # 使用对数尺度
        mse_temp = mean_squared_error(np.log(v_true), np.log(v_pred))
        r2_temp = r2_score(np.log(v_true), np.log(v_pred))
        print(f"  {tcol}: MSE = {mse_temp:.6f}, R2 = {r2_temp:.6f}")

# ========== 保存结果 ========== #
id_col = df.columns[0]  # 物质ID/名称所在列
out_path = "vapor_pressure_actual_vs_pred_long.xlsx"

rows = []
for idx, _ in df.iterrows():
    ID = df.at[idx, id_col]
    for j, (tcol, vcol) in enumerate(zip(temp_cols, v_cols), start=1):
        T_val = df.at[idx, tcol]
        V_act = df.at[idx, vcol]
        V_prd = V_pred.at[idx, vcol]
        err = (V_prd - V_act) if (pd.notna(V_prd) and pd.notna(V_act)) else np.nan

        rows.append({
            id_col: ID,
            "temp_index": j,
            "temp_col": tcol,
            "T": T_val,
            "Vapor_Pressure_actual": V_act,
            "Vapor_Pressure_pred": V_prd,
            "error = pred - actual": err,
            "T_ref": Tb_pred_all[idx],
            "Pc_pred": Pc_pred_all[idx] if idx < len(Pc_pred_all) else np.nan,
            "is_valid": valid_mask[idx] if idx < len(valid_mask) else False
        })

long_compare = pd.DataFrame(rows).sort_values([id_col, "temp_index"])

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    long_compare.to_excel(writer, sheet_name="compare_long", index=False)

print(f"\n✅ 结果已保存到: {out_path}")
print(f"有效数据点数量: {np.sum(valid_mask)}")