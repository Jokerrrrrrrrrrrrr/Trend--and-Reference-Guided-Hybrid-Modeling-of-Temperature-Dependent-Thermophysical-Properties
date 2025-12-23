import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# ========== 读取数据 ========== #
df = pd.read_excel("vp209.xlsx", sheet_name='Sheet1')

# ========== 定义列 ========== #
group_cols = df.columns[12:31]  # 第13~31列：基团
temp_cols = df.columns[31:41]  # 第32~41列：温度
v_cols = df.columns[41:51]  # 第42~51列：蒸汽压

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

# ========== 生成基准蒸汽压预测 ========== #
V_pred_baseline = pd.DataFrame(index=df.index, columns=v_cols, dtype=float)

for i in range(len(df)):  # 遍历所有物质
    Tb_i = Tb_pred_all[i]  # 物质i的参考温度
    V_ref = 101325  # 参考蒸汽压值，Pa

    for j, (tcol, vcol) in enumerate(zip(temp_cols, v_cols)):
        Tj = df.at[i, tcol]  # 温度值

        if np.isnan(Tj):
            V_pred_baseline.at[i, vcol] = np.nan
            continue

        # 特征：(T - T_ref) × G (使用原始19个基团)
        Xj = (Tj - Tb_i) * G[i]

        # 预测：ln(V_ref) + A_k × (T - T_ref) × G，然后取指数
        ln_V_pred_j = np.log(V_ref) + Xj @ A_vec
        V_pred_j = np.exp(ln_V_pred_j)
        V_pred_baseline.at[i, vcol] = V_pred_j

# ========== 残差机器学习模型 ========== #
print("训练残差机器学习模型...")

# 构建残差训练数据集
residual_features = []
residual_targets = []
sample_info = []  # 保存样本信息用于追踪

for tcol, vcol in zip(temp_cols, v_cols):
    Tj = df[tcol].to_numpy()
    Vj = df[vcol].to_numpy()
    # 只使用有效数据
    msk = valid_mask & (~np.isnan(Tj)) & (~np.isnan(Vj)) & (Vj > 0)

    for i in np.where(msk)[0]:
        # 基础特征：基团组成
        base_features = list(G[i])

        # 温度相关特征
        temp_features = [
            Tj[i],  # 绝对温度
            Tj[i] - Tb_pred_all[i],  # 相对于参考温度的差值
            Tj[i] / Tb_pred_all[i] if Tb_pred_all[i] > 0 else 0,  # 相对温度
            np.log(Tj[i]) if Tj[i] > 0 else 0,  # 温度对数
        ]

        # 基准预测值作为特征（对数尺度）
        baseline_pred = V_pred_baseline.at[i, vcol]
        baseline_features = [np.log(baseline_pred) if baseline_pred > 0 else 0]

        # 参考值特征
        ref_features = [
            Tb_pred_all[i],  # 参考温度
            np.log(101325),  # 参考蒸汽压的对数
            Pc_pred_all[i] if i < len(Pc_pred_all) else 0,  # 临界压力
        ]

        # 分子量特征
        mw_features = [MW[i][0] if i < len(MW) else 0]

        # 组合所有特征
        all_features = base_features + temp_features + baseline_features + ref_features + mw_features
        residual_features.append(all_features)

        # 残差目标：实际值的对数 - 基准预测的对数
        residual = np.log(Vj[i]) - np.log(baseline_pred)
        residual_targets.append(residual)

        sample_info.append((i, tcol, vcol))

residual_features = np.array(residual_features)
residual_targets = np.array(residual_targets)

print(f"残差训练集形状: {residual_features.shape}")
print(f"残差目标形状: {residual_targets.shape}")

# 标准化特征
scaler_residual = StandardScaler()
residual_features_scaled = scaler_residual.fit_transform(residual_features)

# 训练残差模型（使用梯度提升回归）
residual_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# 交叉验证评估残差模型
cv_scores = cross_val_score(residual_model, residual_features_scaled, residual_targets,
                            cv=5, scoring='r2')
print(f"残差模型交叉验证 R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 训练最终残差模型
residual_model.fit(residual_features_scaled, residual_targets)

# ========== 生成最终预测（基准 + 残差修正） ========== #
V_pred_final = pd.DataFrame(index=df.index, columns=v_cols, dtype=float)

for tcol, vcol in zip(temp_cols, v_cols):
    Tj = df[tcol].to_numpy()

    # 为所有样本构建特征
    features_list = []
    valid_indices = []

    for i in range(len(df)):
        if np.isnan(Tj[i]) or (i < len(valid_mask) and not valid_mask[i]):
            continue

        # 构建与训练时相同的特征
        base_features = list(G[i])
        temp_features = [
            Tj[i],
            Tj[i] - Tb_pred_all[i],
            Tj[i] / Tb_pred_all[i] if Tb_pred_all[i] > 0 else 0,
            np.log(Tj[i]) if Tj[i] > 0 else 0,
        ]
        baseline_pred = V_pred_baseline.at[i, vcol]
        baseline_features = [np.log(baseline_pred) if baseline_pred > 0 else 0]
        ref_features = [
            Tb_pred_all[i],
            np.log(101325),
            Pc_pred_all[i] if i < len(Pc_pred_all) else 0,
        ]
        mw_features = [MW[i][0] if i < len(MW) else 0]

        all_features = base_features + temp_features + baseline_features + ref_features + mw_features
        features_list.append(all_features)
        valid_indices.append(i)

    if features_list:
        features_array = np.array(features_list)
        features_scaled = scaler_residual.transform(features_array)

        # 预测残差
        residual_pred = residual_model.predict(features_scaled)

        # 最终预测：ln(V_final) = ln(V_baseline) + 残差，然后取指数
        for idx, residual_val in zip(valid_indices, residual_pred):
            baseline_pred = V_pred_baseline.at[idx, vcol]
            if baseline_pred > 0:
                ln_V_final = np.log(baseline_pred) + residual_val
                V_final = np.exp(ln_V_final)
                V_pred_final.at[idx, vcol] = V_final
            else:
                V_pred_final.at[idx, vcol] = np.nan

    # 对于无效温度点，保持NaN
    V_pred_final[vcol] = np.where(np.isnan(Tj), np.nan, V_pred_final[vcol])

# ========== 评估模型性能 ========== #
# 只使用有效数据进行评估
print("\n=== 基准模型性能 ===")
y_true_all, y_pred_baseline = [], []
for vcol in v_cols:
    # 只考虑有效掩码为True的数据
    m = valid_mask & (~df[vcol].isna()) & (~V_pred_baseline[vcol].isna()) & (df[vcol] > 0) & (V_pred_baseline[vcol] > 0)
    if m.any():
        y_true_all.append(df.loc[m, vcol].to_numpy())
        y_pred_baseline.append(V_pred_baseline.loc[m, vcol].to_numpy())

if y_true_all and y_pred_baseline:
    y_true_all = np.concatenate(y_true_all)
    y_pred_baseline = np.concatenate(y_pred_baseline)

    # 使用对数尺度评估，因为蒸汽压通常用对数形式
    mse_baseline = mean_squared_error(np.log(y_true_all), np.log(y_pred_baseline))
    r2_baseline = r2_score(np.log(y_true_all), np.log(y_pred_baseline))
    print(f"基准模型 - MSE (on ln(Vapor Pressure)): {mse_baseline:.6f}")
    print(f"基准模型 - R2  (on ln(Vapor Pressure)): {r2_baseline:.6f}")
else:
    print("没有有效数据用于基准模型评估")

print("\n=== 最终模型性能（基准 + 残差修正）===")
y_true_all, y_pred_final = [], []
for vcol in v_cols:
    # 只考虑有效掩码为True的数据
    m = valid_mask & (~df[vcol].isna()) & (~V_pred_final[vcol].isna()) & (df[vcol] > 0) & (V_pred_final[vcol] > 0)
    if m.any():
        y_true_all.append(df.loc[m, vcol].to_numpy())
        y_pred_final.append(V_pred_final.loc[m, vcol].to_numpy())

if y_true_all and y_pred_final:
    y_true_all = np.concatenate(y_true_all)
    y_pred_final = np.concatenate(y_pred_final)

    # 使用对数尺度评估，因为蒸汽压通常用对数形式
    mse_final = mean_squared_error(y_true_all, y_pred_final)
    r2_final = r2_score(y_true_all, y_pred_final)
    print(f"最终模型 - MSE (on ln(Vapor Pressure)): {mse_final:.6f}")
    print(f"最终模型 - R2  (on ln(Vapor Pressure)): {r2_final:.6f}")

    # 改进程度
    if 'r2_baseline' in locals():
        improvement = r2_final - r2_baseline
        print(f"\n改进程度: R² 提升了 {improvement:.4f} ({improvement / r2_baseline * 100:.2f}%)")
else:
    print("没有有效数据用于最终模型评估")

# ========== 分温度点评估 ========== #
print("\n分温度点评估:")
for tcol, vcol in zip(temp_cols, v_cols):
    # 只考虑有效掩码为True的数据
    m = valid_mask & (~df[tcol].isna()) & (~df[vcol].isna()) & (~V_pred_final[vcol].isna()) & (df[vcol] > 0) & (
                V_pred_final[vcol] > 0)
    if m.any():
        v_true = df.loc[m, vcol].to_numpy()
        v_pred = V_pred_final.loc[m, vcol].to_numpy()
        # 使用对数尺度
        mse_temp = mean_squared_error(v_true, v_pred)
        r2_temp = r2_score(v_true,v_pred)
        print(f"  {tcol}: MSE = {mse_temp:.6f}, R2 = {r2_temp:.6f}")

# ========== 保存结果 ========== #
id_col = df.columns[0]  # 物质ID/名称所在列
out_path = "vapor_pressure_actual_vs_pred_with_residual_correction.xlsx"

rows = []
for idx, _ in df.iterrows():
    ID = df.at[idx, id_col]
    for j, (tcol, vcol) in enumerate(zip(temp_cols, v_cols), start=1):
        T_val = df.at[idx, tcol]
        V_act = df.at[idx, vcol]
        V_base = V_pred_baseline.at[idx, vcol] if pd.notna(V_pred_baseline.at[idx, vcol]) else np.nan
        V_final = V_pred_final.at[idx, vcol] if pd.notna(V_pred_final.at[idx, vcol]) else np.nan

        # 计算误差（对数尺度）
        if pd.notna(V_act) and pd.notna(V_base) and V_act > 0 and V_base > 0:
            err_base_log = np.log(V_base) - np.log(V_act)
        else:
            err_base_log = np.nan

        if pd.notna(V_act) and pd.notna(V_final) and V_act > 0 and V_final > 0:
            err_final_log = np.log(V_final) - np.log(V_act)
        else:
            err_final_log = np.nan

        residual_correction = (np.log(V_final) - np.log(V_base)) if (
                    pd.notna(V_final) and pd.notna(V_base) and V_final > 0 and V_base > 0) else np.nan

        rows.append({
            id_col: ID,
            "temp_index": j,
            "temp_col": tcol,
            "T": T_val,
            "Vapor_Pressure_actual": V_act,
            "Vapor_Pressure_baseline": V_base,
            "Vapor_Pressure_final": V_final,
            "error_baseline_log": err_base_log,
            "error_final_log": err_final_log,
            "residual_correction_log": residual_correction,
            "T_ref": Tb_pred_all[idx],
            "Pc_pred": Pc_pred_all[idx] if idx < len(Pc_pred_all) else np.nan,
            "is_valid": valid_mask[idx] if idx < len(valid_mask) else False
        })

long_compare = pd.DataFrame(rows).sort_values([id_col, "temp_index"])

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    long_compare.to_excel(writer, sheet_name="compare_long", index=False)

print(f"\n✅ 结果已保存到: {out_path}")
print(f"有效数据点数量: {np.sum(valid_mask)}")
# —— 简洁相对误差统计（最终）——
relative_error_final = np.abs((y_pred_final - y_true_all) / y_true_all) * 100
within_1pct_final  = np.sum(relative_error_final <= 1)
within_5pct_final  = np.sum(relative_error_final <= 5)
within_10pct_final = np.sum(relative_error_final <= 10)
ard_final = np.mean(relative_error_final)  # 平均相对偏差（%）

print("\n📊 总模型评估（基准 + 残差修正）：")
print(f"R²  = {r2_final:.4f}")
print(f"MSE = {mse_final:.6f}")
print(f"ARD = {ard_final:.2f}%")
print(f"✅ 误差 ≤ 1% 的数据点数量: {within_1pct_final}")
print(f"✅ 误差 ≤ 5% 的数据点数量: {within_5pct_final}")
print(f"✅ 误差 ≤ 10% 的数据点数量: {within_10pct_final}")
