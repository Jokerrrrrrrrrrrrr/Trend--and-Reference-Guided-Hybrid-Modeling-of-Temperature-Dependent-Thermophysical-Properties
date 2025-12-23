import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# ==== 1. 读取数据 ====
df = pd.read_excel("pure component isentropic exponent 207.xlsx", sheet_name="Sheet1")

# ==== 2. 定义列 ====
group_cols = df.columns[12:31]  # 第13~31列：基团
temp_cols = df.columns[31:41]  # 第32~41列：温度
v_cols = df.columns[41:51]  # 第42~51列：等熵指数

# ==== 3. 数据预处理 ====
# 确保数值列正确转换
for col in temp_cols.tolist() + v_cols.tolist():
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 基团数据
Nk_all = df.iloc[:, 12:31].apply(pd.to_numeric, errors='coerce')

# ==== 4. 等熵指数模型（Tb） ====
df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
X_Tb = df_Tb.drop(columns=["ASPEN isentropic exponent at boiling Temperature(bar)"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["ASPEN isentropic exponent at boiling Temperature(bar)"])
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 5. Tb 模型预测 ====
Tb_raw = df.iloc[:, 5].values
Tb0 = 222.543
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)

mask_tb = ~np.isnan(Tb_raw)
model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))

# ==== 6. A_k 系数训练 ====
# 使用原始19个基团
G = Nk_all.values  # (n, 19) 原始基团数据
X_rows, y_rows = [], []
temp_eval = []  # 保存温度点信息用于评估

# 构建训练集
for i in range(len(df)):  # 遍历所有物质
    for j, (tcol, vcol) in enumerate(zip(temp_cols, v_cols)):
        Tj = df.at[i, tcol]  # 温度值
        Vj = df.at[i, vcol]  # 等熵指数值

        # 跳过NaN值
        if np.isnan(Tj) or np.isnan(Vj):
            continue

        Tb_i = Tb_pred_all[i]  # 物质i的参考温度
        HVap_Tb_i = HVap_Tb_all[i]  # 物质i的参考等熵指数值

        # 特征：(T - T_ref) × G (使用原始19个基团)
        Xj = (Tj - Tb_i) * G[i]  # 形状: (19,)

        # 目标：V - V_ref
        yj = Vj - HVap_Tb_i

        X_rows.append(Xj)
        y_rows.append(yj)
        temp_eval.append((tcol, vcol, i, j))

X_A = np.array(X_rows)  # (n_samples, 19)
y_A = np.array(y_rows)  # (n_samples,)

# 训练 A_k 系数模型
A_solver = HuberRegressor(fit_intercept=False, max_iter=5000)
A_solver.fit(X_A, y_A)
A_vec = A_solver.coef_  # 长度19，对应19个基团

# ==== 7. 生成基准等熵指数预测 ====
V_pred_baseline = pd.DataFrame(index=df.index, columns=v_cols, dtype=float)

for i in range(len(df)):  # 遍历所有物质
    Tb_i = Tb_pred_all[i]  # 物质i的参考温度
    HVap_Tb_i = HVap_Tb_all[i]  # 物质i的参考等熵指数值

    for j, (tcol, vcol) in enumerate(zip(temp_cols, v_cols)):
        Tj = df.at[i, tcol]  # 温度值

        if np.isnan(Tj):
            V_pred_baseline.at[i, vcol] = np.nan
            continue

        # 特征：(T - T_ref) × G (使用原始19个基团)
        Xj = (Tj - Tb_i) * G[i]

        # 预测：V_ref + A_k × (T - T_ref) × G
        V_pred_j = HVap_Tb_i + Xj @ A_vec
        V_pred_baseline.at[i, vcol] = V_pred_j

# ==== 8. 残差机器学习模型 ====
print("训练残差机器学习模型...")

# 构建残差训练数据集
residual_features = []
residual_targets = []
sample_info = []  # 保存样本信息用于追踪

for tcol, vcol in zip(temp_cols, v_cols):
    Tj = df[tcol].to_numpy()
    Vj = df[vcol].to_numpy()
    msk = (~np.isnan(Tj)) & (~np.isnan(Vj))

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

        # 基准预测值作为特征
        baseline_pred = V_pred_baseline.at[i, vcol]
        baseline_features = [baseline_pred]

        # 参考值特征
        ref_features = [
            Tb_pred_all[i],  # 参考温度
            HVap_Tb_all[i],  # 参考等熵指数值
        ]

        # 组合所有特征
        all_features = base_features + temp_features + baseline_features + ref_features
        residual_features.append(all_features)

        # 残差目标：实际值 - 基准预测值
        residual = Vj[i] - baseline_pred
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

# ==== 9. 生成最终预测（基准 + 残差修正） ====
V_pred_final = pd.DataFrame(index=df.index, columns=v_cols, dtype=float)

for tcol, vcol in zip(temp_cols, v_cols):
    Tj = df[tcol].to_numpy()

    # 为所有样本构建特征
    features_list = []
    valid_indices = []

    for i in range(len(df)):
        if np.isnan(Tj[i]):
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
        baseline_features = [baseline_pred]
        ref_features = [Tb_pred_all[i], HVap_Tb_all[i]]

        all_features = base_features + temp_features + baseline_features + ref_features
        features_list.append(all_features)
        valid_indices.append(i)

    if features_list:
        features_array = np.array(features_list)
        features_scaled = scaler_residual.transform(features_array)

        # 预测残差
        residual_pred = residual_model.predict(features_scaled)

        # 最终预测 = 基准预测 + 残差修正
        for idx, residual_val in zip(valid_indices, residual_pred):
            final_pred = V_pred_baseline.at[idx, vcol] + residual_val
            V_pred_final.at[idx, vcol] = final_pred

    # 对于无效温度点，保持NaN
    V_pred_final[vcol] = np.where(np.isnan(Tj), np.nan, V_pred_final[vcol])

# ==== 10. 评估模型性能 ====
# 基准模型评估
print("\n=== 基准模型性能 ===")
y_true_all, y_pred_baseline = [], []
for vcol in v_cols:
    m = (~df[vcol].isna()) & (~V_pred_baseline[vcol].isna())
    if m.any():
        y_true_all.append(df.loc[m, vcol].to_numpy())
        y_pred_baseline.append(V_pred_baseline.loc[m, vcol].to_numpy())

y_true_all = np.concatenate(y_true_all)
y_pred_baseline = np.concatenate(y_pred_baseline)

mse_baseline = mean_squared_error(y_true_all, y_pred_baseline)
r2_baseline = r2_score(y_true_all, y_pred_baseline)
print(f"基准模型 - MSE: {mse_baseline:.6f}, R²: {r2_baseline:.6f}")

# 最终模型评估
print("\n=== 最终模型性能（基准 + 残差修正）===")
y_true_all, y_pred_final = [], []
for vcol in v_cols:
    m = (~df[vcol].isna()) & (~V_pred_final[vcol].isna())
    if m.any():
        y_true_all.append(df.loc[m, vcol].to_numpy())
        y_pred_final.append(V_pred_final.loc[m, vcol].to_numpy())

y_true_all = np.concatenate(y_true_all)
y_pred_final = np.concatenate(y_pred_final)

mse_final = mean_squared_error(y_true_all, y_pred_final)
r2_final = r2_score(y_true_all, y_pred_final)
print(f"最终模型 - MSE: {mse_final:.6f}, R²: {r2_final:.6f}")

# 改进程度
improvement = r2_final - r2_baseline
print(f"\n改进程度: R² 提升了 {improvement:.4f} ({improvement / r2_baseline * 100:.2f}%)")

# ==== 11. 分温度点评估 ====
print("\n分温度点评估:")
for tcol, vcol in zip(temp_cols, v_cols):
    m = (~df[tcol].isna()) & (~df[vcol].isna()) & (~V_pred_final[vcol].isna())
    if m.any():
        v_true = df.loc[m, vcol].to_numpy()
        v_pred = V_pred_final.loc[m, vcol].to_numpy()
        mse_temp = mean_squared_error(v_true, v_pred)
        r2_temp = r2_score(v_true, v_pred)
        print(f"  {tcol}: MSE = {mse_temp:.6f}, R2 = {r2_temp:.6f}")

# ==== 12. 保存结果 ====
id_col = df.columns[0]  # 物质ID/名称所在列
out_path = "isentropic_exponent_actual_vs_pred_with_residual_correction.xlsx"

rows = []
for idx, _ in df.iterrows():
    ID = df.at[idx, id_col]
    for j, (tcol, vcol) in enumerate(zip(temp_cols, v_cols), start=1):
        T = df.at[idx, tcol]
        V_act = df.at[idx, vcol]
        V_base = V_pred_baseline.at[idx, vcol] if pd.notna(V_pred_baseline.at[idx, vcol]) else np.nan
        V_final = V_pred_final.at[idx, vcol] if pd.notna(V_pred_final.at[idx, vcol]) else np.nan

        # 计算误差
        err_base = (V_base - V_act) if (pd.notna(V_base) and pd.notna(V_act)) else np.nan
        err_final = (V_final - V_act) if (pd.notna(V_final) and pd.notna(V_act)) else np.nan
        residual_correction = (V_final - V_base) if (pd.notna(V_final) and pd.notna(V_base)) else np.nan

        rows.append({
            id_col: ID,
            "temp_index": j,
            "temp_col": tcol,
            "T": T,
            "Isentropic_Exponent_actual": V_act,
            "Isentropic_Exponent_baseline": V_base,
            "Isentropic_Exponent_final": V_final,
            "error_baseline": err_base,
            "error_final": err_final,
            "residual_correction": residual_correction,
            "T_ref": Tb_pred_all[idx],
            "Isentropic_Exponent_ref": HVap_Tb_all[idx]
        })

long_compare = pd.DataFrame(rows).sort_values([id_col, "temp_index"])

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    long_compare.to_excel(writer, sheet_name="compare_long", index=False)

print(f"\n✅ 结果已保存到: {out_path}")
# —— 简洁相对误差统计（最终）——
relative_error_final = np.abs((y_pred_final - y_true_all) / y_true_all) * 100
within_1pct_final  = np.sum(relative_error_final <= 1)
within_5pct_final  = np.sum(relative_error_final <= 5)
within_10pct_final = np.sum(relative_error_final <= 10)
ard_final = np.mean(relative_error_final)

print("\n📊 总模型评估（基准 + 残差修正）：")
print(f"R²  = {r2_final:.4f}")
print(f"MSE = {mse_final:.6f}")
print(f"ARD = {ard_final:.2f}%")
print(f"✅ 误差 ≤ 1% 的数据点数量: {within_1pct_final}")
print(f"✅ 误差 ≤ 5% 的数据点数量: {within_5pct_final}")
print(f"✅ 误差 ≤ 10% 的数据点数量: {within_10pct_final}")
