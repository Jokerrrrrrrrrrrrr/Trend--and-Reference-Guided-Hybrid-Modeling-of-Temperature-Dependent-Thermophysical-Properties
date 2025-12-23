import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score

# ==== 常数与路径 ====
HV0, HVB, Tb0 = 9612.7, 15419.9, 222.543
T_ref = 298.15

# ==== 读取数据 ====
df_main = pd.read_excel("heat of vaporization 204.xlsx", sheet_name="Sheet1")
Nk_all = df_main.iloc[:, 13:32].apply(pd.to_numeric, errors='coerce')  # 19基团
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)

# ==== 读取需要的列 ====
temp_cols = list(df_main.columns[32:42])  # 10个温度列
hvap_cols = list(df_main.columns[42:52])  # 10个汽化焓列

# 数值化处理
for col in temp_cols + hvap_cols:
    df_main[col] = pd.to_numeric(df_main[col], errors='coerce')

# ==== Tb 模型 (参考温度预测模型) ====
Tb_raw = df_main.iloc[:, 5].values  # 读取参考温度列
mask_tb_ref = ~np.isnan(Tb_raw)  # 筛选参考温度有效数据
model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly[mask_tb_ref], np.exp(Tb_raw[mask_tb_ref] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))  # 所有物质的参考温度预测

# ==== HVPb 模型 (汽化焓预测模型) ====
df_Tb = pd.read_excel("selected_25_descriptors_data_boiling_point.xlsx")
X_Tb = df_Tb.drop(columns=["Heat of vaporization at boiling temperature"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["Heat of vaporization at boiling temperature"])
HVap_Tb_all = rf_Tb.predict(X_Tb)  # 预测参考汽化焓

# ==== A_k 系数训练 ====
G = Nk_all.values  # (n, 19) 基团数据
X_rows, y_rows = [], []
temp_eval = []  # 保存温度点信息用于评估

# 修正：正确构建训练集，处理NaN值
for i in range(len(df_main)):  # 遍历所有物质
    for j, (tcol, hvcol) in enumerate(zip(temp_cols, hvap_cols)):
        Tj = df_main.at[i, tcol]  # 温度值
        Hvapj = df_main.at[i, hvcol]  # 汽化焓值

        # 跳过NaN值
        if np.isnan(Tj) or np.isnan(Hvapj):
            continue

        Tb_i = Tb_pred_all[i]  # 物质i的参考温度
        HVap_Tb_i = HVap_Tb_all[i]  # 物质i的参考汽化焓

        # 特征：(T - T_ref) × G
        Xj = (Tj - Tb_i) * G[i]  # 形状: (19,)

        # 目标：Hvap - Hvap_ref
        yj = Hvapj - HVap_Tb_i

        X_rows.append(Xj)
        y_rows.append(yj)
        temp_eval.append((tcol, hvcol, i, j))

X_A = np.array(X_rows)  # (n_samples, 19)
y_A = np.array(y_rows)  # (n_samples,)

# 训练 A_k 系数模型
A_solver = HuberRegressor(fit_intercept=False, max_iter=5000)
A_solver.fit(X_A, y_A)
A_vec = A_solver.coef_  # 长度19，对应基团列顺序

# ==== 生成基准汽化焓预测 ====
HVap_pred_baseline = pd.DataFrame(index=df_main.index, columns=hvap_cols, dtype=float)

for i in range(len(df_main)):  # 遍历所有物质
    Tb_i = Tb_pred_all[i]  # 物质i的参考温度
    HVap_Tb_i = HVap_Tb_all[i]  # 物质i的参考汽化焓

    for j, (tcol, hvcol) in enumerate(zip(temp_cols, hvap_cols)):
        Tj = df_main.at[i, tcol]  # 温度值

        if np.isnan(Tj):
            HVap_pred_baseline.at[i, hvcol] = np.nan
            continue

        # 特征：(T - T_ref) × G
        Xj = (Tj - Tb_i) * G[i]

        # 预测：Hvap_ref + A_k × (T - T_ref) × G
        HVap_pred_j = HVap_Tb_i + Xj @ A_vec
        HVap_pred_baseline.at[i, hvcol] = HVap_pred_j

# ==== 残差机器学习模型 ====
print("训练残差机器学习模型...")

# 构建残差训练数据集
residual_features = []
residual_targets = []
sample_info = []  # 保存样本信息用于追踪

for tcol, hvcol in zip(temp_cols, hvap_cols):
    Tj = df_main[tcol].to_numpy()
    Hvapj = df_main[hvcol].to_numpy()
    msk = (~np.isnan(Tj)) & (~np.isnan(Hvapj))

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
        baseline_pred = HVap_pred_baseline.at[i, hvcol]
        baseline_features = [baseline_pred]

        # 参考值特征
        ref_features = [
            Tb_pred_all[i],  # 参考温度
            HVap_Tb_all[i],  # 参考汽化焓
        ]

        # 组合所有特征
        all_features = base_features + temp_features + baseline_features + ref_features
        residual_features.append(all_features)

        # 残差目标：实际值 - 基准预测值
        residual = Hvapj[i] - baseline_pred
        residual_targets.append(residual)

        sample_info.append((i, tcol, hvcol))

residual_features = np.array(residual_features)
residual_targets = np.array(residual_targets)

print(f"残差训练集形状: {residual_features.shape}")
print(f"残差目标形状: {residual_targets.shape}")

# 标准化特征
scaler = StandardScaler()
residual_features_scaled = scaler.fit_transform(residual_features)

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

# ==== 生成最终预测（基准 + 残差修正） ====
HVap_pred_final = pd.DataFrame(index=df_main.index, columns=hvap_cols, dtype=float)

for tcol, hvcol in zip(temp_cols, hvap_cols):
    Tj = df_main[tcol].to_numpy()

    # 为所有样本构建特征
    features_list = []
    valid_indices = []

    for i in range(len(df_main)):
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
        baseline_pred = HVap_pred_baseline.at[i, hvcol]
        baseline_features = [baseline_pred]
        ref_features = [Tb_pred_all[i], HVap_Tb_all[i]]

        all_features = base_features + temp_features + baseline_features + ref_features
        features_list.append(all_features)
        valid_indices.append(i)

    if features_list:
        features_array = np.array(features_list)
        features_scaled = scaler.transform(features_array)

        # 预测残差
        residual_pred = residual_model.predict(features_scaled)

        # 最终预测 = 基准预测 + 残差修正
        for idx, residual_val in zip(valid_indices, residual_pred):
            final_pred = HVap_pred_baseline.at[idx, hvcol] + residual_val
            HVap_pred_final.at[idx, hvcol] = final_pred

    # 对于无效温度点，保持NaN
    HVap_pred_final[hvcol] = np.where(np.isnan(Tj), np.nan, HVap_pred_final[hvcol])

# ==== 评估模型性能 ====
# 基准模型评估
print("\n=== 基准模型性能 ===")
y_true_all, y_pred_baseline = [], []
for hvcol in hvap_cols:
    m = (~df_main[hvcol].isna()) & (~HVap_pred_baseline[hvcol].isna())
    if m.any():
        y_true_all.append(df_main.loc[m, hvcol].to_numpy())
        y_pred_baseline.append(HVap_pred_baseline.loc[m, hvcol].to_numpy())

y_true_all = np.concatenate(y_true_all)
y_pred_baseline = np.concatenate(y_pred_baseline)

mse_baseline = mean_squared_error(y_true_all, y_pred_baseline)
r2_baseline = r2_score(y_true_all, y_pred_baseline)
print(f"基准模型 - MSE: {mse_baseline:.6f}, R²: {r2_baseline:.6f}")

# 最终模型评估
print("\n=== 最终模型性能（基准 + 残差修正）===")
y_true_all, y_pred_final = [], []
for hvcol in hvap_cols:
    m = (~df_main[hvcol].isna()) & (~HVap_pred_final[hvcol].isna())
    if m.any():
        y_true_all.append(df_main.loc[m, hvcol].to_numpy())
        y_pred_final.append(HVap_pred_final.loc[m, hvcol].to_numpy())

y_true_all = np.concatenate(y_true_all)
y_pred_final = np.concatenate(y_pred_final)

mse_final = mean_squared_error(y_true_all, y_pred_final)
r2_final = r2_score(y_true_all, y_pred_final)
print(f"最终模型 - MSE: {mse_final:.6f}, R²: {r2_final:.6f}")

# 改进程度
improvement = r2_final - r2_baseline
print(f"\n改进程度: R² 提升了 {improvement:.4f} ({improvement / r2_baseline * 100:.2f}%)")

# ==== 分温度点评估 ====
print("\n分温度点评估:")
for tcol, hvcol in zip(temp_cols, hvap_cols):
    m = (~df_main[tcol].isna()) & (~df_main[hvcol].isna()) & (~HVap_pred_final[hvcol].isna())
    if m.any():
        hvap_true = df_main.loc[m, hvcol].to_numpy()
        hvap_pred = HVap_pred_final.loc[m, hvcol].to_numpy()
        mse_temp = mean_squared_error(hvap_true, hvap_pred)
        r2_temp = r2_score(hvap_true, hvap_pred)
        print(f"  {tcol}: MSE = {mse_temp:.6f}, R2 = {r2_temp:.6f}")

# ==== 保存最终的结果 ====
id_col = df_main.columns[0]  # 物质ID/名称所在列
out_path = "hvap_actual_vs_pred_with_residual_correction.xlsx"

rows = []
for idx, _ in df_main.iterrows():
    ID = df_main.at[idx, id_col]
    for j, (tcol, hvcol) in enumerate(zip(temp_cols, hvap_cols), start=1):
        T = df_main.at[idx, tcol]
        HVap_act = df_main.at[idx, hvcol]
        HVap_base = HVap_pred_baseline.at[idx, hvcol] if pd.notna(HVap_pred_baseline.at[idx, hvcol]) else np.nan
        HVap_final = HVap_pred_final.at[idx, hvcol] if pd.notna(HVap_pred_final.at[idx, hvcol]) else np.nan

        # 计算误差
        err_base = (HVap_base - HVap_act) if (pd.notna(HVap_base) and pd.notna(HVap_act)) else np.nan
        err_final = (HVap_final - HVap_act) if (pd.notna(HVap_final) and pd.notna(HVap_act)) else np.nan
        residual_correction = (HVap_final - HVap_base) if (pd.notna(HVap_final) and pd.notna(HVap_base)) else np.nan

        rows.append({
            id_col: ID,
            "temp_index": j,
            "temp_col": tcol,
            "T": T,
            "HVap_actual": HVap_act,
            "HVap_baseline": HVap_base,
            "HVap_final": HVap_final,
            "error_baseline": err_base,
            "error_final": err_final,
            "residual_correction": residual_correction,
            "T_ref": Tb_pred_all[idx],
            "HVap_ref": HVap_Tb_all[idx]
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
