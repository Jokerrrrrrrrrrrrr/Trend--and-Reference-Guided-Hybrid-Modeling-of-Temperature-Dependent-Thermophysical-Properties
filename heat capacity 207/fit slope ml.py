import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# ========= 1. 读取数据 =========
file_path = "heat capacity 207.xlsx"
sheet = "Sheet1"
df = pd.read_excel(file_path, sheet_name=sheet).copy()

df = df.dropna(subset=[df.columns[0]])
df[df.columns[0]] = df[df.columns[0]].astype(int)

# ========= 2. 列定义 =========
group_cols = list(df.columns[11:30])  # 19个基团列
temp_cols = list(df.columns[30:40])  # 10个温度点
cp_cols = list(df.columns[40:50])  # 10个 Cp 值
target_column_T1 = 'ASPEN Half Critical T'

# 做一个温度列 -> 对应Cp列 的映射（索引一一对应）
temp_to_cp = {t: c for t, c in zip(temp_cols, cp_cols)}

# 数值化
for cols in [group_cols, temp_cols, cp_cols]:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ========= 3. 子模型训练：T_ref(=T1) 与 C_pref(=Cp1) =========
X_groups = df[group_cols].fillna(0)

valid_mask = ~df[target_column_T1].isna()
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_groups[valid_mask])

y_T1 = df.loc[valid_mask, target_column_T1].to_numpy()
T1_model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
).fit(X_poly_train, y_T1)

# 对所有样本预测 T_ref
X_poly_all = poly.transform(X_groups)
T_ref_pred = T1_model.predict(X_poly_all)  # (n,)

# Cp1模型（示例里用第9列作为目标）
y_cp1_target = df.iloc[:, 9].to_numpy()
Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, y_cp1_target)

# 对所有样本预测 C_pref
C_pref_pred = Cp1_model.predict(X_groups)  # (n,)

# ========= 4. 构造 A_k 的训练集（物质×温度点展开）=========
G = X_groups.to_numpy()  # (n, 19)
X_rows, y_rows = [], []
temp_eval = []  # 保存 (tcol, cpcol, msk) 以便分温度评估

for tcol, cpcol in zip(temp_cols, cp_cols):
    Tj = df[tcol].to_numpy()  # (n,)
    CPj = df[cpcol].to_numpy()  # (n,)
    msk = (~np.isnan(Tj)) & (~np.isnan(CPj))
    if msk.sum() == 0:
        continue

    # 特征：(T - T_ref)[:, None] * G  → (n_j, 19)
    Xj = ((Tj - T_ref_pred)[:, None] * G)[msk]
    # 目标：Cp - C_pref（用于训练A）
    yj = (CPj - C_pref_pred)[msk]

    X_rows.append(Xj)
    y_rows.append(yj)
    temp_eval.append((tcol, cpcol, msk))

X_A = np.vstack(X_rows)  # (sum_j n_j, 19)
y_A = np.concatenate(y_rows)  # (sum_j n_j,)

# ========= 5. 拟合 A_k（无截距；截距由 C_pref 承担）=========
A_solver = HuberRegressor(fit_intercept=False, max_iter=5000)
A_solver.fit(X_A, y_A)
A_vec = A_solver.coef_  # 长度19，对应 group_cols 顺序

# ========= 6. 生成基准预测 =========
Cp_pred_baseline = pd.DataFrame(index=df.index, columns=cp_cols, dtype=float)
for tcol, cpcol in zip(temp_cols, cp_cols):
    Tj = df[tcol].to_numpy()
    Xj = (Tj - T_ref_pred)[:, None] * G
    Cp_pred_j = C_pref_pred + Xj @ A_vec
    Cp_pred_baseline[cpcol] = np.where(np.isnan(Tj), np.nan, Cp_pred_j)

# ========= 7. 残差机器学习模型 =========
print("训练残差机器学习模型...")

# 构建残差训练数据集
residual_features = []
residual_targets = []
sample_info = []  # 保存样本信息用于追踪

for tcol, cpcol in zip(temp_cols, cp_cols):
    Tj = df[tcol].to_numpy()
    CPj = df[cpcol].to_numpy()
    msk = (~np.isnan(Tj)) & (~np.isnan(CPj))

    for i in np.where(msk)[0]:
        # 基础特征：基团组成
        base_features = list(G[i])

        # 温度相关特征
        temp_features = [
            Tj[i],  # 绝对温度
            Tj[i] - T_ref_pred[i],  # 相对于参考温度的差值
            Tj[i] / T_ref_pred[i] if T_ref_pred[i] > 0 else 0,  # 相对温度
            np.log(Tj[i]) if Tj[i] > 0 else 0,  # 温度对数
        ]

        # 基准预测值作为特征
        baseline_pred = Cp_pred_baseline.at[i, cpcol]
        baseline_features = [baseline_pred]

        # 组合所有特征
        all_features = base_features + temp_features + baseline_features
        residual_features.append(all_features)

        # 残差目标：实际值 - 基准预测值
        residual = CPj[i] - baseline_pred
        residual_targets.append(residual)

        sample_info.append((i, tcol, cpcol))

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

# ========= 8. 生成最终预测（基准 + 残差修正）=========
Cp_pred_final = pd.DataFrame(index=df.index, columns=cp_cols, dtype=float)

for tcol, cpcol in zip(temp_cols, cp_cols):
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
            Tj[i] - T_ref_pred[i],
            Tj[i] / T_ref_pred[i] if T_ref_pred[i] > 0 else 0,
            np.log(Tj[i]) if Tj[i] > 0 else 0,
        ]
        baseline_pred = Cp_pred_baseline.at[i, cpcol]
        baseline_features = [baseline_pred]

        all_features = base_features + temp_features + baseline_features
        features_list.append(all_features)
        valid_indices.append(i)

    if features_list:
        features_array = np.array(features_list)
        features_scaled = scaler.transform(features_array)

        # 预测残差
        residual_pred = residual_model.predict(features_scaled)

        # 最终预测 = 基准预测 + 残差修正
        for idx, residual_val in zip(valid_indices, residual_pred):
            final_pred = Cp_pred_baseline.at[idx, cpcol] + residual_val
            Cp_pred_final.at[idx, cpcol] = final_pred

    # 对于无效温度点，保持NaN
    Cp_pred_final[cpcol] = np.where(np.isnan(Tj), np.nan, Cp_pred_final[cpcol])

# ========= 9. 评估模型性能 =========
# 9.1 基准模型评估
print("\n=== 基准模型性能 ===")
y_true_all, y_pred_baseline = [], []
for cpcol in cp_cols:
    m = (~df[cpcol].isna()) & (~Cp_pred_baseline[cpcol].isna())
    if m.any():
        y_true_all.append(df.loc[m, cpcol].to_numpy())
        y_pred_baseline.append(Cp_pred_baseline.loc[m, cpcol].to_numpy())

y_true_all = np.concatenate(y_true_all)
y_pred_baseline = np.concatenate(y_pred_baseline)

mse_baseline = mean_squared_error(y_true_all, y_pred_baseline)
r2_baseline = r2_score(y_true_all, y_pred_baseline)
print(f"基准模型 - MSE: {mse_baseline:.6f}, R²: {r2_baseline:.6f}")

# 9.2 最终模型评估
print("\n=== 最终模型性能（基准 + 残差修正）===")
y_true_all, y_pred_final = [], []
for cpcol in cp_cols:
    m = (~df[cpcol].isna()) & (~Cp_pred_final[cpcol].isna())
    if m.any():
        y_true_all.append(df.loc[m, cpcol].to_numpy())
        y_pred_final.append(Cp_pred_final.loc[m, cpcol].to_numpy())

y_true_all = np.concatenate(y_true_all)
y_pred_final = np.concatenate(y_pred_final)

mse_final = mean_squared_error(y_true_all, y_pred_final)
r2_final = r2_score(y_true_all, y_pred_final)
print(f"最终模型 - MSE: {mse_final:.6f}, R²: {r2_final:.6f}")

# 9.3 改进程度
improvement = r2_final - r2_baseline
print(f"\n改进程度: R² 提升了 {improvement:.4f} ({improvement / r2_baseline * 100:.2f}%)")

# 9.4 分温度点评估最终模型
print("\n=== 分温度点评估（最终模型）===")
for tcol, cpcol, msk in temp_eval:
    cp_true = df[cpcol].to_numpy()[msk]
    cp_pred = Cp_pred_final[cpcol].to_numpy()[msk]
    print(f"  {tcol}: MSE = {mean_squared_error(cp_true, cp_pred):.6f}, "
          f"R² = {r2_score(cp_true, cp_pred):.6f}")

# ========= 10. 保存结果 =========
id_col = df.columns[0]  # 物质ID/名称所在列
out_path = "cp_actual_vs_pred_with_residual_correction.xlsx"

rows = []
for idx, _ in df.iterrows():
    ID = df.at[idx, id_col]
    for j, (tcol, cpcol) in enumerate(zip(temp_cols, cp_cols), start=1):
        T = df.at[idx, tcol]
        Cp_act = df.at[idx, cpcol]
        Cp_base = Cp_pred_baseline.at[idx, cpcol] if pd.notna(Cp_pred_baseline.at[idx, cpcol]) else np.nan
        Cp_final = Cp_pred_final.at[idx, cpcol] if pd.notna(Cp_pred_final.at[idx, cpcol]) else np.nan

        # 计算误差
        err_base = (Cp_base - Cp_act) if (pd.notna(Cp_base) and pd.notna(Cp_act)) else np.nan
        err_final = (Cp_final - Cp_act) if (pd.notna(Cp_final) and pd.notna(Cp_act)) else np.nan
        residual_correction = (Cp_final - Cp_base) if (pd.notna(Cp_final) and pd.notna(Cp_base)) else np.nan

        rows.append({
            id_col: ID,
            "temp_index": j,
            "temp_col": tcol,
            "T": T,
            "Cp_actual": Cp_act,
            "Cp_baseline": Cp_base,
            "Cp_final": Cp_final,
            "error_baseline": err_base,
            "error_final": err_final,
            "residual_correction": residual_correction,
            "T_ref": T_ref_pred[idx] if idx < len(T_ref_pred) else np.nan,
            "Cp_ref": C_pref_pred[idx] if idx < len(C_pref_pred) else np.nan
        })

long_compare = pd.DataFrame(rows).sort_values([id_col, "temp_index"])

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    long_compare.to_excel(writer, sheet_name="compare_long", index=False)

print(f"\n✅ 结果已保存到: {out_path}")
# ========= 9.x 简单误差统计 =========
relative_error = np.abs((y_pred_final - y_true_all) / y_true_all) * 100
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

ard = np.mean(relative_error)  # 平均相对偏差 %

print("\n📊 总模型评估（含 slope×T 特征）：")
print(f"R²  = {r2_final:.4f}")
print(f"MSE = {mse_final:.2f}")
print(f"ARD = {ard:.2f}%")
print(f"✅ 误差 ≤ 1% 的数据点数量: {within_1pct}")
print(f"✅ 误差 ≤ 5% 的数据点数量: {within_5pct}")
print(f"✅ 误差 ≤ 10% 的数据点数量: {within_10pct}")
