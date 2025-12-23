import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from scipy.optimize import minimize


# ===== 0. 工具：候选权重采样（外层） =====
def sample_weight_triplets(n=2, seed=2025):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    return W


# ===== 1. 读取数据 =====
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.dropna(subset=[df.columns[0]])
df[df.columns[0]] = df[df.columns[0]].astype(int)

# ===== 2. 列定义 =====
group_cols = df.columns[11:30]
temp_cols = df.columns[30:40]
cp_cols = df.columns[40:50]
target_column_T1 = 'ASPEN Half Critical T'

material_id_col = df.columns[0]
material_ids_all = df[material_id_col].values

# ===== 3. 子模型训练 =====
X_groups = df[group_cols]
valid_mask = ~df[target_column_T1].isna()

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_groups[valid_mask])

y_T1 = df.loc[valid_mask, target_column_T1].values
T1_model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0
).fit(X_poly, y_T1)

Cp1_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 9])
Cp2_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 50])

X_poly_all = poly.transform(X_groups)
T1_hat_all = T1_model.predict(X_poly_all)
T2_hat_all = 1.5 * T1_hat_all

Cp1_pred_all = Cp1_model.predict(X_groups)
Cp2_pred_all = Cp2_model.predict(X_groups)

# ===== 4. 计算斜率 =====
with np.errstate(divide='ignore', invalid='ignore'):
    slope_pred_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)

# ===== 5. 真实参考点值 =====
Cp1_true_all = df.iloc[:, 9].astype(float).values
Cp2_true_all = df.iloc[:, 50].astype(float).values
T1_true_all = df[target_column_T1].astype(float).values
T2_true_all = 1.5 * T1_true_all

with np.errstate(divide='ignore', invalid='ignore'):
    slope_true_all = (Cp2_true_all - Cp1_true_all) / (T2_true_all - T1_true_all)

# ===== 6. 构建实验点样本 =====
slope_feat_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)

X_exp_list, y_exp_list, mat_idx_list, T_list = [], [], [], []
for i in range(len(df)):
    Nk = X_groups.iloc[i].values.astype(float)
    s_feat = slope_feat_all[i]
    temps_i = df.loc[df.index[i], temp_cols].values.astype(float)
    cps_i = df.loc[df.index[i], cp_cols].values.astype(float)

    for T, Cp in zip(temps_i, cps_i):
        if not (np.isfinite(T) and np.isfinite(Cp)):
            continue
        x = np.concatenate([Nk, Nk * T, [s_feat * T]])
        X_exp_list.append(x)
        y_exp_list.append(Cp)
        mat_idx_list.append(i)
        T_list.append(T)

X_exp = np.asarray(X_exp_list)
y_exp = np.asarray(y_exp_list)
mat_idx_per_sample = np.asarray(mat_idx_list)
T_per_sample = np.asarray(T_list)


# ===== 7. Huber损失函数 =====
def huber_loss(residuals, epsilon=1.35):
    abs_res = np.abs(residuals)
    return np.where(abs_res <= epsilon,
                    0.5 * residuals ** 2,
                    epsilon * (abs_res - 0.5 * epsilon))


# ===== 8. 标准化函数 =====
def normalize_loss_values(loss_values):
    """对损失值进行Min-Max标准化"""
    if len(loss_values) == 0:
        return 0

    min_val = np.min(loss_values)
    max_val = np.max(loss_values)

    if max_val - min_val < 1e-10:  # 避免除零
        return np.sum(loss_values)  # 如果所有值相同，返回原始和

    # 对每个损失值进行标准化，然后求和
    normalized_values = (loss_values - min_val) / (max_val - min_val)
    return np.sum(normalized_values)


# ===== 9. 加权损失函数（完全标准化版本） =====
def weighted_huber_loss(theta, X_exp, y_exp, mat_idx,
                        Cp1_true, Cp2_true, slope_true,
                        T1_hat, T2_hat, w1, w2, w3, alpha=0.0001, epsilon=1.35):
    """使用Huber损失的加权目标函数（所有损失项都标准化）"""
    beta = theta[:-1]
    b = theta[-1]

    # 收集所有损失值（不求和，用于标准化）
    all_exp_losses = []  # 每个实验点的损失
    all_ref_losses = []  # 每个参考点的损失
    all_slope_losses = []  # 每个斜率点的损失

    # 1. 实验点损失
    y_pred_exp = X_exp @ beta + b
    exp_residuals = y_exp - y_pred_exp
    exp_losses = huber_loss(exp_residuals, epsilon)
    all_exp_losses.extend(exp_losses)

    # 2. 参考点损失
    unique_materials = np.unique(mat_idx)

    for mat_idx_val in unique_materials:
        Nk = X_groups.iloc[mat_idx_val].values.astype(float)
        s_feat = slope_feat_all[mat_idx_val]

        # 参考点1 (T1)
        x_T1 = np.concatenate([Nk, Nk * T1_hat[mat_idx_val], [s_feat * T1_hat[mat_idx_val]]])
        Cp1_pred = x_T1 @ beta + b
        ref1_residual = Cp1_true[mat_idx_val] - Cp1_pred
        ref1_loss = huber_loss(ref1_residual, epsilon)
        all_ref_losses.append(ref1_loss)

        # 参考点2 (T2)
        x_T2 = np.concatenate([Nk, Nk * T2_hat[mat_idx_val], [s_feat * T2_hat[mat_idx_val]]])
        Cp2_pred = x_T2 @ beta + b
        ref2_residual = Cp2_true[mat_idx_val] - Cp2_pred
        ref2_loss = huber_loss(ref2_residual, epsilon)
        all_ref_losses.append(ref2_loss)

        # 3. 斜率损失
        if T2_hat[mat_idx_val] - T1_hat[mat_idx_val] > 1e-10:
            slope_pred = (Cp2_pred - Cp1_pred) / (T2_hat[mat_idx_val] - T1_hat[mat_idx_val])
            slope_residual = slope_true[mat_idx_val] - slope_pred
            slope_loss = huber_loss(slope_residual, epsilon)
            all_slope_losses.append(slope_loss)

    # 对所有损失项进行标准化
    L_exp_norm = normalize_loss_values(np.array(all_exp_losses))
    L_ref_norm = normalize_loss_values(np.array(all_ref_losses))
    L_slope_norm = normalize_loss_values(np.array(all_slope_losses))

    # 输出调试信息
    # print(f"损失项数量 - 实验点: {len(all_exp_losses)}, 参考点: {len(all_ref_losses)}, 斜率: {len(all_slope_losses)}")
    # if len(all_exp_losses) > 0:
    #     print(f"实验点损失范围: [{np.min(all_exp_losses):.3f}, {np.max(all_exp_losses):.3f}]")
    # if len(all_ref_losses) > 0:
    #     print(f"参考点损失范围: [{np.min(all_ref_losses):.3f}, {np.max(all_ref_losses):.3f}]")
    # if len(all_slope_losses) > 0:
    #     print(f"斜率损失范围: [{np.min(all_slope_losses):.6f}, {np.max(all_slope_losses):.6f}]")

    # 正则化项
    regularization = alpha * np.sum(beta ** 2)

    # 返回加权的标准化损失
    total_loss = w1 * L_exp_norm + w2 * L_ref_norm + w3 * L_slope_norm + regularization
    # print(
    #    f"加权损失: {total_loss:.3f} (实验点: {w1 * L_exp_norm:.3f}, 参考点: {w2 * L_ref_norm:.3f}, 斜率: {w3 * L_slope_norm:.3f}, 正则化: {regularization:.3f})")

    return total_loss


# ===== 10. 优化循环 =====
candidate_ws = sample_weight_triplets(n=20, seed=2025)
best_w = None
best_r2 = -np.inf
best_theta = None

for i, w in enumerate(candidate_ws):
    # 直接使用采样权重，不再进行放大处理
    adjusted_w = w.copy()

    print(f"\n=== 第{i + 1}组权重: {[f'{x:.6f}' for x in adjusted_w]} ===")

    # 使用线性回归结果作为更好的初始值
    lin_model = HuberRegressor(max_iter=100000).fit(X_exp, y_exp)
    theta0 = np.concatenate([lin_model.coef_, [lin_model.intercept_]])

    # 使用L-BFGS优化
    res = minimize(
        weighted_huber_loss,
        theta0,
        args=(X_exp, y_exp, mat_idx_per_sample,
              Cp1_true_all, Cp2_true_all, slope_true_all,
              T1_hat_all, T2_hat_all,
              adjusted_w[0], adjusted_w[1], adjusted_w[2],
              0.0001, 1.35),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-5, 'iprint': -1,'disp': True}
    )

    print(f"优化状态: {res.message}")

    if not res.success:
        print(f"优化未完全收敛，但继续使用当前结果")

    theta = res.x
    y_pred = X_exp @ theta[:-1] + theta[-1]
    r2 = r2_score(y_exp, y_pred)

    print(f"R² = {r2:.6f}")

    if r2 > best_r2:
        best_r2 = r2
        best_w = adjusted_w
        best_theta = theta

print(f"\n最优权重: {best_w}, R² = {best_r2:.6f}")

# ===== 11. 最终评估 =====
y_pred_final = X_exp @ best_theta[:-1] + best_theta[-1]
r2_final = r2_score(y_exp, y_pred_final)
rel_err = np.abs((y_pred_final - y_exp) / np.maximum(np.abs(y_exp), 1e-12)) * 100

print(f"\n最终模型 R² = {r2_final:.6f}")
print(f"平均相对误差: {np.mean(rel_err):.2f}%")
print(f"中位数相对误差: {np.median(rel_err):.2f}%")

# ===== 12. 输出特征重要性 =====
feature_names = list(group_cols) + [f"{col}_T" for col in group_cols] + ["slope_feat_T"]
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': best_theta[:-1],
    'importance': np.abs(best_theta[:-1]) / np.sum(np.abs(best_theta[:-1]))
})
print("\n特征系数重要性 (前10):")
print(coef_df.sort_values('importance', ascending=False).head(10))