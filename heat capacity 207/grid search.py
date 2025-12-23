import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from scipy.optimize import minimize
from itertools import product

# ===== 0. 工具：权重网格生成 =====
def generate_weight_grid(step=0.2):
    grid = []
    for w1 in np.arange(0, 1.01, step):
        for w2 in np.arange(0, 1.01 - w1, step):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-8:  # 允许微小的浮点误差
                continue
            grid.append([w1, w2, w3])
    return np.array(grid)

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

# ===== 4. 真实参考点值 =====
Cp1_true_all = df.iloc[:, 9].astype(float).values
Cp2_true_all = df.iloc[:, 50].astype(float).values
T1_true_all = df[target_column_T1].astype(float).values
T2_true_all = 1.5 * T1_true_all
slope_true_all = (Cp2_true_all - Cp1_true_all) / (T2_true_all - T1_true_all)

# ===== 5. 构建实验点样本 =====
slope_feat_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)

X_exp_list, y_exp_list, mat_idx_list = [], [], []
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

X_exp = np.asarray(X_exp_list)
y_exp = np.asarray(y_exp_list)
mat_idx_per_sample = np.asarray(mat_idx_list)

# ===== 6. Huber损失函数 =====
def huber_loss(residuals, epsilon=1.35):
    abs_res = np.abs(residuals)
    return np.where(abs_res <= epsilon,
                    0.5 * residuals ** 2,
                    epsilon * (abs_res - 0.5 * epsilon))

# ===== 7. 标准化函数 =====
def normalize_loss_values(loss_values):
    if len(loss_values) == 0:
        return 0
    min_val = np.min(loss_values)
    max_val = np.max(loss_values)
    if max_val - min_val < 1e-10:
        return np.sum(loss_values)
    normalized_values = (loss_values - min_val) / (max_val - min_val)
    return np.sum(normalized_values)

# ===== 8. 加权损失函数 =====
def weighted_huber_loss(theta, X_exp, y_exp, mat_idx,
                        Cp1_true, Cp2_true, slope_true,
                        T1_hat, T2_hat, w1, w2, w3,
                        alpha=0.0001, epsilon=1.35):
    beta = theta[:-1]
    b = theta[-1]

    all_exp_losses = []
    all_ref_losses = []
    all_slope_losses = []

    y_pred_exp = X_exp @ beta + b
    exp_residuals = y_exp - y_pred_exp
    exp_losses = huber_loss(exp_residuals, epsilon)
    all_exp_losses.extend(exp_losses)

    unique_materials = np.unique(mat_idx)
    for mat_idx_val in unique_materials:
        Nk = X_groups.iloc[mat_idx_val].values.astype(float)
        s_feat = slope_feat_all[mat_idx_val]

        x_T1 = np.concatenate([Nk, Nk * T1_hat[mat_idx_val], [s_feat * T1_hat[mat_idx_val]]])
        Cp1_pred = x_T1 @ beta + b
        all_ref_losses.append(huber_loss(Cp1_true[mat_idx_val] - Cp1_pred, epsilon))

        x_T2 = np.concatenate([Nk, Nk * T2_hat[mat_idx_val], [s_feat * T2_hat[mat_idx_val]]])
        Cp2_pred = x_T2 @ beta + b
        all_ref_losses.append(huber_loss(Cp2_true[mat_idx_val] - Cp2_pred, epsilon))

        if T2_hat[mat_idx_val] - T1_hat[mat_idx_val] > 1e-10:
            slope_pred = (Cp2_pred - Cp1_pred) / (T2_hat[mat_idx_val] - T1_hat[mat_idx_val])
            all_slope_losses.append(huber_loss(slope_true[mat_idx_val] - slope_pred, epsilon))

    L_exp_norm = normalize_loss_values(np.array(all_exp_losses))
    L_ref_norm = normalize_loss_values(np.array(all_ref_losses))
    L_slope_norm = normalize_loss_values(np.array(all_slope_losses))

    regularization = alpha * np.sum(beta ** 2)
    return w1 * L_exp_norm + w2 * L_ref_norm + w3 * L_slope_norm + regularization

# ===== 9. 网格搜索 =====
weight_grid = generate_weight_grid(step=0.25)
best_w = None
best_r2 = -np.inf
best_theta = None

for w in weight_grid:
    print(f"\n=== 权重: {[f'{x:.2f}' for x in w]} ===")

    lin_model = HuberRegressor(max_iter=100000).fit(X_exp, y_exp)
    theta0 = np.concatenate([lin_model.coef_, [lin_model.intercept_]])

    res = minimize(
        weighted_huber_loss,
        theta0,
        args=(X_exp, y_exp, mat_idx_per_sample,
              Cp1_true_all, Cp2_true_all, slope_true_all,
              T1_hat_all, T2_hat_all,
              w[0], w[1], w[2],
              0.0001, 1.35),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-5}
    )

    theta = res.x
    y_pred = X_exp @ theta[:-1] + theta[-1]
    r2 = r2_score(y_exp, y_pred)
    print(f"R² = {r2:.6f}")

    if r2 > best_r2:
        best_r2 = r2
        best_w = w
        best_theta = theta

print(f"\n最优权重: {best_w}, 最优R² = {best_r2:.6f}")

# ===== 10. 最终评估 =====
y_pred_final = X_exp @ best_theta[:-1] + best_theta[-1]
r2_final = r2_score(y_exp, y_pred_final)
rel_err = np.abs((y_pred_final - y_exp) / np.maximum(np.abs(y_exp), 1e-12)) * 100

print(f"\n最终R² = {r2_final:.6f}, 平均相对误差 = {np.mean(rel_err):.2f}%, 中位数相对误差 = {np.median(rel_err):.2f}%")
