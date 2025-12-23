# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import HuberRegressor
# from scipy.optimize import minimize
#
#
# # ===== 0. 工具：候选权重采样（外层） =====
# def sample_weight_triplets(n=2, seed=2025):
#     """
#     采样 n 组 (w1,w2,w3)，非负且和为1（避免全0退化）。
#     用 Dirichlet(1,1,1) 随机权重，覆盖面更均匀。
#     """
#     rng = np.random.default_rng(seed)
#     W = rng.dirichlet([1.0, 1.0, 1.0], size=n)
#     return W  # shape: (n, 3)
#
#
# # ===== 1. 读取数据 =====
# file_path = "heat capacity 207.xlsx"
# df = pd.read_excel(file_path, sheet_name="Sheet1")
# df = df.dropna(subset=[df.columns[0]])
# df[df.columns[0]] = df[df.columns[0]].astype(int)
#
# # ===== 2. 列定义 =====
# group_cols = df.columns[11:30]  # 19 个基团特征
# temp_cols = df.columns[30:40]  # 10 个实验温度点（网格）
# cp_cols = df.columns[40:50]  # 10 个实验 Cp（对应上面温度网格）
# target_column_T1 = 'ASPEN Half Critical T'  # 作为 T1_true 的列（若缺失则该物质在参考项/斜率项里跳过）
#
# material_id_col = df.columns[0]
# material_ids_all = df[material_id_col].values
#
# # ===== 3. 子模型训练 =====
# X_groups = df[group_cols]  # 基团特征
# valid_mask = ~df[target_column_T1].isna()
#
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X_groups[valid_mask])
#
# # 使用 GradientBoostingRegressor 预测 T1
# y_T1 = df.loc[valid_mask, target_column_T1].values
# T1_model = GradientBoostingRegressor(
#     n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
# ).fit(X_poly, y_T1)
#
# # 使用 HuberRegressor 预测 Cp1 和 Cp2
# Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 9])
# Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 50])
#
# # 预测 T1 和 T2
# X_poly_all = poly.transform(X_groups)
# T1_hat_all = T1_model.predict(X_poly_all)  # 预测 T1
# T2_hat_all = 1.5 * T1_hat_all  # 假设 T2 = 1.5 * T1
#
# # 预测 Cp1 和 Cp2
# Cp1_pred_all = Cp1_model.predict(X_groups)
# Cp2_pred_all = Cp2_model.predict(X_groups)
#
# # ===== 4. 计算斜率 =====
# # 每个物质的 T1 和 T2 应该重复与样本数量一致的次数
# T1_hat_all_expanded = np.repeat(T1_hat_all, len(temp_cols))  # 扩展 T1 为与样本数匹配
# T2_hat_all_expanded = np.repeat(T2_hat_all, len(temp_cols))  # 扩展 T2 为与样本数匹配
# Cp1_pred_all_expanded = np.repeat(Cp1_pred_all, len(temp_cols))  # 扩展 Cp1 为与样本数匹配
# Cp2_pred_all_expanded = np.repeat(Cp2_pred_all, len(temp_cols))  # 扩展 Cp2 为与样本数匹配
#
# # 计算斜率：每个物质一个斜率
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_pred_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# # ===== 5. 真实参考点值 (Cp1_true_all, Cp2_true_all) =====
# # 真实的 Cp1 和 Cp2
# Cp1_true_all = df.iloc[:, 9].astype(float).values  # 参考点1的真实 Cp
# Cp2_true_all = df.iloc[:, 50].astype(float).values  # 参考点2的真实 Cp
#
# # 真实的 T1 和 T2
# T1_true_all = df[target_column_T1].astype(float).values  # 参考点1的真实温度
# T2_true_all = 1.5 * T1_true_all  # 假设 T2 = 1.5 * T1
#
# # 计算真实斜率
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_true_all = (Cp2_true_all - Cp1_true_all) / (T2_true_all - T1_true_all)
#
# # ===== 6. 构建"实验点样本"：X(T) & y（线性模型的输入/输出） =====
# slope_feat_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# X_exp_list, y_exp_list, mat_idx_list, T_list = [], [], [], []
# for i in range(len(df)):
#     Nk = X_groups.iloc[i].values.astype(float)
#     s_feat = slope_feat_all[i]  # 固定特征，不随内层参数变化
#     temps_i = df.loc[df.index[i], temp_cols].values.astype(float)
#     cps_i = df.loc[df.index[i], cp_cols].values.astype(float)
#
#     for T, Cp in zip(temps_i, cps_i):
#         if not (np.isfinite(T) and np.isfinite(Cp)):
#             continue
#         x = np.concatenate([Nk, Nk * T, [s_feat * T]])  # 19 + 19 + 1 = 39 维
#         X_exp_list.append(x)
#         y_exp_list.append(Cp)
#         mat_idx_list.append(i)  # 记录该样本属于哪个物质
#         T_list.append(T)
#
# X_exp = np.asarray(X_exp_list)  # (N_samples, 39)
# y_exp = np.asarray(y_exp_list)  # (N_samples,)
# mat_idx_per_sample = np.asarray(mat_idx_list)
# T_per_sample = np.asarray(T_list)
#
#
# # ===== 7. 损失函数：自定义三项损失（不做平均） =====
# def loss_sum_three_parts(y_exp_true, y_exp_pred,
#                          Cp1_true, Cp2_true,
#                          Cp1_pred, Cp2_pred,
#                          slope_true, slope_pred,
#                          w1, w2, w3):
#     """
#     L = w1 * Σ|y_exp_true - y_exp_pred|
#       + w2 * Σ ( |Cp1_true - Cp1_pred| + |Cp2_true - Cp2_pred| )
#       + w3 * Σ |slope_true - slope_pred|  # 斜率的偏差作为损失项
#     """
#     L_exp = np.sum(np.abs(y_exp_true - y_exp_pred))
#     L_ref = np.sum(np.abs(Cp1_true - Cp1_pred)) + np.sum(np.abs(Cp2_true - Cp2_pred))
#     L_slope = np.sum(np.abs(slope_true - slope_pred))
#     return w1 * L_exp + w2 * L_ref + w3 * L_slope
#
#
# # ===== 8. 内层优化：给定 (w1, w2, w3)，最小化损失，求线性模型参数 =====
# def fit_inner_linear_model(w, X_exp_train, y_exp_train, mat_idx_train,
#                            Cp1_true_train, Cp2_true_train, slope_true_train,
#                            T1_hat_all, T2_hat_all):
#     w1, w2, w3 = w
#     n_feat = X_exp_train.shape[1]
#     theta0 = np.zeros(n_feat + 1)  # 初始全0
#
#     def objective(theta):
#         beta = theta[:-1]  # 回归系数
#         b = theta[-1]  # 偏置项
#
#         # 实验点预测
#         y_pred_train = X_exp_train @ beta + b
#
#         # 参考点预测（正确的计算方式）
#         # 对于每个物质，我们需要计算其在参考温度下的预测值
#         # 首先为每个物质构建参考点特征
#         Cp1_pred_train = np.zeros(len(np.unique(mat_idx_train)))
#         Cp2_pred_train = np.zeros(len(np.unique(mat_idx_train)))
#
#         for i, mat_idx in enumerate(np.unique(mat_idx_train)):
#             # 获取该物质的基团特征
#             Nk = X_groups.iloc[mat_idx].values.astype(float)
#             s_feat = slope_feat_all[mat_idx]
#
#             # 构建 T1 和 T2 的特征向量
#             x_T1 = np.concatenate([Nk, Nk * T1_hat_all[mat_idx], [s_feat * T1_hat_all[mat_idx]]])
#             x_T2 = np.concatenate([Nk, Nk * T2_hat_all[mat_idx], [s_feat * T2_hat_all[mat_idx]]])
#
#             # 预测 Cp1 和 Cp2
#             Cp1_pred_train[i] = x_T1 @ beta + b
#             Cp2_pred_train[i] = x_T2 @ beta + b
#
#         # 斜率预测
#         slope_pred_train = (Cp2_pred_train - Cp1_pred_train) / (T2_hat_all - T1_hat_all)
#
#         return loss_sum_three_parts(
#             y_exp_true=y_exp_train, y_exp_pred=y_pred_train,
#             Cp1_true=Cp1_true_train, Cp2_true=Cp2_true_train,
#             Cp1_pred=Cp1_pred_train, Cp2_pred=Cp2_pred_train,
#             slope_true=slope_true_train, slope_pred=slope_pred_train,
#             w1=w1, w2=w2, w3=w3
#         )
#
#     res = minimize(objective, theta0, method="Powell", options={"maxiter": 100, "xtol": 1e-2, "ftol": 1e-2})
#     return res
#
#
# # ===== 9. 外层：选择最优 (w1, w2, w3) =====
# candidate_ws = sample_weight_triplets(n=40, seed=2025)
#
# best_w = None
# best_r2 = -np.inf
# best_theta = None
#
# for w in candidate_ws:
#     res = fit_inner_linear_model(
#         w=w,
#         X_exp_train=X_exp,
#         y_exp_train=y_exp,
#         mat_idx_train=mat_idx_per_sample,
#         Cp1_true_train=Cp1_true_all,
#         Cp2_true_train=Cp2_true_all,
#         slope_true_train=slope_feat_all,
#         T1_hat_all=T1_hat_all,
#         T2_hat_all=T2_hat_all
#     )
#     theta = res.x
#     beta, b = theta[:-1], theta[-1]
#
#     # 用验证集评估（外层目标）
#     y_val_pred = X_exp @ beta + b
#     r2 = r2_score(y_exp, y_val_pred)
#
#     # 记录最优
#     if r2 > best_r2:
#         best_r2 = r2
#         best_w = w
#         best_theta = theta
#
# print(f"外层最优权重 w* = {best_w}, 验证集 R² = {best_r2:.4f}")
#
# # ===== 10. 用最优权重 w* 在"所有训练样本"上重训 =====
# res_final = fit_inner_linear_model(
#     w=best_w,
#     X_exp_train=X_exp,
#     y_exp_train=y_exp,
#     mat_idx_train=mat_idx_per_sample,
#     Cp1_true_train=Cp1_true_all,
#     Cp2_true_train=Cp2_true_all,
#     slope_true_train=slope_feat_all,
#     T1_hat_all=T1_hat_all,
#     T2_hat_all=T2_hat_all
# )
# theta_final = res_final.x
# beta_final, b_final = theta_final[:-1], theta_final[-1]
#
# # ===== 11. 训练集整体拟合指标 =====
# y_pred_all = X_exp @ beta_final + b_final
# mse_all = mean_squared_error(y_exp, y_pred_all)
# r2_all = r2_score(y_exp, y_pred_all)
#
# rel_err = np.abs((y_pred_all - y_exp) / np.where(np.abs(y_exp) < 1e-12, 1e-12, y_exp)) * 100
# print("\n📊 最终模型（用 w* 内层重训后）")
# print(f"R²  = {r2_all:.4f}")
# print(f"MSE = {mse_all:.4f}")
# print(f"≤1%: {(rel_err <= 1).sum()}, ≤5%: {(rel_err <= 5).sum()}, ≤10%: {(rel_err <= 10).sum()}")
#
# # ===== 12. 导出（可选）=====
# results = pd.DataFrame({
#     "Material_ID": material_ids_all[mat_idx_per_sample],
#     "Temperature (K)": T_per_sample,
#     "Cp_measured": y_exp,
#     "Cp_predicted": y_pred_all
# })
# results.to_excel("Cp预测结果_三项损失_双层优化_分组留出.xlsx", index=False)
#
# feature_labels = list(group_cols) + [f"{g}_T" for g in group_cols] + ["slope×T"]
# coef_df = pd.DataFrame({"Feature": feature_labels, "Contribution": beta_final})
# coef_df.to_excel("Cp系数表_三项损失_双层优化.xlsx", index=False)
#
# print("\n✅ 已保存：Cp预测结果_三项损失_双层优化_分组留出.xlsx")
# print("✅ 已保存：Cp系数表_三项损失_双层优化.xlsx")


import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from scipy.optimize import minimize


# ===== 0. 工具：候选权重采样（外层） =====
def sample_weight_triplets(n=2, seed=2025):  # 减少外层循环次数
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
    n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0  # 简化模型
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


# ===== 7. 损失函数 =====
def loss_sum_three_parts(y_exp_true, y_exp_pred,
                         Cp1_true, Cp2_true,
                         Cp1_pred, Cp2_pred,
                         slope_true, slope_pred,
                         w1, w2, w3):
    L_exp = np.sum(np.abs(y_exp_true - y_exp_pred))
    L_ref = np.sum(np.abs(Cp1_true - Cp1_pred)) + np.sum(np.abs(Cp2_true - Cp2_pred))
    L_slope = np.sum(np.abs(slope_true - slope_pred))
    return w1 * L_exp + w2 * L_ref + w3 * L_slope


# ===== 8. 内层优化：使用L-BFGS-B算法 =====
def fit_inner_linear_model(w, X_exp_train, y_exp_train, mat_idx_train,
                           Cp1_true_train, Cp2_true_train, slope_true_train,
                           T1_hat_all, T2_hat_all):
    w1, w2, w3 = w
    n_feat = X_exp_train.shape[1]
    theta0 = np.zeros(n_feat + 1)

    # 为L-BFGS-B定义边界（可选，可以约束参数范围）
    bounds = [(-10, 10)] * (n_feat + 1)  # 所有参数在-10到10之间

    def objective(theta):
        beta = theta[:-1]
        b = theta[-1]

        # 实验点预测
        y_pred_train = X_exp_train @ beta + b

        # 参考点预测
        unique_materials = np.unique(mat_idx_train)
        Cp1_pred_train = np.zeros(len(unique_materials))
        Cp2_pred_train = np.zeros(len(unique_materials))

        for i, mat_idx in enumerate(unique_materials):
            Nk = X_groups.iloc[mat_idx].values.astype(float)
            s_feat = slope_feat_all[mat_idx]

            x_T1 = np.concatenate([Nk, Nk * T1_hat_all[mat_idx], [s_feat * T1_hat_all[mat_idx]]])
            x_T2 = np.concatenate([Nk, Nk * T2_hat_all[mat_idx], [s_feat * T2_hat_all[mat_idx]]])

            Cp1_pred_train[i] = x_T1 @ beta + b
            Cp2_pred_train[i] = x_T2 @ beta + b

        # 斜率预测
        slope_pred_train = (Cp2_pred_train - Cp1_pred_train) / (
                    T2_hat_all[unique_materials] - T1_hat_all[unique_materials])

        return loss_sum_three_parts(
            y_exp_true=y_exp_train, y_exp_pred=y_pred_train,
            Cp1_true=Cp1_true_train[unique_materials],
            Cp2_true=Cp2_true_train[unique_materials],
            Cp1_pred=Cp1_pred_train, Cp2_pred=Cp2_pred_train,
            slope_true=slope_true_train[unique_materials],
            slope_pred=slope_pred_train,
            w1=w1, w2=w2, w3=w3
        )

    # 使用L-BFGS-B算法（更快！）
    res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-4, "gtol": 1e-4, "disp": True})

    return res


# ===== 9. 外层优化（减少循环次数） =====
candidate_ws = sample_weight_triplets(n=10, seed=2025)  # 只测试10组权重

best_w = None
best_r2 = -np.inf
best_theta = None

for i, w in enumerate(candidate_ws):
    print(f"正在测试第 {i + 1}/10 组权重: {w}")

    res = fit_inner_linear_model(
        w=w,
        X_exp_train=X_exp,
        y_exp_train=y_exp,
        mat_idx_train=mat_idx_per_sample,
        Cp1_true_train=Cp1_true_all,
        Cp2_true_train=Cp2_true_all,
        slope_true_train=slope_feat_all,
        T1_hat_all=T1_hat_all,
        T2_hat_all=T2_hat_all
    )

    if not res.success:
        print(f"警告: 第 {i + 1} 组权重优化失败: {res.message}")
        continue

    theta = res.x
    beta, b = theta[:-1], theta[-1]

    y_val_pred = X_exp @ beta + b
    r2 = r2_score(y_exp, y_val_pred)

    print(f"权重 {w} -> R² = {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_w = w
        best_theta = theta

print(f"外层最优权重 w* = {best_w}, 验证集 R² = {best_r2:.4f}")

# ===== 10. 用最优权重重训 =====
print("使用最优权重进行最终训练...")
res_final = fit_inner_linear_model(
    w=best_w,
    X_exp_train=X_exp,
    y_exp_train=y_exp,
    mat_idx_train=mat_idx_per_sample,
    Cp1_true_train=Cp1_true_all,
    Cp2_true_train=Cp2_true_all,
    slope_true_train=slope_feat_all,
    T1_hat_all=T1_hat_all,
    T2_hat_all=T2_hat_all
)

theta_final = res_final.x
beta_final, b_final = theta_final[:-1], theta_final[-1]

# ===== 11. 评估结果 =====
y_pred_all = X_exp @ beta_final + b_final
mse_all = mean_squared_error(y_exp, y_pred_all)
r2_all = r2_score(y_exp, y_pred_all)

rel_err = np.abs((y_pred_all - y_exp) / np.where(np.abs(y_exp) < 1e-12, 1e-12, y_exp)) * 100
print("\n📊 最终模型结果")
print(f"R²  = {r2_all:.4f}")
print(f"MSE = {mse_all:.4f}")
print(f"≤1%: {(rel_err <= 1).sum()}, ≤5%: {(rel_err <= 5).sum()}, ≤10%: {(rel_err <= 10).sum()}")

# ===== 12. 导出结果 =====
results = pd.DataFrame({
    "Material_ID": material_ids_all[mat_idx_per_sample],
    "Temperature (K)": T_per_sample,
    "Cp_measured": y_exp,
    "Cp_predicted": y_pred_all
})
results.to_excel("Cp预测结果_优化后.xlsx", index=False)

feature_labels = list(group_cols) + [f"{g}_T" for g in group_cols] + ["slope×T"]
coef_df = pd.DataFrame({"Feature": feature_labels, "Contribution": beta_final})
coef_df.to_excel("Cp系数表_优化后.xlsx", index=False)

print("\n✅ 完成！")