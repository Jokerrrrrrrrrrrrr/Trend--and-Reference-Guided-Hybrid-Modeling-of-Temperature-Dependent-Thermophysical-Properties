
#slow
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import HuberRegressor
# from scipy.optimize import minimize
#
#
# # ===== 0. 工具：候选权重采样（外层） =====
# def sample_weight_triplets(n=2, seed=2025):
#     """
#     采样 n 组 (w1, w2, w3)，非负且和为1（避免全0退化）。
#     用 Dirichlet(1,1,1) 随机权重，覆盖面更均匀。
#     """
#     rng = np.random.default_rng(seed)
#     W = rng.dirichlet([1.0, 1.0, 1.0], size=n)
#     return W
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
# y_T1 = df.loc[valid_mask, target_column_T1].values
# T1_model = GradientBoostingRegressor(
#     n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0  # 简化模型
# ).fit(X_poly, y_T1)
#
# Cp1_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 9])
# Cp2_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 50])
#
# X_poly_all = poly.transform(X_groups)
# T1_hat_all = T1_model.predict(X_poly_all)
# T2_hat_all = 1.5 * T1_hat_all
#
# Cp1_pred_all = Cp1_model.predict(X_groups)
# Cp2_pred_all = Cp2_model.predict(X_groups)
#
# # ===== 4. 计算斜率 =====
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_pred_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# # ===== 5. 真实参考点值 =====
# Cp1_true_all = df.iloc[:, 9].astype(float).values
# Cp2_true_all = df.iloc[:, 50].astype(float).values
# T1_true_all = df[target_column_T1].astype(float).values
# T2_true_all = 1.5 * T1_true_all
#
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_true_all = (Cp2_true_all - Cp1_true_all) / (T2_true_all - T1_true_all)
#
# # ===== 6. 构建实验点样本 =====
# slope_feat_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# X_exp_list, y_exp_list, mat_idx_list, T_list = [], [], [], []
# for i in range(len(df)):
#     Nk = X_groups.iloc[i].values.astype(float)
#     s_feat = slope_feat_all[i]
#     temps_i = df.loc[df.index[i], temp_cols].values.astype(float)
#     cps_i = df.loc[df.index[i], cp_cols].values.astype(float)
#
#     for T, Cp in zip(temps_i, cps_i):
#         if not (np.isfinite(T) and np.isfinite(Cp)):
#             continue
#         x = np.concatenate([Nk, Nk * T, [s_feat * T]])
#         X_exp_list.append(x)
#         y_exp_list.append(Cp)
#         mat_idx_list.append(i)
#         T_list.append(T)
#
# X_exp = np.asarray(X_exp_list)
# y_exp = np.asarray(y_exp_list)
# mat_idx_per_sample = np.asarray(mat_idx_list)
# T_per_sample = np.asarray(T_list)
#
#
# # ===== 7. Huber 损失函数 =====
# def huber_loss(y_true, y_pred, delta=1.0):
#     """计算 Huber 损失"""
#     error = np.abs(y_true - y_pred)
#     loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
#     return np.sum(loss)
#
#
# # ===== 8. 修正：基于反比例的自适应权重计算 =====
# def calculate_adaptive_weights_inverse(X_exp, y_exp, Cp1_true, Cp2_true, slope_true):
#     """计算每个损失项的数量级并使用反比例调整权重"""
#     L_exp_typical = np.mean(np.abs(y_exp - np.mean(y_exp)))  # 实验点的数量级
#     L_ref_typical = (np.mean(np.abs(Cp1_true - np.mean(Cp1_true))) +
#                      np.mean(np.abs(Cp2_true - np.mean(Cp2_true)))) / 2  # 参考点的数量级
#     L_slope_typical = np.mean(np.abs(slope_true - np.mean(slope_true)))  # 斜率的数量级
#
#     L_exp_typical = max(L_exp_typical, 1e-10)
#     L_ref_typical = max(L_ref_typical, 1e-10)
#     L_slope_typical = max(L_slope_typical, 1e-10)
#
#     print(f"损失项典型值: 实验点={L_exp_typical:.2f}, 参考点={L_ref_typical:.2f}, 斜率={L_slope_typical:.6f}")
#
#     # 使用反比例关系：数值越小，权重应该越大（放大作用）
#     w1_base = 1.0/ L_exp_typical  # 实验点数值大，权重小
#     w2_base = 1.0 / L_ref_typical  # 参考点数值大，权重小
#     w3_base = 1.0 / L_slope_typical  # 斜率数值小，权重大（放大！）
#
#     # 归一化
#     total_base = w1_base + w2_base + w3_base
#     w1_normalized = w1_base / total_base
#     w2_normalized = w2_base / total_base
#     w3_normalized = w3_base / total_base
#
#     return w1_normalized, w2_normalized, w3_normalized
#
#
# # ===== 8.1 计算损失倍数（用于损失函数内部放大） =====
# def calculate_loss_multipliers(L_exp, L_ref, L_slope, max_multiplier=1000):
#     """计算损失放大倍数"""
#     max_loss = max(L_exp, L_ref, L_slope)
#
#     # 计算需要放大的倍数
#     multiplier_exp = max_loss / L_exp
#     multiplier_ref = max_loss / L_ref
#     multiplier_slope = max_loss / L_slope
#
#     # 限制最大倍数，避免极端值
#     multiplier_exp = min(multiplier_exp, max_multiplier)
#     multiplier_ref = min(multiplier_ref, max_multiplier)
#     multiplier_slope = min(multiplier_slope, max_multiplier)
#
#     return multiplier_exp, multiplier_ref, multiplier_slope
#
#
# # ===== 7.1 修正：使用放大倍数的损失函数 =====
# def loss_sum_three_parts_with_multipliers(y_exp_true, y_exp_pred,
#                                           Cp1_true, Cp2_true,
#                                           Cp1_pred, Cp2_pred,
#                                           slope_true, slope_pred,
#                                           w1, w2, w3):
#     """使用放大倍数后的损失函数"""
#     L_exp = np.sum(np.abs(y_exp_true - y_exp_pred)) * multiplier_exp
#     L_ref = (np.sum(np.abs(Cp1_true - Cp1_pred)) +
#              np.sum(np.abs(Cp2_true - Cp2_pred))) * multiplier_ref
#     L_slope = np.sum(np.abs(slope_true - slope_pred)) * multiplier_slope
#
#     return w1 * L_exp + w2 * L_ref + w3 * L_slope
#
#
# # 计算反比例基准权重
# base_w1, base_w2, base_w3 = calculate_adaptive_weights_inverse(
#     X_exp, y_exp, Cp1_true_all, Cp2_true_all, slope_true_all
# )
#
# # 计算损失放大倍数
# L_exp, L_ref, L_slope = (
#     np.mean(np.abs(y_exp - np.mean(y_exp))),
#     (np.mean(np.abs(Cp1_true_all - np.mean(Cp1_true_all))) +
#      np.mean(np.abs(Cp2_true_all - np.mean(Cp2_true_all)))) / 2,
#     np.mean(np.abs(slope_true_all - np.mean(slope_true_all)))
# )
# multiplier_exp, multiplier_ref, multiplier_slope = calculate_loss_multipliers(L_exp, L_ref, L_slope)
#
#
# # ===== 9. 外层优化（随机采样并结合自适应基准权重） =====
# candidate_ws = sample_weight_triplets(n=10, seed=2025)  # 只测试10组权重
#
# best_w = None
# best_r2 = -np.inf
# best_theta = None
#
#
# # ===== 9.1 内层优化函数定义 =====
# def fit_inner_linear_model(w, X_exp_train, y_exp_train, mat_idx_train,
#                            Cp1_true_train, Cp2_true_train, slope_true_train,
#                            T1_hat_all, T2_hat_all):
#     w1, w2, w3 = w
#     n_feat = X_exp_train.shape[1]
#     theta0 = np.zeros(n_feat + 1)  # 初始全0
#
#     def objective(theta):
#         beta = theta[:-1]
#         b = theta[-1]
#
#         # 实验点预测
#         y_pred_train = X_exp_train @ beta + b
#
#         # 参考点预测（正确的计算方式）
#         unique_materials = np.unique(mat_idx_train)
#         Cp1_pred_train = np.zeros(len(unique_materials))
#         Cp2_pred_train = np.zeros(len(unique_materials))
#
#         for i, mat_idx in enumerate(unique_materials):
#             Nk = X_groups.iloc[mat_idx].values.astype(float)
#             s_feat = slope_feat_all[mat_idx]
#
#             x_T1 = np.concatenate([Nk, Nk * T1_hat_all[mat_idx], [s_feat * T1_hat_all[mat_idx]]])
#             x_T2 = np.concatenate([Nk, Nk * T2_hat_all[mat_idx], [s_feat * T2_hat_all[mat_idx]]])
#
#             Cp1_pred_train[i] = x_T1 @ beta + b
#             Cp2_pred_train[i] = x_T2 @ beta + b
#
#         # 斜率预测
#         slope_pred_train = (Cp2_pred_train - Cp1_pred_train) / (
#                 T2_hat_all[unique_materials] - T1_hat_all[unique_materials])
#
#         # 使用带放大倍数的损失函数
#         return loss_sum_three_parts_with_multipliers(
#             y_exp_true=y_exp_train, y_exp_pred=y_pred_train,
#             Cp1_true=Cp1_true_train[unique_materials],
#             Cp2_true=Cp2_true_train[unique_materials],
#             Cp1_pred=Cp1_pred_train, Cp2_pred=Cp2_pred_train,
#             slope_true=slope_true_train[unique_materials],
#             slope_pred=slope_pred_train,
#             w1=w1, w2=w2, w3=w3
#         )
#
#     res = minimize(objective, theta0, method="Powell", options={"maxiter": 5000, "xtol": 1e-6, "ftol": 1e-6})
#     return res
#
#
# # ===== 9.2 外层优化循环 =====
# for i, w in enumerate(candidate_ws):
#     # 使用幂函数进一步放大斜率的重要性（如果基准权重显示斜率很重要）
#     # 斜率基准权重越大，说明越需要重视，进一步放大
#     slope_emphasis = base_w3 ** 0.5  # 开平方根，避免过度放大
#
#     adjusted_w = [
#         w[0] * 100 * base_w1,
#         w[1] * base_w2,
#         w[2] * base_w3  # 额外放大斜率权重
#     ]
#
#     # 归一化调整后的权重
#     total_w = sum(adjusted_w)
#     adjusted_w = [w / total_w for w in adjusted_w]
#
#     print(f"\n第{i + 1}组调整后权重: {[f'{x:.6f}' for x in adjusted_w]}")
#
#     res = fit_inner_linear_model(
#         w=adjusted_w,
#         X_exp_train=X_exp,
#         y_exp_train=y_exp,
#         mat_idx_train=mat_idx_per_sample,
#         Cp1_true_train=Cp1_true_all,
#         Cp2_true_train=Cp2_true_all,
#         slope_true_train=slope_true_all,  # 使用真实斜率而不是预测斜率
#         T1_hat_all=T1_hat_all,
#         T2_hat_all=T2_hat_all
#     )
#
#     if not res.success:
#         print(f"警告: 第 {i + 1} 组权重优化失败: {res.message}")
#         continue
#
#     theta = res.x
#     beta, b = theta[:-1], theta[-1]
#
#     y_val_pred = X_exp @ beta + b
#     r2 = r2_score(y_exp, y_val_pred)
#
#     print(f"权重 {[f'{x:.6f}' for x in adjusted_w]} -> R² = {r2:.6f}")
#
#     if r2 > best_r2:
#         best_r2 = r2
#         best_w = adjusted_w
#         best_theta = theta
#
# print(f"\n外层最优权重 w* = {best_w}, 验证集 R² = {best_r2:.6f}")
#
# # ===== 10. 用最优权重重训 =====
# print("\n使用最优权重进行最终训练...")
# res_final = fit_inner_linear_model(
#     w=best_w,
#     X_exp_train=X_exp,
#     y_exp_train=y_exp,
#     mat_idx_train=mat_idx_per_sample,
#     Cp1_true_train=Cp1_true_all,
#     Cp2_true_train=Cp2_true_all,
#     slope_true_train=slope_true_all,
#     T1_hat_all=T1_hat_all,
#     T2_hat_all=T2_hat_all
# )
#
# theta_final = res_final.x
# beta_final, b_final = theta_final[:-1], theta_final[-1]
#
# # ===== 11. 评估结果 =====
# y_pred_all = X_exp @ beta_final + b_final
# mse_all = mean_squared_error(y_exp, y_pred_all)
# r2_all = r2_score(y_exp, y_pred_all)
#
# rel_err = np.abs((y_pred_all - y_exp) / np.where(np.abs(y_exp) < 1e-12, 1e-12, y_exp)) * 100
#
# # ===== 12. 导出结果 =====
# results = pd.DataFrame({
#     "Material_ID": material_ids_all[mat_idx_per_sample],
#     "Temperature (K)": T_per_sample,
#     "Cp_measured": y_exp,
#     "Cp_predicted": y_pred_all,
#     "Relative_Error_%": rel_err
# })
# results.to_excel("Cp预测结果_优化后.xlsx", index=False)
#
# feature_labels = list(group_cols) + [f"{g}_T" for g in group_cols] + ["slope×T"]
# coef_df = pd.DataFrame({"Feature": feature_labels, "Contribution": beta_final})
# coef_df.to_excel("Cp系数表_优化后.xlsx", index=False)
#
# # ===== 13. 最终结果汇总 =====
# print("\n" + "=" * 60)
# print("🎯 最终优化结果汇总")
# print("=" * 60)
# print(f"最优权重组合: w1={best_w[0]:.6f}, w2={best_w[1]:.6f}, w3={best_w[2]:.6f}")
# print(f"最终模型性能:")
# print(f"  R²  = {r2_all:.6f}")
# print(f"  MSE = {mse_all:.6f}")
# print(f"  ≤1%: {(rel_err <= 1).sum()}/{len(rel_err)} ({(rel_err <= 1).sum() / len(rel_err) * 100:.2f}%)")
# print(f"  ≤5%: {(rel_err <= 5).sum()}/{len(rel_err)} ({(rel_err <= 5).sum() / len(rel_err) * 100:.2f}%)")
# print(f"  ≤10%: {(rel_err <= 10).sum()}/{len(rel_err)} ({(rel_err <= 10).sum() / len(rel_err) * 100:.2f}%)")
#
# # 计算平均相对误差
# mean_rel_err = np.mean(rel_err)
# median_rel_err = np.median(rel_err)
# print(f"  平均相对误差: {mean_rel_err:.2f}%")
# print(f"  中位数相对误差: {median_rel_err:.2f}%")
#
# # 计算R²_adjusted
# n_samples = len(y_exp)
# n_features = X_exp.shape[1]
# r2_adjusted = 1 - (1 - r2_all) * (n_samples - 1) / (n_samples - n_features - 1)
# print(f"  调整后R²: {r2_adjusted:.6f}")
#
# print("=" * 60)
# print("✅ 完成！预测结果和系数表已保存到Excel文件")


# very slow
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import HuberRegressor
# from scipy.optimize import minimize
#
#
# # ===== 0. 工具：候选权重采样（外层） =====
# def sample_weight_triplets(n=2, seed=2025):
#     """
#     采样 n 组 (w1, w2, w3)，非负且和为1（避免全0退化）。
#     用 Dirichlet(1,1,1) 随机权重，覆盖面更均匀。
#     """
#     rng = np.random.default_rng(seed)
#     W = rng.dirichlet([1.0, 1.0, 1.0], size=n)
#     return W
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
# y_T1 = df.loc[valid_mask, target_column_T1].values
# T1_model = GradientBoostingRegressor(
#     n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0  # 简化模型
# ).fit(X_poly, y_T1)
#
# Cp1_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 9])
# Cp2_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 50])
#
# X_poly_all = poly.transform(X_groups)
# T1_hat_all = T1_model.predict(X_poly_all)
# T2_hat_all = 1.5 * T1_hat_all
#
# Cp1_pred_all = Cp1_model.predict(X_groups)
# Cp2_pred_all = Cp2_model.predict(X_groups)
#
# # ===== 4. 计算斜率 =====
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_pred_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# # ===== 5. 真实参考点值 =====
# Cp1_true_all = df.iloc[:, 9].astype(float).values
# Cp2_true_all = df.iloc[:, 50].astype(float).values
# T1_true_all = df[target_column_T1].astype(float).values
# T2_true_all = 1.5 * T1_true_all
#
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_true_all = (Cp2_true_all - Cp1_true_all) / (T2_true_all - T1_true_all)
#
# # ===== 6. 构建实验点样本 =====
# slope_feat_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# X_exp_list, y_exp_list, mat_idx_list, T_list = [], [], [], []
# for i in range(len(df)):
#     Nk = X_groups.iloc[i].values.astype(float)
#     s_feat = slope_feat_all[i]
#     temps_i = df.loc[df.index[i], temp_cols].values.astype(float)
#     cps_i = df.loc[df.index[i], cp_cols].values.astype(float)
#
#     for T, Cp in zip(temps_i, cps_i):
#         if not (np.isfinite(T) and np.isfinite(Cp)):
#             continue
#         x = np.concatenate([Nk, Nk * T, [s_feat * T]])
#         X_exp_list.append(x)
#         y_exp_list.append(Cp)
#         mat_idx_list.append(i)
#         T_list.append(T)
#
# X_exp = np.asarray(X_exp_list)
# y_exp = np.asarray(y_exp_list)
# mat_idx_per_sample = np.asarray(mat_idx_list)
# T_per_sample = np.asarray(T_list)
#
#
# # ===== 7. Huber 损失函数 =====
# def huber_loss(y_true, y_pred, delta=1.0):
#     """计算 Huber 损失"""
#     error = np.abs(y_true - y_pred)
#     loss = np.where(error <= delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))
#     return np.sum(loss)
#
#
# # ===== 7.1 修改：三项损失函数，使用 Huber 损失 =====
# def loss_sum_three_parts_with_huber(y_exp_true, y_exp_pred,
#                                     Cp1_true, Cp2_true,
#                                     Cp1_pred, Cp2_pred,
#                                     slope_true, slope_pred,
#                                     w1, w2, w3, delta=1.0):
#     # 使用 Huber 损失计算各个项的损失
#     L_exp = huber_loss(y_exp_true, y_exp_pred, delta)
#     L_ref = huber_loss(Cp1_true, Cp1_pred, delta) + huber_loss(Cp2_true, Cp2_pred, delta)
#     L_slope = huber_loss(slope_true, slope_pred, delta)
#
#     # 总损失是加权的三项损失
#     return w1 * L_exp + w2 * L_ref + w3 * L_slope
#
#
# # ===== 9. 内层优化函数：目标函数 =====
# def fit_inner_linear_model_with_huber(w, X_exp_train, y_exp_train, mat_idx_train,
#                                       Cp1_true_train, Cp2_true_train, slope_true_train,
#                                       T1_hat_all, T2_hat_all, delta=1.0):
#     w1, w2, w3 = w
#     n_feat = X_exp_train.shape[1]
#     theta0 = np.zeros(n_feat + 1)  # 初始全0
#
#     def objective(theta):
#         beta = theta[:-1]
#         b = theta[-1]
#
#         # 实验点预测
#         y_pred_train = X_exp_train @ beta + b
#
#         # 参考点预测（正确的计算方式）
#         unique_materials = np.unique(mat_idx_train)
#         Cp1_pred_train = np.zeros(len(unique_materials))
#         Cp2_pred_train = np.zeros(len(unique_materials))
#
#         for i, mat_idx in enumerate(unique_materials):
#             Nk = X_groups.iloc[mat_idx].values.astype(float)
#             s_feat = slope_feat_all[mat_idx]
#
#             x_T1 = np.concatenate([Nk, Nk * T1_hat_all[mat_idx], [s_feat * T1_hat_all[mat_idx]]])
#             x_T2 = np.concatenate([Nk, Nk * T2_hat_all[mat_idx], [s_feat * T2_hat_all[mat_idx]]])
#
#             Cp1_pred_train[i] = x_T1 @ beta + b
#             Cp2_pred_train[i] = x_T2 @ beta + b
#
#         # 斜率预测
#         slope_pred_train = (Cp2_pred_train - Cp1_pred_train) / (
#                     T2_hat_all[unique_materials] - T1_hat_all[unique_materials])
#
#         # 使用带 Huber 损失的三项损失
#         return loss_sum_three_parts_with_huber(
#             y_exp_true=y_exp_train, y_exp_pred=y_pred_train,
#             Cp1_true=Cp1_true_train[unique_materials],
#             Cp2_true=Cp2_true_train[unique_materials],
#             Cp1_pred=Cp1_pred_train, Cp2_pred=Cp2_pred_train,
#             slope_true=slope_true_train[unique_materials],
#             slope_pred=slope_pred_train,
#             w1=w1, w2=w2, w3=w3, delta=delta
#         )
#
#     res = minimize(objective, theta0, method="Powell", options={"maxiter": 5000, "xtol": 1e-6, "ftol": 1e-6})
#     return res
#
#
# # ===== 9.2 外层优化循环 =====
# candidate_ws = sample_weight_triplets(n=100, seed=2025)  # 只测试10组权重
#
# best_w = None
# best_r2 = -np.inf
# best_theta = None
#
# for i, w in enumerate(candidate_ws):
#     # 归一化调整后的权重
#     total_w = sum(w)
#     adjusted_w = [wi / total_w for wi in w]
#
#     print(f"\n第{i + 1}组调整后权重: {[f'{x:.6f}' for x in adjusted_w]}")
#
#     res = fit_inner_linear_model_with_huber(
#         w=adjusted_w,
#         X_exp_train=X_exp,
#         y_exp_train=y_exp,
#         mat_idx_train=mat_idx_per_sample,
#         Cp1_true_train=Cp1_true_all,
#         Cp2_true_train=Cp2_true_all,
#         slope_true_train=slope_true_all,  # 使用真实斜率而不是预测斜率
#         T1_hat_all=T1_hat_all,
#         T2_hat_all=T2_hat_all,
#         delta=1.0  # 可以调整delta值
#     )
#
#     if not res.success:
#         print(f"警告: 第 {i + 1} 组权重优化失败: {res.message}")
#         continue
#
#     theta = res.x
#     beta, b = theta[:-1], theta[-1]
#
#     y_val_pred = X_exp @ beta + b
#     r2 = r2_score(y_exp, y_val_pred)
#
#     print(f"权重 {[f'{x:.6f}' for x in adjusted_w]} -> R² = {r2:.6f}")
#
#     if r2 > best_r2:
#         best_r2 = r2
#         best_w = adjusted_w
#         best_theta = theta
#
# print(f"\n外层最优权重 w* = {best_w}, 验证集 R² = {best_r2:.6f}")
#
# # ===== 10. 用最优权重重训 =====
# print("\n使用最优权重进行最终训练...")
# res_final = fit_inner_linear_model_with_huber(
#     w=best_w,
#     X_exp_train=X_exp,
#     y_exp_train=y_exp,
#     mat_idx_train=mat_idx_per_sample,
#     Cp1_true_train=Cp1_true_all,
#     Cp2_true_train=Cp2_true_all,
#     slope_true_train=slope_true_all,
#     T1_hat_all=T1_hat_all,
#     T2_hat_all=T2_hat_all,
#     delta=1.0
# )
#
# theta_final = res_final.x
# beta_final, b_final = theta_final[:-1], theta_final[-1]
#
# # ===== 11. 评估结果 =====
# y_pred_all = X_exp @ beta_final + b_final
# mse_all = mean_squared_error(y_exp, y_pred_all)
# r2_all = r2_score(y_exp, y_pred_all)
#
# rel_err = np.abs((y_pred_all - y_exp) / np.where(np.abs(y_exp) < 1e-12, 1e-12, y_exp)) * 100
#
# # ===== 12. 导出结果 =====
# results = pd.DataFrame({
#     "Material_ID": material_ids_all[mat_idx_per_sample],
#     "Temperature (K)": T_per_sample,
#     "Cp_measured": y_exp,
#     "Cp_predicted": y_pred_all,
#     "Relative_Error_%": rel_err
# })
# results.to_excel("Cp预测结果_优化后.xlsx", index=False)
#
# feature_labels = list(group_cols) + [f"{g}_T" for g in group_cols] + ["slope×T"]
# coef_df = pd.DataFrame({"Feature": feature_labels, "Contribution": beta_final})
# coef_df.to_excel("Cp系数表_优化后.xlsx", index=False)
#
# # ===== 13. 最终结果汇总 =====
# print("\n" + "=" * 60)
# print("🎯 最终优化结果汇总")
# print("=" * 60)
# print(f"最优权重组合: w1={best_w[0]:.6f}, w2={best_w[1]:.6f}, w3={best_w[2]:.6f}")
# print(f"最终模型性能:")
# print(f"  R²  = {r2_all:.6f}")
# print(f"  MSE = {mse_all:.6f}")
# print(f"  ≤1%: {(rel_err <= 1).sum()}/{len(rel_err)} ({(rel_err <= 1).sum() / len(rel_err) * 100:.2f}%)")
# print(f"  ≤5%: {(rel_err <= 5).sum()}/{len(rel_err)} ({(rel_err <= 5).sum() / len(rel_err) * 100:.2f}%)")
# print(f"  ≤10%: {(rel_err <= 10).sum()}/{len(rel_err)} ({(rel_err <= 10).sum() / len(rel_err) * 100:.2f}%)")
#
# # 计算平均相对误差
# mean_rel_err = np.mean(rel_err)
# median_rel_err = np.median(rel_err)
# print(f"  平均相对误差: {mean_rel_err:.2f}%")
# print(f"  中位数相对误差: {median_rel_err:.2f}%")
#
# # 计算R²_adjusted
# n_samples = len(y_exp)
# n_features = X_exp.shape[1]
# r2_adjusted = 1 - (1 - r2_all) * (n_samples - 1) / (n_samples - n_features - 1)
# print(f"  调整后R²: {r2_adjusted:.6f}")
#
# print("=" * 60)
# print("✅ 完成！预测结果和系数表已保存到Excel文件")

# huber with error
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import HuberRegressor
# from scipy.optimize import minimize
#
#
# # ===== 0. 工具：候选权重采样（外层） =====
# def sample_weight_triplets(n=2, seed=2025):
#     """
#     采样 n 组 (w1, w2, w3)，非负且和为1（避免全0退化）。
#     用 Dirichlet(1,1,1) 随机权重，覆盖面更均匀。
#     """
#     rng = np.random.default_rng(seed)
#     W = rng.dirichlet([1.0, 1.0, 1.0], size=n)
#     return W
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
# y_T1 = df.loc[valid_mask, target_column_T1].values
# T1_model = GradientBoostingRegressor(
#     n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0  # 简化模型
# ).fit(X_poly, y_T1)
#
# Cp1_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 9])
# Cp2_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 50])
#
# X_poly_all = poly.transform(X_groups)
# T1_hat_all = T1_model.predict(X_poly_all)
# T2_hat_all = 1.5 * T1_hat_all
#
# Cp1_pred_all = Cp1_model.predict(X_groups)
# Cp2_pred_all = Cp2_model.predict(X_groups)
#
# # ===== 4. 计算斜率 =====
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_pred_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# # ===== 5. 真实参考点值 =====
# Cp1_true_all = df.iloc[:, 9].astype(float).values
# Cp2_true_all = df.iloc[:, 50].astype(float).values
# T1_true_all = df[target_column_T1].astype(float).values
# T2_true_all = 1.5 * T1_true_all
#
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_true_all = (Cp2_true_all - Cp1_true_all) / (T2_true_all - T1_true_all)
#
# # ===== 6. 构建实验点样本 =====
# slope_feat_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# X_exp_list, y_exp_list, mat_idx_list, T_list = [], [], [], []
# for i in range(len(df)):
#     Nk = X_groups.iloc[i].values.astype(float)
#     s_feat = slope_feat_all[i]
#     temps_i = df.loc[df.index[i], temp_cols].values.astype(float)
#     cps_i = df.loc[df.index[i], cp_cols].values.astype(float)
#
#     for T, Cp in zip(temps_i, cps_i):
#         if not (np.isfinite(T) and np.isfinite(Cp)):
#             continue
#         x = np.concatenate([Nk, Nk * T, [s_feat * T]])
#         X_exp_list.append(x)
#         y_exp_list.append(Cp)
#         mat_idx_list.append(i)
#         T_list.append(T)
#
# X_exp = np.asarray(X_exp_list)
# y_exp = np.asarray(y_exp_list)
# mat_idx_per_sample = np.asarray(mat_idx_list)
# T_per_sample = np.asarray(T_list)
#
#
# # ===== 7. Huber 损失函数 =====
# def huber_loss(y_true, y_pred, delta=1.0):
#     """计算 Huber 损失"""
#     error = np.abs(y_true - y_pred)
#     loss = np.where(error <= delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))
#     return np.sum(loss)
#
#
# # ===== 8. 修正：基于反比例的自适应权重计算 =====
# def calculate_adaptive_weights_inverse(X_exp, y_exp, Cp1_true, Cp2_true, slope_true):
#     """计算每个损失项的数量级并使用反比例调整权重"""
#     L_exp_typical = np.mean(np.abs(y_exp - np.mean(y_exp)))  # 实验点的数量级
#     L_ref_typical = (np.mean(np.abs(Cp1_true - np.mean(Cp1_true))) +
#                      np.mean(np.abs(Cp2_true - np.mean(Cp2_true)))) / 2  # 参考点的数量级
#     L_slope_typical = np.mean(np.abs(slope_true - np.mean(slope_true)))  # 斜率的数量级
#
#     L_exp_typical = max(L_exp_typical, 1e-10)
#     L_ref_typical = max(L_ref_typical, 1e-10)
#     L_slope_typical = max(L_slope_typical, 1e-10)
#
#     print(f"损失项典型值: 实验点={L_exp_typical:.2f}, 参考点={L_ref_typical:.2f}, 斜率={L_slope_typical:.6f}")
#
#     # 使用反比例关系：数值越小，权重应该越大（放大作用）
#     w1_base = 1.0 / L_exp_typical  # 实验点数值大，权重小
#     w2_base = 1.0 / L_ref_typical  # 参考点数值大，权重小
#     w3_base = 1.0 / L_slope_typical  # 斜率数值小，权重大（放大！）
#
#     # 归一化
#     total_base = w1_base + w2_base + w3_base
#     w1_normalized = w1_base / total_base
#     w2_normalized = w2_base / total_base
#     w3_normalized = w3_base / total_base
#
#     return w1_normalized, w2_normalized, w3_normalized
#
#
# # ===== 8.1 计算损失倍数（用于损失函数内部放大） =====
# def calculate_loss_multipliers(L_exp, L_ref, L_slope, max_multiplier=1000):
#     """计算损失放大倍数"""
#     max_loss = max(L_exp, L_ref, L_slope)
#
#     # 计算需要放大的倍数
#     multiplier_exp = max_loss / L_exp
#     multiplier_ref = max_loss / L_ref
#     multiplier_slope = max_loss / L_slope
#
#     # 限制最大倍数，避免极端值
#     multiplier_exp = min(multiplier_exp, max_multiplier)
#     multiplier_ref = min(multiplier_ref, max_multiplier)
#     multiplier_slope = min(multiplier_slope, max_multiplier)
#
#     return multiplier_exp, multiplier_ref, multiplier_slope
#
#
# # ===== 7.1 修正：使用放大倍数的损失函数 =====
# def loss_sum_three_parts_with_multipliers(y_exp_true, y_exp_pred,
#                                           Cp1_true, Cp2_true,
#                                           Cp1_pred, Cp2_pred,
#                                           slope_true, slope_pred,
#                                           w1, w2, w3):
#     """使用放大倍数后的损失函数"""
#     L_exp = np.sum(np.abs(y_exp_true - y_exp_pred)) * multiplier_exp
#     L_ref = (np.sum(np.abs(Cp1_true - Cp1_pred)) +
#              np.sum(np.abs(Cp2_true - Cp2_pred))) * multiplier_ref
#     L_slope = np.sum(np.abs(slope_true - slope_pred)) * multiplier_slope
#
#     return w1 * L_exp + w2 * L_ref + w3 * L_slope
#
#
# # 计算反比例基准权重
# base_w1, base_w2, base_w3 = calculate_adaptive_weights_inverse(
#     X_exp, y_exp, Cp1_true_all, Cp2_true_all, slope_true_all
# )
#
# # 计算损失放大倍数
# L_exp, L_ref, L_slope = (
#     np.mean(np.abs(y_exp - np.mean(y_exp))),
#     (np.mean(np.abs(Cp1_true_all - np.mean(Cp1_true_all))) +
#      np.mean(np.abs(Cp2_true_all - np.mean(Cp2_true_all)))) / 2,
#     np.mean(np.abs(slope_true_all - np.mean(slope_true_all)))
# )
# multiplier_exp, multiplier_ref, multiplier_slope = calculate_loss_multipliers(L_exp, L_ref, L_slope)
#
# # ===== 9. 外层优化（随机采样并结合自适应基准权重） =====
# candidate_ws = sample_weight_triplets(n=100, seed=2025)  # 只测试10组权重
#
# best_w = None
# best_r2 = -np.inf
# best_theta = None
#
#
# # ===== 9.1 内层优化函数定义 =====
# def fit_inner_huber_model(w, X_exp_train, y_exp_train, mat_idx_train,
#                           Cp1_true_train, Cp2_true_train, slope_true_train,
#                           T1_hat_all, T2_hat_all):
#     w1, w2, w3 = w
#
#     # 使用Huber回归拟合主模型
#     huber_model = HuberRegressor(max_iter=1000, epsilon=1.35, alpha=0.0001)
#     huber_model.fit(X_exp_train, y_exp_train)
#
#     beta = huber_model.coef_
#     b = huber_model.intercept_
#
#     # 计算各项损失
#     # 实验点预测
#     y_pred_train = X_exp_train @ beta + b
#
#     # 参考点预测
#     unique_materials = np.unique(mat_idx_train)
#     Cp1_pred_train = np.zeros(len(unique_materials))
#     Cp2_pred_train = np.zeros(len(unique_materials))
#
#     for i, mat_idx in enumerate(unique_materials):
#         Nk = X_groups.iloc[mat_idx].values.astype(float)
#         s_feat = slope_feat_all[mat_idx]
#
#         x_T1 = np.concatenate([Nk, Nk * T1_hat_all[mat_idx], [s_feat * T1_hat_all[mat_idx]]])
#         x_T2 = np.concatenate([Nk, Nk * T2_hat_all[mat_idx], [s_feat * T2_hat_all[mat_idx]]])
#
#         Cp1_pred_train[i] = x_T1 @ beta + b
#         Cp2_pred_train[i] = x_T2 @ beta + b
#
#     # 斜率预测
#     slope_pred_train = (Cp2_pred_train - Cp1_pred_train) / (
#             T2_hat_all[unique_materials] - T1_hat_all[unique_materials])
#
#     # 计算总损失
#     total_loss = loss_sum_three_parts_with_multipliers(
#         y_exp_true=y_exp_train, y_exp_pred=y_pred_train,
#         Cp1_true=Cp1_true_train[unique_materials],
#         Cp2_true=Cp2_true_train[unique_materials],
#         Cp1_pred=Cp1_pred_train, Cp2_pred=Cp2_pred_train,
#         slope_true=slope_true_train[unique_materials],
#         slope_pred=slope_pred_train,
#         w1=w1, w2=w2, w3=w3
#     )
#
#     return beta, b, total_loss
#
#
# # ===== 9.2 外层优化循环 =====
# for i, w in enumerate(candidate_ws):
#     # 使用幂函数进一步放大斜率的重要性（如果基准权重显示斜率很重要）
#     # 斜率基准权重越大，说明越需要重视，进一步放大
#     slope_emphasis = base_w3 ** 0.5  # 开平方根，避免过度放大
#
#     adjusted_w = [
#         w[0] * 100 * base_w1,
#         w[1] * base_w2,
#         w[2] * base_w3  # 额外放大斜率权重
#     ]
#
#     # 归一化调整后的权重
#     total_w = sum(adjusted_w)
#     adjusted_w = [w / total_w for w in adjusted_w]
#
#     print(f"\n第{i + 1}组调整后权重: {[f'{x:.6f}' for x in adjusted_w]}")
#
#     beta, b, total_loss = fit_inner_huber_model(
#         w=adjusted_w,
#         X_exp_train=X_exp,
#         y_exp_train=y_exp,
#         mat_idx_train=mat_idx_per_sample,
#         Cp1_true_train=Cp1_true_all,
#         Cp2_true_train=Cp2_true_all,
#         slope_true_train=slope_true_all,
#         T1_hat_all=T1_hat_all,
#         T2_hat_all=T2_hat_all
#     )
#
#     y_val_pred = X_exp @ beta + b
#     r2 = r2_score(y_exp, y_val_pred)
#
#     print(f"权重 {[f'{x:.6f}' for x in adjusted_w]} -> R² = {r2:.6f}, 总损失 = {total_loss:.6f}")
#
#     if r2 > best_r2:
#         best_r2 = r2
#         best_w = adjusted_w
#         best_beta = beta
#         best_b = b
#
# print(f"\n外层最优权重 w* = {best_w}, 验证集 R² = {best_r2:.6f}")
#
# # ===== 10. 用最优权重重训 =====
# print("\n使用最优权重进行最终训练...")
# beta_final, b_final, _ = fit_inner_huber_model(
#     w=best_w,
#     X_exp_train=X_exp,
#     y_exp_train=y_exp,
#     mat_idx_train=mat_idx_per_sample,
#     Cp1_true_train=Cp1_true_all,
#     Cp2_true_train=Cp2_true_all,
#     slope_true_train=slope_true_all,
#     T1_hat_all=T1_hat_all,
#     T2_hat_all=T2_hat_all
# )
#
# # ===== 11. 评估结果 =====
# y_pred_all = X_exp @ beta_final + b_final
# mse_all = mean_squared_error(y_exp, y_pred_all)
# r2_all = r2_score(y_exp, y_pred_all)
#
# rel_err = np.abs((y_pred_all - y_exp) / np.where(np.abs(y_exp) < 1e-12, 1e-12, y_exp)) * 100
#
# # ===== 12. 导出结果 =====
# results = pd.DataFrame({
#     "Material_ID": material_ids_all[mat_idx_per_sample],
#     "Temperature (K)": T_per_sample,
#     "Cp_measured": y_exp,
#     "Cp_predicted": y_pred_all,
#     "Relative_Error_%": rel_err
# })
# results.to_excel("Cp预测结果_Huber优化后.xlsx", index=False)
#
# feature_labels = list(group_cols) + [f"{g}_T" for g in group_cols] + ["slope×T"]
# coef_df = pd.DataFrame({"Feature": feature_labels, "Contribution": beta_final})
# coef_df.to_excel("Cp系数表_Huber优化后.xlsx", index=False)
#
# # ===== 13. 最终结果汇总 =====
# print("\n" + "=" * 60)
# print("🎯 最终优化结果汇总 (Huber回归)")
# print("=" * 60)
# print(f"最优权重组合: w1={best_w[0]:.6f}, w2={best_w[1]:.6f}, w3={best_w[2]:.6f}")
# print(f"最终模型性能:")
# print(f"  R²  = {r2_all:.6f}")
# print(f"  MSE = {mse_all:.6f}")
# print(f"  ≤1%: {(rel_err <= 1).sum()}/{len(rel_err)} ({(rel_err <= 1).sum() / len(rel_err) * 100:.2f}%)")
# print(f"  ≤5%: {(rel_err <= 5).sum()}/{len(rel_err)} ({(rel_err <= 5).sum() / len(rel_err) * 100:.2f}%)")
# print(f"  ≤10%: {(rel_err <= 10).sum()}/{len(rel_err)} ({(rel_err <= 10).sum() / len(rel_err) * 100:.2f}%)")
#
# # 计算平均相对误差
# mean_rel_err = np.mean(rel_err)
# median_rel_err = np.median(rel_err)
# print(f"  平均相对误差: {mean_rel_err:.2f}%")
# print(f"  中位数相对误差: {median_rel_err:.2f}%")
#
# # 计算R²_adjusted
# n_samples = len(y_exp)
# n_features = X_exp.shape[1]
# r2_adjusted = 1 - (1 - r2_all) * (n_samples - 1) / (n_samples - n_features - 1)
# print(f"  调整后R²: {r2_adjusted:.6f}")
#
# print("=" * 60)
# print("✅ 完成！预测结果和系数表已保存到Excel文件")



#
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import HuberRegressor
# from scipy.optimize import minimize
#
#
# # ===== 0. 工具：候选权重采样（外层） =====
# def sample_weight_triplets(n=2, seed=2025):
#     """
#     采样 n 组 (w1, w2, w3)，非负且和为1（避免全0退化）。
#     用 Dirichlet(1,1,1) 随机权重，覆盖面更均匀。
#     """
#     rng = np.random.default_rng(seed)
#     W = rng.dirichlet([1.0, 1.0, 1.0], size=n)
#     return W
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
# y_T1 = df.loc[valid_mask, target_column_T1].values
# T1_model = GradientBoostingRegressor(
#     n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0  # 简化模型
# ).fit(X_poly, y_T1)
#
# Cp1_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 9])
# Cp2_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 50])
#
# X_poly_all = poly.transform(X_groups)
# T1_hat_all = T1_model.predict(X_poly_all)
# T2_hat_all = 1.5 * T1_hat_all
#
# Cp1_pred_all = Cp1_model.predict(X_groups)
# Cp2_pred_all = Cp2_model.predict(X_groups)
#
# # ===== 4. 计算斜率 =====
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_pred_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# # ===== 5. 真实参考点值 =====
# Cp1_true_all = df.iloc[:, 9].astype(float).values
# Cp2_true_all = df.iloc[:, 50].astype(float).values
# T1_true_all = df[target_column_T1].astype(float).values
# T2_true_all = 1.5 * T1_true_all
#
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_true_all = (Cp2_true_all - Cp1_true_all) / (T2_true_all - T1_true_all)
#
# # ===== 6. 构建实验点样本 =====
# slope_feat_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# X_exp_list, y_exp_list, mat_idx_list, T_list = [], [], [], []
# for i in range(len(df)):
#     Nk = X_groups.iloc[i].values.astype(float)
#     s_feat = slope_feat_all[i]
#     temps_i = df.loc[df.index[i], temp_cols].values.astype(float)
#     cps_i = df.loc[df.index[i], cp_cols].values.astype(float)
#
#     for T, Cp in zip(temps_i, cps_i):
#         if not (np.isfinite(T) and np.isfinite(Cp)):
#             continue
#         x = np.concatenate([Nk, Nk * T, [s_feat * T]])
#         X_exp_list.append(x)
#         y_exp_list.append(Cp)
#         mat_idx_list.append(i)
#         T_list.append(T)
#
# X_exp = np.asarray(X_exp_list)
# y_exp = np.asarray(y_exp_list)
# mat_idx_per_sample = np.asarray(mat_idx_list)
# T_per_sample = np.asarray(T_list)
#
#
# # ===== 7. 修正：基于反比例的自适应权重计算 =====
# def calculate_adaptive_weights_inverse(X_exp, y_exp, Cp1_true, Cp2_true, slope_true):
#     """计算每个损失项的数量级并使用反比例调整权重"""
#     L_exp_typical = np.mean(np.abs(y_exp - np.mean(y_exp)))  # 实验点的数量级
#     L_ref_typical = (np.mean(np.abs(Cp1_true - np.mean(Cp1_true))) +
#                      np.mean(np.abs(Cp2_true - np.mean(Cp2_true)))) / 2  # 参考点的数量级
#     L_slope_typical = np.mean(np.abs(slope_true - np.mean(slope_true)))  # 斜率的数量级
#
#     L_exp_typical = max(L_exp_typical, 1e-10)
#     L_ref_typical = max(L_ref_typical, 1e-10)
#     L_slope_typical = max(L_slope_typical, 1e-10)
#
#     print(f"损失项典型值: 实验点={L_exp_typical:.2f}, 参考点={L_ref_typical:.2f}, 斜率={L_slope_typical:.6f}")
#
#     # 使用反比例关系：数值越小，权重应该越大（放大作用）
#     w1_base = 1.0 / L_exp_typical  # 实验点数值大，权重小
#     w2_base = 1.0 / L_ref_typical  # 参考点数值大，权重小
#     w3_base = 1.0 / L_slope_typical  # 斜率数值小，权重大（放大！）
#
#     # 归一化
#     total_base = w1_base + w2_base + w3_base
#     w1_normalized = w1_base / total_base
#     w2_normalized = w2_base / total_base
#     w3_normalized = w3_base / total_base
#
#     return w1_normalized, w2_normalized, w3_normalized
#
#
# # ===== 7.1 计算损失倍数（用于损失函数内部放大） =====
# def calculate_loss_multipliers(L_exp, L_ref, L_slope, max_multiplier=1000):
#     """计算损失放大倍数"""
#     max_loss = max(L_exp, L_ref, L_slope)
#
#     # 计算需要放大的倍数
#     multiplier_exp = max_loss / L_exp
#     multiplier_ref = max_loss / L_ref
#     multiplier_slope = max_loss / L_slope
#
#     # 限制最大倍数，避免极端值
#     multiplier_exp = min(multiplier_exp, max_multiplier)
#     multiplier_ref = min(multiplier_ref, max_multiplier)
#     multiplier_slope = min(multiplier_slope, max_multiplier)
#
#     return multiplier_exp, multiplier_ref, multiplier_slope
#
#
# # 计算反比例基准权重
# base_w1, base_w2, base_w3 = calculate_adaptive_weights_inverse(
#     X_exp, y_exp, Cp1_true_all, Cp2_true_all, slope_true_all
# )
#
# # 计算损失放大倍数
# L_exp, L_ref, L_slope = (
#     np.mean(np.abs(y_exp - np.mean(y_exp))),
#     (np.mean(np.abs(Cp1_true_all - np.mean(Cp1_true_all))) +
#      np.mean(np.abs(Cp2_true_all - np.mean(Cp2_true_all)))) / 2,
#     np.mean(np.abs(slope_true_all - np.mean(slope_true_all)))
# )
# multiplier_exp, multiplier_ref, multiplier_slope = calculate_loss_multipliers(L_exp, L_ref, L_slope)
#
#
# # ===== 8. 自定义Huber损失函数（整合三项损失） =====
# def custom_huber_loss_with_weights(theta, X, y, w1, w2, w3, mat_idx,
#                                    Cp1_true, Cp2_true, slope_true,
#                                    T1_hat, T2_hat, multiplier_exp,
#                                    multiplier_ref, multiplier_slope):
#     """
#     自定义损失函数，将三项损失整合到Huber回归中
#     """
#     beta = theta[:-1]
#     b = theta[-1]
#
#     # 实验点预测和Huber损失
#     y_pred = X @ beta + b
#     huber_delta = 1.0
#     error = np.abs(y - y_pred)
#     L_exp = np.where(error <= huber_delta, 0.5 * error ** 2, huber_delta * (error - 0.5 * huber_delta))
#     L_exp = np.sum(L_exp) * multiplier_exp
#
#     # 参考点预测和损失
#     unique_materials = np.unique(mat_idx)
#     L_ref = 0
#     L_slope = 0
#
#     for mat_idx_val in unique_materials:
#         Nk = X_groups.iloc[mat_idx_val].values.astype(float)
#         s_feat = slope_feat_all[mat_idx_val]
#
#         # 参考点T1
#         x_T1 = np.concatenate([Nk, Nk * T1_hat[mat_idx_val], [s_feat * T1_hat[mat_idx_val]]])
#         Cp1_pred = x_T1 @ beta + b
#         L_ref += np.abs(Cp1_true[mat_idx_val] - Cp1_pred)
#
#         # 参考点T2
#         x_T2 = np.concatenate([Nk, Nk * T2_hat[mat_idx_val], [s_feat * T2_hat[mat_idx_val]]])
#         Cp2_pred = x_T2 @ beta + b
#         L_ref += np.abs(Cp2_true[mat_idx_val] - Cp2_pred)
#
#         # 斜率损失
#         if T2_hat[mat_idx_val] != T1_hat[mat_idx_val]:  # 避免除零
#             slope_pred = (Cp2_pred - Cp1_pred) / (T2_hat[mat_idx_val] - T1_hat[mat_idx_val])
#             L_slope += np.abs(slope_true[mat_idx_val] - slope_pred)
#
#     L_ref = L_ref * multiplier_ref
#     L_slope = L_slope * multiplier_slope
#
#     # 加权总损失
#     total_loss = w1 * L_exp + w2 * L_ref + w3 * L_slope
#     return total_loss
#
#
# # ===== 9. 外层优化（随机采样并结合自适应基准权重） =====
# candidate_ws = sample_weight_triplets(n=10, seed=2025)  # 只测试10组权重
#
# best_w = None
# best_r2 = -np.inf
# best_beta = None
# best_b = None
#
#
# # ===== 9.1 内层优化函数定义 =====
# def fit_inner_custom_model(w, X_exp_train, y_exp_train, mat_idx_train,
#                            Cp1_true_train, Cp2_true_train, slope_true_train,
#                            T1_hat_all, T2_hat_all):
#     w1, w2, w3 = w
#
#     n_feat = X_exp_train.shape[1]
#     theta0 = np.zeros(n_feat + 1)
#
#     # 使用自定义损失函数进行优化
#     res = minimize(
#         custom_huber_loss_with_weights,
#         theta0,
#         args=(X_exp_train, y_exp_train, w1, w2, w3, mat_idx_train,
#               Cp1_true_train, Cp2_true_train, slope_true_train,
#               T1_hat_all, T2_hat_all, multiplier_exp, multiplier_ref, multiplier_slope),
#         method="L-BFGS-B",
#         options={"maxiter": 10000, "ftol": 1e-3}
#     )
#
#     if res.success:
#         theta = res.x
#         beta = theta[:-1]
#         b = theta[-1]
#
#         # 计算预测值和R²
#         y_pred = X_exp_train @ beta + b
#         r2 = r2_score(y_exp_train, y_pred)
#
#         return beta, b, res.fun, r2
#     else:
#         raise ValueError(f"优化失败: {res.message}")
#
#
# # ===== 9.2 外层优化循环 =====
# for i, w in enumerate(candidate_ws):
#     # 调整权重
#     adjusted_w = [
#         w[0] * 100*base_w1,
#         w[1] * base_w2,
#         w[2] * base_w3
#     ]
#
#     # 归一化调整后的权重
#     total_w = sum(adjusted_w)
#     adjusted_w = [w / total_w for w in adjusted_w]
#
#     print(f"\n第{i + 1}组调整后权重: {[f'{x:.6f}' for x in adjusted_w]}")
#
#     try:
#         beta, b, total_loss, r2 = fit_inner_custom_model(
#             w=adjusted_w,
#             X_exp_train=X_exp,
#             y_exp_train=y_exp,
#             mat_idx_train=mat_idx_per_sample,
#             Cp1_true_train=Cp1_true_all,
#             Cp2_true_train=Cp2_true_all,
#             slope_true_train=slope_true_all,
#             T1_hat_all=T1_hat_all,
#             T2_hat_all=T2_hat_all
#         )
#
#         print(f"权重 {[f'{x:.6f}' for x in adjusted_w]} -> R² = {r2:.6f}, 总损失 = {total_loss:.6f}")
#
#         if r2 > best_r2:
#             best_r2 = r2
#             best_w = adjusted_w
#             best_beta = beta
#             best_b = b
#
#     except Exception as e:
#         print(f"权重 {[f'{x:.6f}' for x in adjusted_w]} -> 优化失败: {e}")
#         continue
#
# print(f"\n外层最优权重 w* = {best_w}, 验证集 R² = {best_r2:.6f}")
#
# # ===== 10. 用最优权重重训 =====
# print("\n使用最优权重进行最终训练...")
# beta_final, b_final, _, r2_final = fit_inner_custom_model(
#     w=best_w,
#     X_exp_train=X_exp,
#     y_exp_train=y_exp,
#     mat_idx_train=mat_idx_per_sample,
#     Cp1_true_train=Cp1_true_all,
#     Cp2_true_train=Cp2_true_all,
#     slope_true_train=slope_true_all,
#     T1_hat_all=T1_hat_all,
#     T2_hat_all=T2_hat_all
# )
#
# # ===== 11. 评估结果 =====
# y_pred_all = X_exp @ beta_final + b_final
# mse_all = mean_squared_error(y_exp, y_pred_all)
# r2_all = r2_score(y_exp, y_pred_all)
#
# rel_err = np.abs((y_pred_all - y_exp) / np.where(np.abs(y_exp) < 1e-12, 1e-12, y_exp)) * 100
#
# # ===== 12. 导出结果 =====
# results = pd.DataFrame({
#     "Material_ID": material_ids_all[mat_idx_per_sample],
#     "Temperature (K)": T_per_sample,
#     "Cp_measured": y_exp,
#     "Cp_predicted": y_pred_all,
#     "Relative_Error_%": rel_err
# })
# results.to_excel("Cp预测结果_Huber优化后.xlsx", index=False)
#
# feature_labels = list(group_cols) + [f"{g}_T" for g in group_cols] + ["slope×T"]
# coef_df = pd.DataFrame({"Feature": feature_labels, "Contribution": beta_final})
# coef_df.to_excel("Cp系数表_Huber优化后.xlsx", index=False)
#
# # ===== 13. 最终结果汇总 =====
# print("\n" + "=" * 60)
# print("🎯 最终优化结果汇总 (自定义Huber回归)")
# print("=" * 60)
# print(f"最优权重组合: w1={best_w[0]:.6f}, w2={best_w[1]:.6f}, w3={best_w[2]:.6f}")
# print(f"最终模型性能:")
# print(f"  R²  = {r2_all:.6f}")
# print(f"  MSE = {mse_all:.6f}")
# print(f"  ≤1%: {(rel_err <= 1).sum()}/{len(rel_err)} ({(rel_err <= 1).sum() / len(rel_err) * 100:.2f}%)")
# print(f"  ≤5%: {(rel_err <= 5).sum()}/{len(rel_err)} ({(rel_err <= 5).sum() / len(rel_err) * 100:.2f}%)")
# print(f"  ≤10%: {(rel_err <= 10).sum()}/{len(rel_err)} ({(rel_err <= 10).sum() / len(rel_err) * 100:.2f}%)")
#
# # 计算平均相对误差
# mean_rel_err = np.mean(rel_err)
# median_rel_err = np.median(rel_err)
# print(f"  平均相对误差: {mean_rel_err:.2f}%")
# print(f"  中位数相对误差: {median_rel_err:.2f}%")
#
# # 计算R²_adjusted
# n_samples = len(y_exp)
# n_features = X_exp.shape[1]
# r2_adjusted = 1 - (1 - r2_all) * (n_samples - 1) / (n_samples - n_features - 1)
# print(f"  调整后R²: {r2_adjusted:.6f}")
#
# print("=" * 60)
# print("✅ 完成！预测结果和系数表已保存到Excel文件")
#
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import HuberRegressor
# from scipy.optimize import minimize
#
#
# # ===== 0. 工具：候选权重采样（外层） =====
# def sample_weight_triplets(n=2, seed=2025):
#     """
#     采样 n 组 (w1, w2, w3)，非负且和为1（避免全0退化）。
#     用 Dirichlet(1,1,1) 随机权重，覆盖面更均匀。
#     """
#     rng = np.random.default_rng(seed)
#     W = rng.dirichlet([1.0, 1.0, 1.0], size=n)
#     return W
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
# y_T1 = df.loc[valid_mask, target_column_T1].values
# T1_model = GradientBoostingRegressor(
#     n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0  # 简化模型
# ).fit(X_poly, y_T1)
#
# Cp1_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 9])
# Cp2_model = HuberRegressor(max_iter=1000).fit(X_groups, df.iloc[:, 50])
#
# X_poly_all = poly.transform(X_groups)
# T1_hat_all = T1_model.predict(X_poly_all)
# T2_hat_all = 1.5 * T1_hat_all
#
# Cp1_pred_all = Cp1_model.predict(X_groups)
# Cp2_pred_all = Cp2_model.predict(X_groups)
#
# # ===== 4. 计算斜率 =====
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_pred_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# # ===== 5. 真实参考点值 =====
# Cp1_true_all = df.iloc[:, 9].astype(float).values
# Cp2_true_all = df.iloc[:, 50].astype(float).values
# T1_true_all = df[target_column_T1].astype(float).values
# T2_true_all = 1.5 * T1_true_all
#
# with np.errstate(divide='ignore', invalid='ignore'):
#     slope_true_all = (Cp2_true_all - Cp1_true_all) / (T2_true_all - T1_true_all)
#
# # ===== 6. 构建实验点样本 =====
# slope_feat_all = (Cp2_pred_all - Cp1_pred_all) / (T2_hat_all - T1_hat_all)
#
# X_exp_list, y_exp_list, mat_idx_list, T_list = [], [], [], []
# for i in range(len(df)):
#     Nk = X_groups.iloc[i].values.astype(float)
#     s_feat = slope_feat_all[i]
#     temps_i = df.loc[df.index[i], temp_cols].values.astype(float)
#     cps_i = df.loc[df.index[i], cp_cols].values.astype(float)
#
#     for T, Cp in zip(temps_i, cps_i):
#         if not (np.isfinite(T) and np.isfinite(Cp)):
#             continue
#         x = np.concatenate([Nk, Nk * T, [s_feat * T]])
#         X_exp_list.append(x)
#         y_exp_list.append(Cp)
#         mat_idx_list.append(i)
#         T_list.append(T)
#
# X_exp = np.asarray(X_exp_list)
# y_exp = np.asarray(y_exp_list)
# mat_idx_per_sample = np.asarray(mat_idx_list)
# T_per_sample = np.asarray(T_list)
#
#
# # ===== 7. 基础损失函数 =====
# def loss_sum_three_parts(y_exp_true, y_exp_pred,
#                          Cp1_true, Cp2_true,
#                          Cp1_pred, Cp2_pred,
#                          slope_true, slope_pred,
#                          w1, w2, w3):
#     L_exp = np.sum(np.abs(y_exp_true - y_exp_pred))
#     L_ref = np.sum(np.abs(Cp1_true - Cp1_pred)) + np.sum(np.abs(Cp2_true - Cp2_pred))
#     L_slope = np.sum(np.abs(slope_true - slope_pred))
#     return w1 * L_exp + w2 * L_ref + w3 * L_slope
#
#
# # ===== 8. 修正：基于反比例的自适应权重计算 =====
# def calculate_adaptive_weights_inverse(X_exp, y_exp, Cp1_true, Cp2_true, slope_true):
#     """计算每个损失项的数量级并使用反比例调整权重"""
#     L_exp_typical = np.mean(np.abs(y_exp - np.mean(y_exp)))  # 实验点的数量级
#     L_ref_typical = (np.mean(np.abs(Cp1_true - np.mean(Cp1_true))) +
#                      np.mean(np.abs(Cp2_true - np.mean(Cp2_true)))) / 2  # 参考点的数量级
#     L_slope_typical = np.mean(np.abs(slope_true - np.mean(slope_true)))  # 斜率的数量级
#
#     L_exp_typical = max(L_exp_typical, 1e-10)
#     L_ref_typical = max(L_ref_typical, 1e-10)
#     L_slope_typical = max(L_slope_typical, 1e-10)
#
#     print(f"损失项典型值: 实验点={L_exp_typical:.2f}, 参考点={L_ref_typical:.2f}, 斜率={L_slope_typical:.6f}")
#
#     # 使用反比例关系：数值越小，权重应该越大（放大作用）
#     w1_base = 1.0/ L_exp_typical  # 实验点数值大，权重小
#     w2_base = 1.0 / L_ref_typical  # 参考点数值大，权重小
#     w3_base = 1.0 / L_slope_typical  # 斜率数值小，权重大（放大！）
#
#     # 归一化
#     total_base = w1_base + w2_base + w3_base
#     w1_normalized = w1_base / total_base
#     w2_normalized = w2_base / total_base
#     w3_normalized = w3_base / total_base
#
#     return w1_normalized, w2_normalized, w3_normalized
#
#
# # 计算反比例基准权重
# base_w1, base_w2, base_w3 = calculate_adaptive_weights_inverse(
#     X_exp, y_exp, Cp1_true_all, Cp2_true_all, slope_true_all
# )
#
#
# # ===== 9. 外层优化（随机采样并结合自适应基准权重） =====
# candidate_ws = sample_weight_triplets(n=10, seed=2025)  # 随机采样权重
#
# best_w = None
# best_r2 = -np.inf
# best_theta = None
#
#
# # ===== 9.1 外层优化循环 =====
# for i, w in enumerate(candidate_ws):
#     adjusted_w = [w[0] * base_w1, w[1] * base_w2, w[2] * base_w3]  # 可选的权重调整
#     adjusted_w = np.array(adjusted_w) / np.sum(adjusted_w)  # 归一化
#
#     print(f"\n第{i + 1}组调整后权重: {[f'{x:.6f}' for x in adjusted_w]}")
#
#     try:
#         # 使用 Huber 回归模型进行优化
#         model_huber = HuberRegressor(epsilon=1.35, max_iter=10000, alpha=0.001)
#         model_huber.fit(X_exp, y_exp)
#
#         y_pred = model_huber.predict(X_exp)
#         r2 = r2_score(y_exp, y_pred)
#
#         print(f"权重 {[f'{x:.6f}' for x in adjusted_w]} -> R² = {r2:.6f}")
#
#         if r2 > best_r2:
#             best_r2 = r2
#             best_w = adjusted_w
#             best_theta = model_huber.coef_
#
#     except Exception as e:
#         print(f"优化失败: {e}")
#         continue
#
# print(f"\n最优权重组合: w* = {best_w}, 最小优化损失 = {best_r2:.6f}")
#
# # ===== 10. 用最优权重重训 =====
# print("\n使用最优权重进行最终训练...")
#
# model_final = HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001)
# model_final.fit(X_exp, y_exp)
# y_pred_final = model_final.predict(X_exp)
#
# # ===== 11. 评估结果 =====
# r2_final = r2_score(y_exp, y_pred_final)
#
# # 计算相对误差
# rel_err = np.abs((y_pred_final - y_exp) / np.where(np.abs(y_exp) < 1e-12, 1e-12, y_exp)) * 100
#
# # 计算平均相对误差
# mean_rel_err = np.mean(rel_err)
# median_rel_err = np.median(rel_err)
#
# print(f"最终模型 R² = {r2_final:.6f}")
# print(f"平均相对误差: {mean_rel_err:.2f}%")
# print(f"中位数相对误差: {median_rel_err:.2f}%")

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


# ===== 8. 加权损失函数 =====
def weighted_huber_loss(theta, X_exp, y_exp, mat_idx,
                        Cp1_true, Cp2_true, slope_true,
                        T1_hat, T2_hat, w1, w2, w3, alpha=0.0001, epsilon=1.35):
    """使用Huber损失的加权目标函数"""
    beta = theta[:-1]
    b = theta[-1]

    # 实验点损失
    y_pred_exp = X_exp @ beta + b
    exp_residuals = y_exp - y_pred_exp
    L_exp = np.sum(huber_loss(exp_residuals, epsilon))

    # 参考点损失
    unique_materials = np.unique(mat_idx)
    L_ref = 0
    L_slope = 0

    for mat_idx_val in unique_materials:
        Nk = X_groups.iloc[mat_idx_val].values.astype(float)
        s_feat = slope_feat_all[mat_idx_val]

        # 参考点1
        x_T1 = np.concatenate([Nk, Nk * T1_hat[mat_idx_val], [s_feat * T1_hat[mat_idx_val]]])
        Cp1_pred = x_T1 @ beta + b
        ref1_residual = Cp1_true[mat_idx_val] - Cp1_pred
        L_ref += huber_loss(ref1_residual, epsilon)

        # 参考点2
        x_T2 = np.concatenate([Nk, Nk * T2_hat[mat_idx_val], [s_feat * T2_hat[mat_idx_val]]])
        Cp2_pred = x_T2 @ beta + b
        ref2_residual = Cp2_true[mat_idx_val] - Cp2_pred
        L_ref += huber_loss(ref2_residual, epsilon)

        # 斜率损失
        if T2_hat[mat_idx_val] - T1_hat[mat_idx_val] > 1e-10:
            slope_pred = (Cp2_pred - Cp1_pred) / (T2_hat[mat_idx_val] - T1_hat[mat_idx_val])
            slope_residual = slope_true[mat_idx_val] - slope_pred
            L_slope += huber_loss(slope_residual, epsilon)

    # 正则化项
    regularization = alpha * np.sum(beta ** 2)

    return w1 * L_exp + w2 * L_ref + w3 * L_slope + regularization


# ===== 9. 自适应权重计算 =====
def calculate_adaptive_weights_inverse(X_exp, y_exp, Cp1_true, Cp2_true, slope_true):
    L_exp_typical = np.mean(np.abs(y_exp - np.mean(y_exp)))
    L_ref_typical = (np.mean(np.abs(Cp1_true - np.mean(Cp1_true))) +
                     np.mean(np.abs(Cp2_true - np.mean(Cp2_true)))) / 2
    L_slope_typical = np.mean(np.abs(slope_true - np.mean(slope_true)))

    L_exp_typical = max(L_exp_typical, 1e-10)
    L_ref_typical = max(L_ref_typical, 1e-10)
    L_slope_typical = max(L_slope_typical, 1e-10)

    print(f"损失项典型值: 实验点={L_exp_typical:.2f}, 参考点={L_ref_typical:.2f}, 斜率={L_slope_typical:.6f}")

    w1_base = 1.0 / L_exp_typical
    w2_base = 1.0 / L_ref_typical
    w3_base = 1.0 / L_slope_typical

    total_base = w1_base + w2_base + w3_base
    return w1_base / total_base, w2_base / total_base, w3_base / total_base


# 计算基准权重
base_w1, base_w2, base_w3 = calculate_adaptive_weights_inverse(
    X_exp, y_exp, Cp1_true_all, Cp2_true_all, slope_true_all
)

# ===== 10. 优化循环 =====
candidate_ws = sample_weight_triplets(n=10, seed=2025)
best_w = None
best_r2 = -np.inf
best_theta = None

for i, w in enumerate(candidate_ws):
    adjusted_w = [w[0] *base_w1, w[1] * base_w2, w[2] * base_w3]
    adjusted_w = np.array(adjusted_w) / np.sum(adjusted_w)

    print(f"\n第{i + 1}组权重: {[f'{x:.6f}' for x in adjusted_w]}")

    # 初始参数
    n_features = X_exp.shape[1]
    theta0 = np.zeros(n_features + 1)

    # 使用L-BFGS优化（与Huber回归相同的算法）
    res = minimize(
        weighted_huber_loss,
        theta0,
        args=(X_exp, y_exp, mat_idx_per_sample,
              Cp1_true_all, Cp2_true_all, slope_true_all,
              T1_hat_all, T2_hat_all,
              adjusted_w[0], adjusted_w[1], adjusted_w[2],
              0.0001, 1.35),  # alpha=0.0001, epsilon=1.35
        method='L-BFGS-B',
        options={'maxiter': 10000, 'ftol': 1e-3, 'disp': False}
    )

    if not res.success:
        print(f"优化失败: {res.message}")
        continue

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