# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
# from scipy import optimize
#
# # ========= 1. 读取数据 =========
# file_path = "heat capacity 207.xlsx"
# df = pd.read_excel(file_path, sheet_name="Sheet1")
# df = df.dropna(subset=[df.columns[0]])
# df[df.columns[0]] = df[df.columns[0]].astype(int)
#
# # ========= 2. 列定义 =========
# group_cols = df.columns[11:30]  # 19个基团
# temp_cols = df.columns[30:40]  # 10个温度点
# cp_cols = df.columns[40:50]  # 10个Cp值
# target_column_T1 = 'ASPEN Half Critical T'
#
# # ========= 3. 子模型训练 =========
# X_groups = df[group_cols]
# valid_mask = ~df[target_column_T1].isna()
#
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X_groups[valid_mask])
#
# # GradientBoostingRegressor 预测 T1
# y_T1 = df.loc[valid_mask, target_column_T1].values
# T1_model = GradientBoostingRegressor(
#     n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
# ).fit(X_poly, y_T1)
#
# # Cp1, Cp2 使用 HuberRegressor
# Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 9])
# Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 50])
#
# # ========= 4. 构建原始训练数据 =========
# X_total, y_total, material_ids, temperatures = [], [], [], []
# X_poly_all = poly.transform(X_groups)
#
# for i, row in df.iterrows():
#     material_id = row.iloc[0]
#     Nk = row[group_cols].values
#     temps = row[temp_cols].values
#     cps = row[cp_cols].values
#
#     Nk_df = pd.DataFrame([Nk], columns=group_cols)
#     Nk_poly = X_poly_all[i:i + 1]
#
#     try:
#         T1 = T1_model.predict(Nk_poly)[0]
#         if T1 <= 0 or np.isnan(T1):
#             continue
#         T2 = T1 * 1.5
#         Cp1 = Cp1_model.predict(Nk_df)[0]
#         Cp2 = Cp2_model.predict(Nk_df)[0]
#         slope = (Cp2 - Cp1) / (T2 - T1)
#     except:
#         continue
#
#     for T, Cp in zip(temps, cps):
#         if np.isnan(T) or np.isnan(Cp):
#             continue
#         features = np.concatenate([
#             Nk,
#             Nk * T,
#             [slope * T]
#         ])
#         X_total.append(features)
#         y_total.append(Cp)
#         material_ids.append(material_id)
#         temperatures.append(T)
#
# # ========= 5. 添加额外约束点 =========
# extra_weight = 0  # 给额外点的权重
# X_extra_list, y_extra_list = [], []
#
# for i, row in df.iterrows():
#     Nk = row[group_cols].values.reshape(1, -1)
#     Nk_df = pd.DataFrame([Nk.flatten()], columns=group_cols)
#
#     try:
#         T1 = row[target_column_T1]
#         if np.isnan(T1) or T1 <= 0:
#             continue
#         T2 = 1.5 * T1
#         Cp1_pred = Cp1_model.predict(Nk_df)[0]
#         Cp2_pred = Cp2_model.predict(Nk_df)[0]
#         slope = (Cp2_pred - Cp1_pred) / (T2 - T1)
#
#         for T_extra in [T1, T2]:
#             features = np.concatenate([
#                 Nk.flatten(),
#                 Nk.flatten() * T_extra,
#                 [slope * T_extra]
#             ])
#             X_extra_list.append(features)
#             Cp_extra = Cp1_pred + slope * (T_extra - T1)
#             y_extra_list.append(Cp_extra)
#     except:
#         continue
#
# X_extra = np.array(X_extra_list)
# y_extra = np.array(y_extra_list)
#
# # 合并原始数据和额外点
# X_combined = np.vstack([X_total, X_extra])
# y_combined = np.hstack([y_total, y_extra])
#
# # ========= 6. 去除截距列（scikit-learn风格） =========
# # X_combined 不包含截距列，截距作为参数的一部分
#
# # ========= 7. 给额外点加权 =========
# weights = np.ones(len(y_combined))
# weights[-len(y_extra):] = extra_weight
#
#
# # ========= 8. 完全模仿scikit-learn的Huber损失和梯度计算 =========
# def _huber_loss_and_gradient_scikit(w, X, y, epsilon, alpha, sample_weight):
#     """
#     完全模仿scikit-learn的Huber损失和梯度计算
#     包含scale参数优化
#     """
#     n_samples, n_features = X.shape
#     fit_intercept = True  # 我们总是拟合截距
#
#     # 参数分解：w = [系数, 截距, scale]
#     coef = w[:n_features]  # 前n_features个是系数
#     intercept = w[n_features]  # 下一个是截距
#     sigma = w[n_features + 1]  # 最后一个是scale
#
#     # 计算预测值和残差
#     y_pred = X @ coef + intercept
#     linear_loss = y - y_pred
#
#     # 标准化残差
#     scaled_residuals = linear_loss / sigma
#     abs_scaled_residuals = np.abs(scaled_residuals)
#     outliers_mask = abs_scaled_residuals > epsilon
#
#     # 计算异常值
#     outliers = abs_scaled_residuals[outliers_mask]
#     num_outliers = np.count_nonzero(outliers_mask)
#     n_non_outliers = X.shape[0] - num_outliers
#
#     outliers_sw = sample_weight[outliers_mask]
#     n_sw_outliers = np.sum(outliers_sw)
#
#     # 异常值损失
#     outlier_loss = (
#             2.0 * epsilon * sigma * np.sum(outliers_sw * outliers)
#             - sigma ** 2 * n_sw_outliers * epsilon ** 2
#     )
#
#     # 非异常值的二次损失
#     non_outliers = linear_loss[~outliers_mask]
#     weighted_non_outliers = sample_weight[~outliers_mask] * non_outliers
#     squared_loss = np.dot(weighted_non_outliers.T, non_outliers) / sigma
#
#     # 初始化梯度
#     grad = np.zeros_like(w)
#
#     # 二次部分的梯度（系数）
#     if n_non_outliers > 0:
#         X_non_outliers = X[~outliers_mask]
#         grad[:n_features] = (
#                 2.0 / sigma * (X_non_outliers.T @ weighted_non_outliers)
#         )
#
#     # 线性部分的梯度（系数）
#     if num_outliers > 0:
#         signed_outliers = np.ones_like(outliers)
#         signed_outliers_mask = linear_loss[outliers_mask] < 0
#         signed_outliers[signed_outliers_mask] = -1.0
#         X_outliers = X[outliers_mask]
#         sw_outliers = sample_weight[outliers_mask] * signed_outliers
#         grad[:n_features] -= 2.0 * epsilon * (X_outliers.T @ sw_outliers)
#
#     # 正则化梯度（系数）
#     grad[:n_features] += alpha * 2.0 * coef
#
#     # scale梯度
#     grad[n_features + 1] = n_samples * sigma
#     grad[n_features + 1] -= n_sw_outliers * epsilon ** 2 * sigma
#     grad[n_features + 1] -= squared_loss / sigma
#
#     # 截距梯度
#     grad[n_features] = -2.0 * np.sum(weighted_non_outliers) / sigma
#     if num_outliers > 0:
#         grad[n_features] -= 2.0 * epsilon * np.sum(sw_outliers)
#
#     # 总损失
#     loss = n_samples * sigma + squared_loss + outlier_loss
#     loss += alpha * np.dot(coef, coef)
#
#     return loss, grad
#
#
# # ========= 9. 优化（完全scikit-learn风格） =========
# # 初始化参数：系数 + 截距 + scale
# n_features = X_combined.shape[1]  # 39个特征
# parameters = np.zeros(n_features + 2, dtype=np.float64)  # 39系数 + 1截距 + 1scale
# parameters[n_features + 1] = 1.0  # scale初始化为1（最后一个位置）
#
# # 设置边界（与scikit-learn完全一致）
# bounds = [(-np.inf, np.inf)] * len(parameters)
# bounds[n_features + 1] = (np.finfo(np.float64).eps * 10, np.inf)  # scale的下界
#
# # 设置优化参数
# epsilon = 1.35
# alpha = 0.0001
# max_iter = 10000
# tol = 1e-5
#
# # 确保数据类型一致
# X_combined = np.asarray(X_combined, dtype=np.float64)
# y_combined = np.asarray(y_combined, dtype=np.float64)
# weights = np.asarray(weights, dtype=np.float64)
#
# print(f"开始优化，数据形状: X={X_combined.shape}, y={y_combined.shape}")
# print(f"参数数量: {len(parameters)} (系数: {n_features}, 截距: 1, scale: 1)")
#
# # 执行优化
# opt_res = optimize.minimize(
#     _huber_loss_and_gradient_scikit,
#     parameters,
#     method="L-BFGS-B",
#     jac=True,
#     args=(X_combined, y_combined, epsilon, alpha, weights),
#     options={
#         "maxiter": max_iter,
#         "gtol": tol,
#         "iprint": -1,  # 这里改为 -1
#         "disp": False   # 这里改为 False
#     },
#     bounds=bounds,
# )
#
# # 检查优化结果
# if opt_res.status == 2:
#     raise ValueError(
#         f"HuberRegressor convergence failed: L-BFGS-B solver terminated with {opt_res.message}"
#     )
#
# # 提取参数
# coef_opt = opt_res.x[:n_features]  # 前39个是系数
# intercept_opt = opt_res.x[n_features]  # 第40个是截距
# scale_opt = opt_res.x[n_features + 1]  # 第41个是scale参数
# n_iter_ = opt_res.nit
#
# # 预测
# y_pred_opt = X_combined @ coef_opt + intercept_opt
#
# print(f"优化完成，迭代次数: {n_iter_}")
# print(f"最终损失值: {opt_res.fun:.6f}")
# print(f"Scale参数: {scale_opt:.6f}")
# print(f"截距: {intercept_opt:.6f}")
#
# # ========= 10. 模型评估 =========
# mse = mean_squared_error(y_combined, y_pred_opt)
# r2 = r2_score(y_combined, y_pred_opt)
# ard = np.mean(np.abs((y_combined - y_pred_opt) / np.clip(np.abs(y_combined), 1e-10, None))) * 100
# print(f"R²={r2:.4f}, MSE={mse:.2f}, ARD={ard:.2f}%")
#
# # ========= 11. 输出预测结果 =========
# # 为额外点创建温度列表
# extra_temps = []
# for i, row in df.iterrows():
#     try:
#         T1 = row[target_column_T1]
#         if np.isnan(T1) or T1 <= 0:
#             continue
#         extra_temps.extend([T1, 1.5 * T1])
#     except:
#         continue
#
# min_len = min(len(extra_temps), len(y_extra))
# extra_temps = extra_temps[:min_len]
#
# results = pd.DataFrame({
#     "Material_ID": material_ids + ["Extra"] * len(y_extra),
#     "Temperature (K)": temperatures + extra_temps,
#     "Cp_measured": y_combined,
#     "Cp_predicted": y_pred_opt,
#     "Is_Extra": [False] * len(y_total) + [True] * len(y_extra),
#     "Residual": y_combined - y_pred_opt,
#     "Scaled_Residual": (y_combined - y_pred_opt) / scale_opt
# })
# results.to_excel("Cp预测结果_with_extra_points_scikit.xlsx", index=False)
# print("✅ 已保存预测结果: Cp预测结果_with_extra_points_scikit.xlsx")
#
# # 输出系数信息
# feature_names = [f"Group_{i}" for i in range(19)] + [f"Group*Temp_{i}" for i in range(19)] + ["Slope*Temp"]
# coef_df = pd.DataFrame({
#     "Feature": feature_names,
#     "Coefficient": coef_opt
# })
# print("\n模型系数:")
# print(coef_df.to_string(index=False))
# print(f"\n截距: {intercept_opt:.6f}")
# print(f"Scale参数: {scale_opt:.6f}")
#
# # 检测异常值
# residuals = y_combined - y_pred_opt
# scaled_residuals = residuals / scale_opt
# outliers_mask = np.abs(scaled_residuals) > epsilon
# print(f"\n检测到 {np.sum(outliers_mask)} 个异常值 ({(np.sum(outliers_mask) / len(y_combined) * 100):.2f}%)")
