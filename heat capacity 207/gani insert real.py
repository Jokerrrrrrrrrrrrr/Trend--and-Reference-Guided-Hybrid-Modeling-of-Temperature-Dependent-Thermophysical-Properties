# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ========= 1. 读取数据 =========
# file_path = "heat capacity 207.xlsx"
# df = pd.read_excel(file_path, sheet_name="Sheet1")
# df = df.dropna(subset=[df.columns[0]])
# df[df.columns[0]] = df[df.columns[0]].astype(int)
#
# # ========= 2. 列定义 =========
# group_cols = df.columns[11:30]   # 19个基团列
# temp_cols  = df.columns[30:40]   # 10个温度点
# cp_cols    = df.columns[40:50]   # 10个 Cp 值
# target_column_T1 = 'ASPEN Half Critical T'  # 真实 T1 所在列
#
# # 你给出的“真实四列”：Cp1_true=第10列, Cp2_true=第51列, T1_true=target_column_T1, T2_true=1.5*T1_true
# CP1_TRUE_IDX = 9
# CP2_TRUE_IDX = 50
# T1_TRUE_COL  = target_column_T1
#
# # ========= 2.1 强制数值化（关键修正）=========
# # 将用于建模/计算的列全部转为数值，无法解析的设为 NaN
# cols_to_numeric = list(group_cols) + list(temp_cols) + list(cp_cols) + [T1_TRUE_COL]
# # 注意 iloc 两列需要单独处理后写回
# df[group_cols] = df[group_cols].apply(pd.to_numeric, errors="coerce")
# df[temp_cols]  = df[temp_cols].apply(pd.to_numeric, errors="coerce")
# df[cp_cols]    = df[cp_cols].apply(pd.to_numeric, errors="coerce")
#
# # 把第10、51列也数值化
# df.iloc[:, CP1_TRUE_IDX] = pd.to_numeric(df.iloc[:, CP1_TRUE_IDX], errors="coerce")
# df.iloc[:, CP2_TRUE_IDX] = pd.to_numeric(df.iloc[:, CP2_TRUE_IDX], errors="coerce")
# df[T1_TRUE_COL]          = pd.to_numeric(df[T1_TRUE_COL], errors="coerce")
#
# # ========= 3. 子模型训练 =========
# X_groups = df[group_cols]
# # T1 训练有效掩码
# valid_mask = X_groups.notna().all(1) & df[T1_TRUE_COL].notna()
#
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X_groups[valid_mask])
#
# # 用 GradientBoostingRegressor 预测 T1（与你原有设置一致）
# y_T1 = df.loc[valid_mask, T1_TRUE_COL].values
# T1_model = GradientBoostingRegressor(
#     n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
# ).fit(X_poly, y_T1)
#
# # Cp1, Cp2 子模型（用额外两列的真实 Cp）
# Cp1_true_series = df.iloc[:, CP1_TRUE_IDX]
# Cp2_true_series = df.iloc[:, CP2_TRUE_IDX]
# valid_cp_mask = X_groups.notna().all(1) & Cp1_true_series.notna() & Cp2_true_series.notna()
#
# Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups[valid_cp_mask].values, Cp1_true_series[valid_cp_mask].values)
# Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups[valid_cp_mask].values, Cp2_true_series[valid_cp_mask].values)
#
# # ========= 3.1 子模型评估（in-sample）=========
# y_pred_T1 = T1_model.predict(X_poly)
# r2_T1 = r2_score(y_T1, y_pred_T1)
# mse_T1 = mean_squared_error(y_T1, y_pred_T1)
#
# y_Cp1_true = Cp1_true_series[valid_cp_mask]
# y_Cp1_pred = Cp1_model.predict(X_groups[valid_cp_mask].values)
# r2_Cp1 = r2_score(y_Cp1_true, y_Cp1_pred)
# mse_Cp1 = mean_squared_error(y_Cp1_true, y_Cp1_pred)
#
# y_Cp2_true = Cp2_true_series[valid_cp_mask]
# y_Cp2_pred = Cp2_model.predict(X_groups[valid_cp_mask].values)
# r2_Cp2 = r2_score(y_Cp2_true, y_Cp2_pred)
# mse_Cp2 = mean_squared_error(y_Cp2_true, y_Cp2_pred)
#
# print("\n📌 子模型评估结果：")
# print(f"T1_model ->     R²: {r2_T1:.4f} | MSE: {mse_T1:.4f}")
# print(f"Cp1_model ->    R²: {r2_Cp1:.4f} | MSE: {mse_Cp1:.4f}")
# print(f"Cp2_model ->    R²: {r2_Cp2:.4f} | MSE: {mse_Cp2:.4f}")
#
# # ========= 4. 构建训练数据（两种 slope 变体）=========
# X_A, y_A, id_A, T_A = [], [], [], []  # A：真实ΔCp / 预测ΔT
# X_B, y_B, id_B, T_B = [], [], [], []  # B：预测ΔCp / 真实ΔT
#
# # 对“所有行”的基团特征做 transform（poly 之前在 valid_mask 上 fit 过）
# X_poly_all = poly.transform(X_groups.fillna(0))  # 这里 transform 不会引入 NaN；模型预测前仍会过滤
#
# for i, row in df.iterrows():
#     try:
#         material_id = row.iloc[0]
#
#         # --- 取 Nk，确保数值 & 检查缺失 ---
#         Nk_series = row[group_cols].astype(float)
#         if pd.isna(Nk_series).any():
#             continue
#         Nk = Nk_series.values
#
#         # --- 预测侧 T1/T2 与 Cp1/Cp2 ---
#         # 用与第 i 行对齐的多项式特征（已经 transform 好）
#         T1_pred = float(T1_model.predict(X_poly_all[i:i+1])[0])
#         if not np.isfinite(T1_pred) or T1_pred <= 0:
#             continue
#         T2_pred = 1.5 * T1_pred
#
#         Nk_df = pd.DataFrame([Nk], columns=group_cols)
#         Cp1_pred = float(Cp1_model.predict(Nk_df.values)[0])
#         Cp2_pred = float(Cp2_model.predict(Nk_df.values)[0])
#         if not (np.isfinite(Cp1_pred) and np.isfinite(Cp2_pred)):
#             continue
#
#         # --- 真实侧 Cp1/Cp2/T1/T2 ---
#         Cp1_true = row.iloc[CP1_TRUE_IDX]
#         Cp2_true = row.iloc[CP2_TRUE_IDX]
#         T1_true  = row[T1_TRUE_COL]
#         if not (np.isfinite(Cp1_true) and np.isfinite(Cp2_true) and np.isfinite(T1_true)):
#             continue
#         T2_true  = 1.5 * T1_true
#
#         # 防止除零
#         if T2_pred == T1_pred or T2_true == T1_true:
#             continue
#
#         # --- 两种 slope 变体 ---
#         slope_A = (Cp2_true - Cp1_true) / (T2_pred - T1_pred)   # 分子真实ΔCp，分母预测ΔT
#         slope_B = (Cp2_pred - Cp1_pred) / (T2_true - T1_true)   # 分子预测ΔCp，分母真实ΔT
#         if not (np.isfinite(slope_A) and np.isfinite(slope_B)):
#             continue
#
#         # --- 逐温度点展开 ---
#         temps = row[temp_cols].astype(float).values
#         cps   = row[cp_cols].astype(float).values
#         # 掩码：去除 NaN
#         mask_pts = np.isfinite(temps) & np.isfinite(cps)
#         if not mask_pts.any():
#             continue
#
#         for T, Cp in zip(temps[mask_pts], cps[mask_pts]):
#             feats_A = np.concatenate([Nk, Nk*T, [slope_A*T]])
#             feats_B = np.concatenate([Nk, Nk*T, [slope_B*T]])
#             X_A.append(feats_A); y_A.append(Cp); id_A.append(material_id); T_A.append(T)
#             X_B.append(feats_B); y_B.append(Cp); id_B.append(material_id); T_B.append(T)
#
#     except Exception as e:
#         print(f"[WARN] row {i} skipped: {e}")
#         continue
#
# X_A = np.asarray(X_A); y_A = np.asarray(y_A)
# X_B = np.asarray(X_B); y_B = np.asarray(y_B)
#
# if X_A.size == 0 or X_B.size == 0:
#     raise RuntimeError(
#         f"没有可用样本：X_A.shape={X_A.shape}, X_B.shape={X_B.shape}。"
#         "请检查 group/temp/cp 列是否为数值、以及真实列是否存在缺失。"
#     )
#
# # ========= 5. 模型拟合（Huber）=========
# model_A = HuberRegressor(max_iter=10000).fit(X_A, y_A)
# model_B = HuberRegressor(max_iter=10000).fit(X_B, y_B)
#
# # ========= 6. 评估 =========
# def eval_and_print(tag, model, X, y):
#     y_pred = model.predict(X)
#     mse = mean_squared_error(y, y_pred)
#     r2 = r2_score(y, y_pred)
#     ard = np.mean(np.abs((y - y_pred) / y)) * 100
#     rel_err = np.abs((y_pred - y) / y) * 100
#     within_1pct  = int((rel_err <= 1).sum())
#     within_5pct  = int((rel_err <= 5).sum())
#     within_10pct = int((rel_err <= 10).sum())
#
#     print(f"\n📊 总模型评估（{tag}）：")
#     print(f"R²  = {r2:.4f}")
#     print(f"MSE = {mse:.2f}")
#     print(f"ARD = {ard:.2f}%")
#     print(f"✅ 误差 ≤ 1% : {within_1pct}")
#     print(f"✅ 误差 ≤ 5% : {within_5pct}")
#     print(f"✅ 误差 ≤ 10%: {within_10pct}")
#     return y_pred
#
# y_pred_A = eval_and_print("A=真实ΔCp / 预测ΔT", model_A, X_A, y_A)
# y_pred_B = eval_and_print("B=预测ΔCp / 真实ΔT", model_B, X_B, y_B)
#
# # ========= 7. 输出预测结果 =========
# results_A = pd.DataFrame({
#     "Material_ID": id_A,
#     "Temperature (K)": T_A,
#     "Cp_measured": y_A,
#     "Cp_predicted": y_pred_A
# })
# results_B = pd.DataFrame({
#     "Material_ID": id_B,
#     "Temperature (K)": T_B,
#     "Cp_measured": y_B,
#     "Cp_predicted": y_pred_B
# })
# results_A.to_excel("Cp预测结果_真实ΔCp_预测ΔT.xlsx", index=False)
# results_B.to_excel("Cp预测结果_预测ΔCp_真实ΔT.xlsx", index=False)
# print("✅ 已保存：Cp预测结果_真实ΔCp_预测ΔT.xlsx")
# print("✅ 已保存：Cp预测结果_预测ΔCp_真实ΔT.xlsx")
#
# # ========= 8. 输出系数表 =========
# feature_labels = (
#     list(group_cols) +                # 19 个基团
#     [f"{g}_T" for g in group_cols] +  # 19 个基团 × T
#     ["slope×T"]                       # 1 个新特征
# )
# coef_A = pd.DataFrame({"Feature": feature_labels, "Contribution": model_A.coef_})
# coef_B = pd.DataFrame({"Feature": feature_labels, "Contribution": model_B.coef_})
# coef_A.to_excel("Cp系数表_真实ΔCp_预测ΔT.xlsx", index=False)
# coef_B.to_excel("Cp系数表_预测ΔCp_真实ΔT.xlsx", index=False)
# print("📈 已保存：Cp系数表_真实ΔCp_预测ΔT.xlsx")
# print("📈 已保存：Cp系数表_预测ΔCp_真实ΔT.xlsx")
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ========= 1. 读取数据 =========
# file_path = "heat capacity 207.xlsx"
# df = pd.read_excel(file_path, sheet_name="Sheet1")
# df = df.dropna(subset=[df.columns[0]])
# df[df.columns[0]] = df[df.columns[0]].astype(int)
#
# # ========= 2. 列定义 =========
# group_cols = df.columns[11:30]   # 19个基团列
# temp_cols  = df.columns[30:40]   # 10个温度点
# cp_cols    = df.columns[40:50]   # 10个 Cp 值
# target_column_T1 = 'ASPEN Half Critical T'  # 真实 T1 所在列
#
# # 你给出的“真实四列”：Cp1_true=第10列, Cp2_true=第51列, T1_true=target_column_T1, T2_true=1.5*T1_true
# CP1_TRUE_IDX = 9
# CP2_TRUE_IDX = 50
# T1_TRUE_COL  = target_column_T1
#
# # ========= 2.1 强制数值化（关键修正）=========
# # 将用于建模/计算的列全部转为数值，无法解析的设为 NaN
# df[group_cols] = df[group_cols].apply(pd.to_numeric, errors="coerce")
# df[temp_cols]  = df[temp_cols].apply(pd.to_numeric, errors="coerce")
# df[cp_cols]    = df[cp_cols].apply(pd.to_numeric, errors="coerce")
# # 把第10、51列也数值化
# df.iloc[:, CP1_TRUE_IDX] = pd.to_numeric(df.iloc[:, CP1_TRUE_IDX], errors="coerce")
# df.iloc[:, CP2_TRUE_IDX] = pd.to_numeric(df.iloc[:, CP2_TRUE_IDX], errors="coerce")
# df[T1_TRUE_COL]          = pd.to_numeric(df[T1_TRUE_COL], errors="coerce")
#
# # ========= 3. 子模型训练 =========
# X_groups = df[group_cols]
# # T1 训练有效掩码
# valid_mask = X_groups.notna().all(1) & df[T1_TRUE_COL].notna()
#
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X_groups[valid_mask])
#
# # 用 GradientBoostingRegressor 预测 T1（与你原有设置一致）
# y_T1 = df.loc[valid_mask, T1_TRUE_COL].values
# T1_model = GradientBoostingRegressor(
#     n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
# ).fit(X_poly, y_T1)
#
# # Cp1, Cp2 子模型（用额外两列的真实 Cp）
# Cp1_true_series = df.iloc[:, CP1_TRUE_IDX]
# Cp2_true_series = df.iloc[:, CP2_TRUE_IDX]
# valid_cp_mask = X_groups.notna().all(1) & Cp1_true_series.notna() & Cp2_true_series.notna()
#
# Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups[valid_cp_mask].values, Cp1_true_series[valid_cp_mask].values)
# Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups[valid_cp_mask].values, Cp2_true_series[valid_cp_mask].values)
#
# # ========= 3.1 子模型评估（in-sample）=========
# y_pred_T1 = T1_model.predict(X_poly)
# r2_T1 = r2_score(y_T1, y_pred_T1)
# mse_T1 = mean_squared_error(y_T1, y_pred_T1)
#
# y_Cp1_true = Cp1_true_series[valid_cp_mask]
# y_Cp1_pred = Cp1_model.predict(X_groups[valid_cp_mask].values)
# r2_Cp1 = r2_score(y_Cp1_true, y_Cp1_pred)
# mse_Cp1 = mean_squared_error(y_Cp1_true, y_Cp1_pred)
#
# y_Cp2_true = Cp2_true_series[valid_cp_mask]
# y_Cp2_pred = Cp2_model.predict(X_groups[valid_cp_mask].values)
# r2_Cp2 = r2_score(y_Cp2_true, y_Cp2_pred)
# mse_Cp2 = mean_squared_error(y_Cp2_true, y_Cp2_pred)
#
# print("\n📌 子模型评估结果：")
# print(f"T1_model ->     R²: {r2_T1:.4f} | MSE: {mse_T1:.4f}")
# print(f"Cp1_model ->    R²: {r2_Cp1:.4f} | MSE: {mse_Cp1:.4f}")
# print(f"Cp2_model ->    R²: {r2_Cp2:.4f} | MSE: {mse_Cp2:.4f}")
#
# # ========= 4. 构建训练数据（A/B/C 三种 slope 变体）=========
# X_A, y_A, id_A, T_A = [], [], [], []  # A：真实ΔCp / 预测ΔT
# X_B, y_B, id_B, T_B = [], [], [], []  # B：预测ΔCp / 真实ΔT
# X_C, y_C, id_C, T_C = [], [], [], []  # C：预测ΔCp / 预测ΔT（全预测）
#
# # 对“所有行”的基团特征做 transform（poly 之前在 valid_mask 上 fit 过）
# X_poly_all = poly.transform(X_groups.fillna(0))  # 这里 transform 不会引入 NaN
#
# for i, row in df.iterrows():
#     try:
#         material_id = row.iloc[0]
#
#         # --- 取 Nk，确保数值 & 检查缺失 ---
#         Nk_series = row[group_cols].astype(float)
#         if pd.isna(Nk_series).any():
#             continue
#         Nk = Nk_series.values
#
#         # --- 预测侧 T1/T2 与 Cp1/Cp2 ---
#         T1_pred = float(T1_model.predict(X_poly_all[i:i+1])[0])
#         if not np.isfinite(T1_pred) or T1_pred <= 0:
#             continue
#         T2_pred = 1.5 * T1_pred
#
#         Nk_df_vals = pd.DataFrame([Nk], columns=group_cols).values
#         Cp1_pred = float(Cp1_model.predict(Nk_df_vals)[0])
#         Cp2_pred = float(Cp2_model.predict(Nk_df_vals)[0])
#         if not (np.isfinite(Cp1_pred) and np.isfinite(Cp2_pred)):
#             continue
#
#         # --- 真实侧 Cp1/Cp2/T1/T2 ---
#         Cp1_true = row.iloc[CP1_TRUE_IDX]
#         Cp2_true = row.iloc[CP2_TRUE_IDX]
#         T1_true  = row[T1_TRUE_COL]
#         if not (np.isfinite(Cp1_true) and np.isfinite(Cp2_true) and np.isfinite(T1_true)):
#             continue
#         T2_true  = 1.5 * T1_true
#
#         # 防止除零
#         if T2_pred == T1_pred or T2_true == T1_true:
#             continue
#
#         # --- 三种 slope 变体 ---
#         slope_A = (Cp2_true - Cp1_true) / (T2_pred - T1_pred)   # A：分子真实ΔCp，分母预测ΔT
#         slope_B = (Cp2_pred - Cp1_pred) / (T2_true - T1_true)   # B：分子预测ΔCp，分母真实ΔT
#         slope_C = (Cp2_pred - Cp1_pred) / (T2_pred - T1_pred)   # C：分子预测ΔCp，分母预测ΔT（全预测）
#         if not (np.isfinite(slope_A) and np.isfinite(slope_B) and np.isfinite(slope_C)):
#             continue
#
#         # --- 逐温度点展开 ---
#         temps = row[temp_cols].astype(float).values
#         cps   = row[cp_cols].astype(float).values
#         mask_pts = np.isfinite(temps) & np.isfinite(cps)
#         if not mask_pts.any():
#             continue
#
#         for T, Cp in zip(temps[mask_pts], cps[mask_pts]):
#             feats_A = np.concatenate([Nk, Nk*T, [slope_A*T]])
#             feats_B = np.concatenate([Nk, Nk*T, [slope_B*T]])
#             feats_C = np.concatenate([Nk, Nk*T, [slope_C*T]])
#             X_A.append(feats_A); y_A.append(Cp); id_A.append(material_id); T_A.append(T)
#             X_B.append(feats_B); y_B.append(Cp); id_B.append(material_id); T_B.append(T)
#             X_C.append(feats_C); y_C.append(Cp); id_C.append(material_id); T_C.append(T)
#
#     except Exception as e:
#         print(f"[WARN] row {i} skipped: {e}")
#         continue
#
# X_A = np.asarray(X_A); y_A = np.asarray(y_A)
# X_B = np.asarray(X_B); y_B = np.asarray(y_B)
# X_C = np.asarray(X_C); y_C = np.asarray(y_C)
#
# if X_A.size == 0 or X_B.size == 0 or X_C.size == 0:
#     raise RuntimeError(
#         f"没有可用样本：X_A{X_A.shape}, X_B{X_B.shape}, X_C{X_C.shape}。"
#         "请检查 group/temp/cp 列是否为数值、以及真实/预测列是否存在缺失。"
#     )
#
# # ========= 5. 模型拟合（Huber）=========
# model_A = HuberRegressor(max_iter=10000).fit(X_A, y_A)
# model_B = HuberRegressor(max_iter=10000).fit(X_B, y_B)
# model_C = HuberRegressor(max_iter=10000).fit(X_C, y_C)
#
# # ========= 6. 评估 =========
# def eval_and_print(tag, model, X, y):
#     y_pred = model.predict(X)
#     mse = mean_squared_error(y, y_pred)
#     r2 = r2_score(y, y_pred)
#     ard = np.mean(np.abs((y - y_pred) / y)) * 100
#     rel_err = np.abs((y_pred - y) / y) * 100
#     within_1pct  = int((rel_err <= 1).sum())
#     within_5pct  = int((rel_err <= 5).sum())
#     within_10pct = int((rel_err <= 10).sum())
#
#     print(f"\n📊 总模型评估（{tag}）：")
#     print(f"R²  = {r2:.4f}")
#     print(f"MSE = {mse:.2f}")
#     print(f"ARD = {ard:.2f}%")
#     print(f"✅ 误差 ≤ 1% : {within_1pct}")
#     print(f"✅ 误差 ≤ 5% : {within_5pct}")
#     print(f"✅ 误差 ≤ 10%: {within_10pct}")
#     return y_pred
#
# y_pred_A = eval_and_print("A=真实ΔCp / 预测ΔT", model_A, X_A, y_A)
# y_pred_B = eval_and_print("B=预测ΔCp / 真实ΔT", model_B, X_B, y_B)
# y_pred_C = eval_and_print("C=预测ΔCp / 预测ΔT", model_C, X_C, y_C)
#
# # ========= 7. 输出预测结果 =========
# results_A = pd.DataFrame({
#     "Material_ID": id_A,
#     "Temperature (K)": T_A,
#     "Cp_measured": y_A,
#     "Cp_predicted": y_pred_A
# })
# results_B = pd.DataFrame({
#     "Material_ID": id_B,
#     "Temperature (K)": T_B,
#     "Cp_measured": y_B,
#     "Cp_predicted": y_pred_B
# })
# results_C = pd.DataFrame({
#     "Material_ID": id_C,
#     "Temperature (K)": T_C,
#     "Cp_measured": y_C,
#     "Cp_predicted": y_pred_C
# })
#
# results_A.to_excel("Cp预测结果_真实ΔCp_预测ΔT.xlsx", index=False)
# results_B.to_excel("Cp预测结果_预测ΔCp_真实ΔT.xlsx", index=False)
# results_C.to_excel("Cp预测结果_预测ΔCp_预测ΔT.xlsx", index=False)
# print("✅ 已保存：Cp预测结果_真实ΔCp_预测ΔT.xlsx")
# print("✅ 已保存：Cp预测结果_预测ΔCp_真实ΔT.xlsx")
# print("✅ 已保存：Cp预测结果_预测ΔCp_预测ΔT.xlsx")
#
# # ========= 8. 输出系数表 =========
# feature_labels = (
#     list(group_cols) +                # 19 个基团
#     [f"{g}_T" for g in group_cols] +  # 19 个基团 × T
#     ["slope×T"]                       # 1 个新特征
# )
# coef_A = pd.DataFrame({"Feature": feature_labels, "Contribution": model_A.coef_})
# coef_B = pd.DataFrame({"Feature": feature_labels, "Contribution": model_B.coef_})
# coef_C = pd.DataFrame({"Feature": feature_labels, "Contribution": model_C.coef_})
#
# coef_A.to_excel("Cp系数表_真实ΔCp_预测ΔT.xlsx", index=False)
# coef_B.to_excel("Cp系数表_预测ΔCp_真实ΔT.xlsx", index=False)
# coef_C.to_excel("Cp系数表_预测ΔCp_预测ΔT.xlsx", index=False)
# print("📈 已保存：Cp系数表_真实ΔCp_预测ΔT.xlsx")
# print("📈 已保存：Cp系数表_预测ΔCp_真实ΔT.xlsx")
# print("📈 已保存：Cp系数表_预测ΔCp_预测ΔT.xlsx")


import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# ========= 1. 读取数据 =========
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.dropna(subset=[df.columns[0]])
df[df.columns[0]] = df[df.columns[0]].astype(int)

# ========= 2. 列定义 =========
group_cols = df.columns[11:30]  # 19个基团列
temp_cols = df.columns[30:40]  # 10个温度点
cp_cols = df.columns[40:50]  # 10个 Cp 值
target_column_T1 = 'ASPEN Half Critical T'  # 真实 T1 所在列

# 你给出的"真实四列"：Cp1_true=第10列, Cp2_true=第51列, T1_true=target_column_T1, T2_true=1.5*T1_true
CP1_TRUE_IDX = 9
CP2_TRUE_IDX = 50
T1_TRUE_COL = target_column_T1

# ========= 2.1 强制数值化（关键修正）=========
# 将用于建模/计算的列全部转为数值，无法解析的设为 NaN
df[group_cols] = df[group_cols].apply(pd.to_numeric, errors="coerce")
df[temp_cols] = df[temp_cols].apply(pd.to_numeric, errors="coerce")
df[cp_cols] = df[cp_cols].apply(pd.to_numeric, errors="coerce")
# 把第10、51列也数值化
df.iloc[:, CP1_TRUE_IDX] = pd.to_numeric(df.iloc[:, CP1_TRUE_IDX], errors="coerce")
df.iloc[:, CP2_TRUE_IDX] = pd.to_numeric(df.iloc[:, CP2_TRUE_IDX], errors="coerce")
df[T1_TRUE_COL] = pd.to_numeric(df[T1_TRUE_COL], errors="coerce")

# ========= 3. 子模型训练 =========
X_groups = df[group_cols]
# T1 训练有效掩码
valid_mask = X_groups.notna().all(1) & df[T1_TRUE_COL].notna()

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_groups[valid_mask])

# 用 GradientBoostingRegressor 预测 T1（与你原有设置一致）
y_T1 = df.loc[valid_mask, T1_TRUE_COL].values
T1_model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
).fit(X_poly, y_T1)

# Cp1, Cp2 子模型（用额外两列的真实 Cp）
Cp1_true_series = df.iloc[:, CP1_TRUE_IDX]
Cp2_true_series = df.iloc[:, CP2_TRUE_IDX]
valid_cp_mask = X_groups.notna().all(1) & Cp1_true_series.notna() & Cp2_true_series.notna()

Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups[valid_cp_mask].values, Cp1_true_series[valid_cp_mask].values)
Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups[valid_cp_mask].values, Cp2_true_series[valid_cp_mask].values)

# ========= 3.1 子模型评估（in-sample）=========
y_pred_T1 = T1_model.predict(X_poly)
r2_T1 = r2_score(y_T1, y_pred_T1)
mse_T1 = mean_squared_error(y_T1, y_pred_T1)

y_Cp1_true = Cp1_true_series[valid_cp_mask]
y_Cp1_pred = Cp1_model.predict(X_groups[valid_cp_mask].values)
r2_Cp1 = r2_score(y_Cp1_true, y_Cp1_pred)
mse_Cp1 = mean_squared_error(y_Cp1_true, y_Cp1_pred)

y_Cp2_true = Cp2_true_series[valid_cp_mask]
y_Cp2_pred = Cp2_model.predict(X_groups[valid_cp_mask].values)
r2_Cp2 = r2_score(y_Cp2_true, y_Cp2_pred)
mse_Cp2 = mean_squared_error(y_Cp2_true, y_Cp2_pred)

print("\n📌 子模型评估结果：")
print(f"T1_model ->     R²: {r2_T1:.4f} | MSE: {mse_T1:.4f}")
print(f"Cp1_model ->    R²: {r2_Cp1:.4f} | MSE: {mse_Cp1:.4f}")
print(f"Cp2_model ->    R²: {r2_Cp2:.4f} | MSE: {mse_Cp2:.4f}")

# ========= 4. 构建训练数据（A/B/C/D 四种 slope 变体）=========
X_A, y_A, id_A, T_A = [], [], [], []  # A：真实ΔCp / 预测ΔT
X_B, y_B, id_B, T_B = [], [], [], []  # B：预测ΔCp / 真实ΔT
X_C, y_C, id_C, T_C = [], [], [], []  # C：预测ΔCp / 预测ΔT（全预测）
X_D, y_D, id_D, T_D = [], [], [], []  # D：真实ΔCp / 真实ΔT（完全真实）

# 对"所有行"的基团特征做 transform（poly 之前在 valid_mask 上 fit 过）
X_poly_all = poly.transform(X_groups.fillna(0))  # 这里 transform 不会引入 NaN

for i, row in df.iterrows():
    try:
        material_id = row.iloc[0]

        # --- 取 Nk，确保数值 & 检查缺失 ---
        Nk_series = row[group_cols].astype(float)
        if pd.isna(Nk_series).any():
            continue
        Nk = Nk_series.values

        # --- 预测侧 T1/T2 与 Cp1/Cp2 ---
        T1_pred = float(T1_model.predict(X_poly_all[i:i + 1])[0])
        if not np.isfinite(T1_pred) or T1_pred <= 0:
            continue
        T2_pred = 1.5 * T1_pred

        Nk_df_vals = pd.DataFrame([Nk], columns=group_cols).values
        Cp1_pred = float(Cp1_model.predict(Nk_df_vals)[0])
        Cp2_pred = float(Cp2_model.predict(Nk_df_vals)[0])
        if not (np.isfinite(Cp1_pred) and np.isfinite(Cp2_pred)):
            continue

        # --- 真实侧 Cp1/Cp2/T1/T2 ---
        Cp1_true = row.iloc[CP1_TRUE_IDX]
        Cp2_true = row.iloc[CP2_TRUE_IDX]
        T1_true = row[T1_TRUE_COL]
        if not (np.isfinite(Cp1_true) and np.isfinite(Cp2_true) and np.isfinite(T1_true)):
            continue
        T2_true = 1.5 * T1_true

        # 防止除零
        if T2_pred == T1_pred or T2_true == T1_true:
            continue

        # --- 四种 slope 变体 ---
        slope_A = (Cp2_true - Cp1_true) / (T2_pred - T1_pred)  # A：分子真实ΔCp，分母预测ΔT
        slope_B = (Cp2_pred - Cp1_pred) / (T2_true - T1_true)  # B：分子预测ΔCp，分母真实ΔT
        slope_C = (Cp2_pred - Cp1_pred) / (T2_pred - T1_pred)  # C：分子预测ΔCp，分母预测ΔT（全预测）
        slope_D = (Cp2_true - Cp1_true) / (T2_true - T1_true)  # D：分子真实ΔCp，分母真实ΔT（完全真实）

        if not (np.isfinite(slope_A) and np.isfinite(slope_B) and
                np.isfinite(slope_C) and np.isfinite(slope_D)):
            continue

        # --- 逐温度点展开 ---
        temps = row[temp_cols].astype(float).values
        cps = row[cp_cols].astype(float).values
        mask_pts = np.isfinite(temps) & np.isfinite(cps)
        if not mask_pts.any():
            continue

        for T, Cp in zip(temps[mask_pts], cps[mask_pts]):
            feats_A = np.concatenate([Nk, Nk * T, [slope_A * T]])
            feats_B = np.concatenate([Nk, Nk * T, [slope_B * T]])
            feats_C = np.concatenate([Nk, Nk * T, [slope_C * T]])
            feats_D = np.concatenate([Nk, Nk * T, [slope_D * T]])  # 新增D变体

            X_A.append(feats_A);
            y_A.append(Cp);
            id_A.append(material_id);
            T_A.append(T)
            X_B.append(feats_B);
            y_B.append(Cp);
            id_B.append(material_id);
            T_B.append(T)
            X_C.append(feats_C);
            y_C.append(Cp);
            id_C.append(material_id);
            T_C.append(T)
            X_D.append(feats_D);
            y_D.append(Cp);
            id_D.append(material_id);
            T_D.append(T)  # 新增D变体

    except Exception as e:
        print(f"[WARN] row {i} skipped: {e}")
        continue

X_A = np.asarray(X_A);
y_A = np.asarray(y_A)
X_B = np.asarray(X_B);
y_B = np.asarray(y_B)
X_C = np.asarray(X_C);
y_C = np.asarray(y_C)
X_D = np.asarray(X_D);
y_D = np.asarray(y_D)  # 新增D变体

if X_A.size == 0 or X_B.size == 0 or X_C.size == 0 or X_D.size == 0:
    raise RuntimeError(
        f"没有可用样本：X_A{X_A.shape}, X_B{X_B.shape}, X_C{X_C.shape}, X_D{X_D.shape}。"
        "请检查 group/temp/cp 列是否为数值、以及真实/预测列是否存在缺失。"
    )

# ========= 5. 模型拟合（Huber）=========
model_A = HuberRegressor(max_iter=10000).fit(X_A, y_A)
model_B = HuberRegressor(max_iter=10000).fit(X_B, y_B)
model_C = HuberRegressor(max_iter=10000).fit(X_C, y_C)
model_D = HuberRegressor(max_iter=10000).fit(X_D, y_D)  # 新增D模型


# ========= 6. 评估 =========
def eval_and_print(tag, model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    ard = np.mean(np.abs((y - y_pred) / y)) * 100
    rel_err = np.abs((y_pred - y) / y) * 100
    within_1pct = int((rel_err <= 1).sum())
    within_5pct = int((rel_err <= 5).sum())
    within_10pct = int((rel_err <= 10).sum())

    print(f"\n📊 总模型评估（{tag}）：")
    print(f"R²  = {r2:.4f}")
    print(f"MSE = {mse:.2f}")
    print(f"ARD = {ard:.2f}%")
    print(f"✅ 误差 ≤ 1% : {within_1pct}")
    print(f"✅ 误差 ≤ 5% : {within_5pct}")
    print(f"✅ 误差 ≤ 10%: {within_10pct}")
    return y_pred


y_pred_A = eval_and_print("A=真实ΔCp / 预测ΔT", model_A, X_A, y_A)
y_pred_B = eval_and_print("B=预测ΔCp / 真实ΔT", model_B, X_B, y_B)
y_pred_C = eval_and_print("C=预测ΔCp / 预测ΔT", model_C, X_C, y_C)
y_pred_D = eval_and_print("D=真实ΔCp / 真实ΔT", model_D, X_D, y_D)  # 新增D评估

# ========= 7. 输出预测结果 =========
results_A = pd.DataFrame({
    "Material_ID": id_A,
    "Temperature (K)": T_A,
    "Cp_measured": y_A,
    "Cp_predicted": y_pred_A
})
results_B = pd.DataFrame({
    "Material_ID": id_B,
    "Temperature (K)": T_B,
    "Cp_measured": y_B,
    "Cp_predicted": y_pred_B
})
results_C = pd.DataFrame({
    "Material_ID": id_C,
    "Temperature (K)": T_C,
    "Cp_measured": y_C,
    "Cp_predicted": y_pred_C
})
results_D = pd.DataFrame({
    "Material_ID": id_D,
    "Temperature (K)": T_D,
    "Cp_measured": y_D,
    "Cp_predicted": y_pred_D
})

results_A.to_excel("Cp预测结果_真实ΔCp_预测ΔT.xlsx", index=False)
results_B.to_excel("Cp预测结果_预测ΔCp_真实ΔT.xlsx", index=False)
results_C.to_excel("Cp预测结果_预测ΔCp_预测ΔT.xlsx", index=False)
results_D.to_excel("Cp预测结果_真实ΔCp_真实ΔT.xlsx", index=False)  # 新增D结果
print("✅ 已保存：Cp预测结果_真实ΔCp_预测ΔT.xlsx")
print("✅ 已保存：Cp预测结果_预测ΔCp_真实ΔT.xlsx")
print("✅ 已保存：Cp预测结果_预测ΔCp_预测ΔT.xlsx")
print("✅ 已保存：Cp预测结果_真实ΔCp_真实ΔT.xlsx")  # 新增D输出

# ========= 8. 输出系数表 =========
feature_labels = (
        list(group_cols) +  # 19 个基团
        [f"{g}_T" for g in group_cols] +  # 19 个基团 × T
        ["slope×T"]  # 1 个新特征
)
coef_A = pd.DataFrame({"Feature": feature_labels, "Contribution": model_A.coef_})
coef_B = pd.DataFrame({"Feature": feature_labels, "Contribution": model_B.coef_})
coef_C = pd.DataFrame({"Feature": feature_labels, "Contribution": model_C.coef_})
coef_D = pd.DataFrame({"Feature": feature_labels, "Contribution": model_D.coef_})

coef_A.to_excel("Cp系数表_真实ΔCp_预测ΔT.xlsx", index=False)
coef_B.to_excel("Cp系数表_预测ΔCp_真实ΔT.xlsx", index=False)
coef_C.to_excel("Cp系数表_预测ΔCp_预测ΔT.xlsx", index=False)
coef_D.to_excel("Cp系数表_真实ΔCp_真实ΔT.xlsx", index=False)
print("📈 已保存：Cp系数表_真实ΔCp_预测ΔT.xlsx")
print("📈 已保存：Cp系数表_预测ΔCp_真实ΔT.xlsx")
print("📈 已保存：Cp系数表_预测ΔCp_预测ΔT.xlsx")
print("📈 已保存：Cp系数表_真实ΔCp_真实ΔT.xlsx")