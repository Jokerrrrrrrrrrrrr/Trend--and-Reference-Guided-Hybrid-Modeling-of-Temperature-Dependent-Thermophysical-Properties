# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures
#
# # ========= 1. 读取数据 =========
# file_path = "heat capacity 207.xlsx"
# df = pd.read_excel(file_path, sheet_name="Sheet1")
# df = df.dropna(subset=[df.columns[0]])
# df[df.columns[0]] = df[df.columns[0]].astype(int)
#
# # ========= 2. 列定义 =========
# group_cols = df.columns[11:30]  # 19个基团列
# temp_cols = df.columns[30:40]  # 10个温度点
# cp_cols = df.columns[40:50]  # 10个 Cp 值
# target_column_T1 = 'ASPEN Half Critical T'
#
# # ========= 3. 子模型训练 =========
# X_groups = df[group_cols]
# valid_mask = ~df[target_column_T1].isna()
#
# # 生成多项式特征
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X_groups[valid_mask])
#
# # 计算相似度时使用 log1p 转换后的基团向量
# group_vectors_log = np.log1p(X_groups)
#
# # 计算相似度函数
# def compute_msc(target_vector, reference_vector, alpha=np.e):
#     target_vector = np.array(target_vector)
#     reference_vector = np.array(reference_vector)
#     min_vals = np.minimum(target_vector, reference_vector)
#     max_vals = np.maximum(target_vector, reference_vector)
#     sum_min = np.sum(min_vals)
#     sum_max = np.sum(max_vals)
#     msc = (alpha ** sum_min - 1) / (alpha ** sum_max - 1)
#     return msc
#
#
# # 改用相似度回归预测 T1
# y_T1 = df.loc[valid_mask, target_column_T1].values
# similarity_threshold = 0 # 设置相似度阈值
#
# # 计算目标分子与其他分子的相似度
# T1_model = []
# for i, target_vector in enumerate(X_poly):
#     similarities_i = []
#     for j, ref_vector in enumerate(X_poly):
#         if i != j:  # 排除自身
#             similarity = compute_msc(group_vectors_log.iloc[i], group_vectors_log.iloc[j])  # 使用对数转换后的向量
#             similarities_i.append((j, similarity))
#
#     # 选择相似度大于阈值的参考分子
#     selected_indices = [j for j, similarity in similarities_i if similarity > similarity_threshold]
#
#     # 如果没有符合条件的分子，则选择相似度最高的一个
#     if len(selected_indices) == 0:
#         max_sim_idx = max(similarities_i, key=lambda x: x[1])[0]
#         selected_indices.append(max_sim_idx)
#
#     # 使用相似度较高的分子训练模型
#     if len(selected_indices) > 0:
#         model = HuberRegressor(max_iter=9000)
#         model.fit(X_poly[selected_indices], y_T1[selected_indices])
#         T1_pred = model.predict([X_poly[i]])  # 预测目标分子的T1
#         T1_model.append(T1_pred[0])
#
# # Cp1, Cp2 使用相似度回归模型
# Cp1_model = []
# Cp2_model = []
# for i, target_vector in enumerate(X_groups[valid_mask]):
#     similarities_i = []
#     for j, ref_vector in enumerate(X_groups[valid_mask]):
#         if i != j:  # 排除自身
#             similarity = compute_msc(group_vectors_log.iloc[i], group_vectors_log.iloc[j])  # 使用对数转换后的向量
#             similarities_i.append((j, similarity))
#
#     # 选择相似度大于阈值的参考分子
#     selected_indices = [j for j, similarity in similarities_i if similarity > similarity_threshold]
#
#     # 如果没有符合条件的分子，则选择相似度最高的一个
#     if len(selected_indices) == 0:
#         max_sim_idx = max(similarities_i, key=lambda x: x[1])[0]
#         selected_indices.append(max_sim_idx)
#
#     if len(selected_indices) > 0:
#         model1 = HuberRegressor(max_iter=9000)
#         model1.fit(X_groups.iloc[selected_indices], df.loc[selected_indices, df.columns[9]])  # 用Cp1的数据训练
#         Cp1_pred = model1.predict([X_groups.iloc[i]])  # 预测目标分子的Cp1
#         Cp1_model.append(Cp1_pred[0])
#
#         model2 = HuberRegressor(max_iter=9000)
#         model2.fit(X_groups.iloc[selected_indices], df.loc[selected_indices, df.columns[50]])  # 用Cp2的数据训练
#         Cp2_pred = model2.predict([X_groups.iloc[i]])  # 预测目标分子的Cp2
#         Cp2_model.append(Cp2_pred[0])
#
# # ========= 3.1 子模型评估 =========
# y_pred_T1 = np.array(T1_model)
# r2_T1 = r2_score(y_T1, y_pred_T1)
# mse_T1 = mean_squared_error(y_T1, y_pred_T1)
#
# y_Cp1_true = df.iloc[:, 9]
# y_Cp1_pred = np.array(Cp1_model)
# r2_Cp1 = r2_score(y_Cp1_true, y_Cp1_pred)
# mse_Cp1 = mean_squared_error(y_Cp1_true, y_Cp1_pred)
#
# y_Cp2_true = df.iloc[:, 50]
# y_Cp2_pred = np.array(Cp2_model)
# r2_Cp2 = r2_score(y_Cp2_true, y_Cp2_pred)
# mse_Cp2 = mean_squared_error(y_Cp2_true, y_Cp2_pred)
#
# print("\n📌 子模型评估结果：")
# print(f"T1_model ->     R²: {r2_T1:.4f} | MSE: {mse_T1:.4f}")
# print(f"Cp1_model ->    R²: {r2_Cp1:.4f} | MSE: {mse_Cp1:.4f}")
# print(f"Cp2_model ->    R²: {r2_Cp2:.4f} | MSE: {mse_Cp2:.4f}")
#
# # ========= 4. 构建训练数据 =========
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
#         # 新模型：直接预测 T1（无需 log 和 exp）
#         T1 = T1_model[i]
#         if T1 <= 0 or np.isnan(T1):
#             continue
#         T2 = T1 * 1.5
#         Cp1 = Cp1_model[i]
#         Cp2 = Cp2_model[i]
#         slope = (Cp2 - Cp1) / (T2 - T1)
#     except:
#         continue
#
#     for T, Cp in zip(temps, cps):
#         if np.isnan(T) or np.isnan(Cp):
#             continue
#
#         features = np.concatenate([
#             Nk,  # 19 个基团
#             Nk * T,  # 19 个交互项
#             [slope * T]  # slope × T
#         ])
#
#         X_total.append(features)
#         y_total.append(Cp)
#         material_ids.append(material_id)
#         temperatures.append(T)
#
# # ========= 5. 模型拟合（Huber） =========
# X_total = np.array(X_total)
# y_total = np.array(y_total)
#
# model = HuberRegressor(max_iter=10000).fit(X_total, y_total)
#
# # ========= 6. 模型评估 =========
# y_pred = model.predict(X_total)
# mse = mean_squared_error(y_total, y_pred)
# r2 = r2_score(y_total, y_pred)
# ard = np.mean(np.abs((y_total - y_pred) / y_total)) * 100
#
# # === 新增误差统计 ===
# relative_error = np.abs((y_pred - y_total) / y_total) * 100
# within_1pct = np.sum(relative_error <= 1)
# within_5pct = np.sum(relative_error <= 5)
# within_10pct = np.sum(relative_error <= 10)
#
# print("\n📊 总模型评估（含 slope×T 特征）：")
# print(f"R²  = {r2:.4f}")
# print(f"MSE = {mse:.2f}")
# print(f"ARD = {ard:.2f}%")
# print(f"✅ 误差 ≤ 1% 的数据点数量: {within_1pct}")
# print(f"✅ 误差 ≤ 5% 的数据点数量: {within_5pct}")
# print(f"✅ 误差 ≤ 10% 的数据点数量: {within_10pct}")
#
# # ========= 7. 输出预测结果 =========
# results = pd.DataFrame({
#     "Material_ID": material_ids,
#     "Temperature (K)": temperatures,
#     "Cp_measured": y_total,
#     "Cp_predicted": y_pred
# })
# results.to_excel("Cp预测结果_slopeT特征_相似度回归.xlsx", index=False)
# print("✅ 已保存预测结果为: Cp预测结果_slopeT特征_相似度回归.xlsx")
#
# # ========= 8. 输出系数表 =========
# feature_labels = (
#         list(group_cols) +  # 19 个基团
#         [f"{g}_T" for g in group_cols] +  # 19 个基团 × T
#         ["slope×T"]  # 1 个新特征
# )
#
# coefficients = pd.DataFrame({
#     "Feature": feature_labels,
#     "Contribution": model.coef_
# })
# coefficients.to_excel("Cp系数表_slopeT特征_相似度回归.xlsx", index=False)
# print("📈 已保存模型系数为: Cp系数表_slopeT特征_相似度回归.xlsx")
#
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ========= 1. 读取数据 =========
# file_path = "heat capacity 207.xlsx"
# df = pd.read_excel(file_path, sheet_name="Sheet1")
# df = df.dropna(subset=[df.columns[0]])
# df[df.columns[0]] = df[df.columns[0]].astype(int)
#
# # ========= 2. 列定义 =========
# group_cols = df.columns[11:30]  # 19个基团列
# temp_cols = df.columns[30:40]  # 10个温度点
# cp_cols = df.columns[40:50]  # 10个 Cp 值
# target_column_T1 = 'ASPEN Half Critical T'
#
# # ========= 3. 子模型训练 =========
# X_groups = df[group_cols]  # 使用原始基团向量
# valid_mask = ~df[target_column_T1].isna()
#
# # 计算相似度时使用 log1p 转换后的基团向量
# group_vectors_log = np.log1p(X_groups)
#
# # 计算相似度函数
# def compute_msc(target_vector, reference_vector, alpha=np.e):
#     target_vector = np.array(target_vector)
#     reference_vector = np.array(reference_vector)
#     min_vals = np.minimum(target_vector, reference_vector)
#     max_vals = np.maximum(target_vector, reference_vector)
#     sum_min = np.sum(min_vals)
#     sum_max = np.sum(max_vals)
#     msc = (alpha ** sum_min - 1) / (alpha ** sum_max - 1)
#     return msc
#
#
# # 改用相似度回归预测 T1
# y_T1 = df.loc[valid_mask, target_column_T1].values
# similarity_threshold = 0  # 设置相似度阈值
#
# print("开始训练T1模型...")
#
# T1_predictions = np.full(len(df), np.nan)  # 为所有分子创建数组，初始为NaN
#
# # 获取有有效T1值的分子索引
# valid_indices = valid_mask[valid_mask].index.tolist()
#
# # 只为有有效T1值的分子预测T1，但从所有分子中选择相似分子
# for i, orig_i_idx in enumerate(valid_indices):
#     if i % 10 == 0:
#         print(f"处理T1第 {i}/{len(valid_indices)} 个分子...")
#
#     similarities_i = []
#     # 从所有分子中选择相似分子（不仅仅是19个有T1值的）
#     for j in range(len(df)):
#         if j != orig_i_idx:  # 排除自身
#             similarity = compute_msc(group_vectors_log.iloc[orig_i_idx], group_vectors_log.iloc[j])
#             similarities_i.append((j, similarity))
#
#     # 选择相似度大于阈值的参考分子（从所有分子中选择）
#     selected_indices = [j for j, similarity in similarities_i if similarity > similarity_threshold]
#
#     # 如果没有符合条件的分子，则选择相似度最高的一个
#     if len(selected_indices) == 0 and len(similarities_i) > 0:
#         selected_indices.append(max(similarities_i, key=lambda x: x[1])[0])
#
#     # 只使用有T1值的相似分子进行训练
#     valid_selected_indices = [idx for idx in selected_indices if valid_mask[idx]]
#
#     if len(valid_selected_indices) > 0:
#         # 获取这些分子在原始基团向量中的索引
#         poly_indices = [valid_indices.index(idx) for idx in valid_selected_indices if idx in valid_indices]
#
#         if len(poly_indices) > 0:
#             model = HuberRegressor(max_iter=9000000)
#             model.fit(X_groups.iloc[poly_indices], y_T1[poly_indices])
#             T1_pred = model.predict([X_groups.iloc[orig_i_idx]])[0]
#             T1_predictions[orig_i_idx] = T1_pred
#     else:
#         # 备用方案：使用平均值
#         T1_predictions[orig_i_idx] = np.mean(y_T1)
#
# print("开始训练Cp1和Cp2模型...")
# Cp1_predictions = np.full(len(df), np.nan)
# Cp2_predictions = np.full(len(df), np.nan)
#
# # 为所有分子预测Cp1和Cp2
# for i in range(len(df)):
#     if i % 10 == 0:
#         print(f"处理Cp第 {i}/{len(df)} 个分子...")
#
#     similarities_i = []
#     for j in range(len(df)):
#         if i != j:  # 排除自身
#             similarity = compute_msc(group_vectors_log.iloc[i], group_vectors_log.iloc[j])
#             similarities_i.append((j, similarity))
#
#     # 选择相似分子
#     selected_indices = [j for j, similarity in similarities_i if similarity > similarity_threshold]
#     if len(selected_indices) == 0 and len(similarities_i) > 0:
#         selected_indices.append(max(similarities_i, key=lambda x: x[1])[0])
#
#     if len(selected_indices) > 0:
#         # 转换为numpy数组避免特征名称警告
#         X_array = X_groups.values
#
#         # 训练Cp1模型
#         model1 = HuberRegressor(max_iter=9000000)
#         model1.fit(X_array[selected_indices], df.iloc[selected_indices, 9])
#         Cp1_pred = model1.predict([X_array[i]])[0]
#         Cp1_predictions[i] = Cp1_pred
#
#         # 训练Cp2模型
#         model2 = HuberRegressor(max_iter=9000000)
#         model2.fit(X_array[selected_indices], df.iloc[selected_indices, 50])
#         Cp2_pred = model2.predict([X_array[i]])[0]
#         Cp2_predictions[i] = Cp2_pred
#     else:
#         # 备用方案：使用平均值
#         Cp1_predictions[i] = np.nanmean(df.iloc[:, 9])
#         Cp2_predictions[i] = np.nanmean(df.iloc[:, 50])
#
# # ========= 3.1 子模型评估 =========
# # 只评估有有效值的部分
# valid_T1_mask = ~np.isnan(T1_predictions) & valid_mask
# y_pred_T1 = T1_predictions[valid_T1_mask]
# y_true_T1 = df.loc[valid_T1_mask, target_column_T1].values
#
# r2_T1 = r2_score(y_true_T1, y_pred_T1) if len(y_true_T1) > 0 else np.nan
# mse_T1 = mean_squared_error(y_true_T1, y_pred_T1) if len(y_true_T1) > 0 else np.nan
#
# # Cp1和Cp2评估所有分子
# valid_Cp_mask = ~np.isnan(Cp1_predictions) & ~np.isnan(df.iloc[:, 9])
# y_Cp1_true = df.iloc[valid_Cp_mask, 9].values
# y_Cp1_pred = Cp1_predictions[valid_Cp_mask]
# r2_Cp1 = r2_score(y_Cp1_true, y_Cp1_pred) if len(y_Cp1_true) > 0 else np.nan
# mse_Cp1 = mean_squared_error(y_Cp1_true, y_Cp1_pred) if len(y_Cp1_true) > 0 else np.nan
#
# valid_Cp2_mask = ~np.isnan(Cp2_predictions) & ~np.isnan(df.iloc[:, 50])
# y_Cp2_true = df.iloc[valid_Cp2_mask, 50].values
# y_Cp2_pred = Cp2_predictions[valid_Cp2_mask]
# r2_Cp2 = r2_score(y_Cp2_true, y_Cp2_pred) if len(y_Cp2_true) > 0 else np.nan
# mse_Cp2 = mean_squared_error(y_Cp2_true, y_Cp2_pred) if len(y_Cp2_true) > 0 else np.nan
#
# print("\n📌 子模型评估结果：")
# print(f"T1_model ->     样本数: {len(y_true_T1)}, R²: {r2_T1:.4f} | MSE: {mse_T1:.4f}")
# print(f"Cp1_model ->    样本数: {len(y_Cp1_true)}, R²: {r2_Cp1:.4f} | MSE: {mse_Cp1:.4f}")
# print(f"Cp2_model ->    样本数: {len(y_Cp2_true)}, R²: {r2_Cp2:.4f} | MSE: {mse_Cp2:.4f}")
#
# # ========= 4. 构建训练数据 =========
# X_total, y_total, material_ids, temperatures = [], [], [], []
#
# for i, row in df.iterrows():
#     material_id = row.iloc[0]
#     Nk = row[group_cols].values
#     temps = row[temp_cols].values
#     cps = row[cp_cols].values
#
#     # 检查是否有预测值
#     if np.isnan(T1_predictions[i]) or np.isnan(Cp1_predictions[i]) or np.isnan(Cp2_predictions[i]):
#         continue
#
#     try:
#         T1 = T1_predictions[i]
#         if T1 <= 0:
#             continue
#         T2 = T1 * 1.5
#         Cp1 = Cp1_predictions[i]
#         Cp2 = Cp2_predictions[i]
#         slope = (Cp2 - Cp1) / (T2 - T1) if (T2 - T1) != 0 else 0
#     except:
#         continue
#
#     for T, Cp in zip(temps, cps):
#         if np.isnan(T) or np.isnan(Cp):
#             continue
#
#         features = np.concatenate([Nk, Nk * T, [slope * T]])
#
#         X_total.append(features)
#         y_total.append(Cp)
#         material_ids.append(material_id)
#         temperatures.append(T)
#
# # ========= 5. 模型拟合（Huber） =========
# if len(X_total) > 0:
#     X_total = np.array(X_total)
#     y_total = np.array(y_total)
#
#     model = HuberRegressor(max_iter=10000).fit(X_total, y_total)
#
#     # ========= 6. 模型评估 =========
#     y_pred = model.predict(X_total)
#     mse = mean_squared_error(y_total, y_pred)
#     r2 = r2_score(y_total, y_pred)
#     ard = np.mean(np.abs((y_total - y_pred) / y_total)) * 100
#
#     # === 新增误差统计 ===
#     relative_error = np.abs((y_pred - y_total) / y_total) * 100
#     within_1pct = np.sum(relative_error <= 1)
#     within_5pct = np.sum(relative_error <= 5)
#     within_10pct = np.sum(relative_error <= 10)
#
#     print("\n📊 总模型评估（含 slope×T 特征）：")
#     print(f"R²  = {r2:.4f}")
#     print(f"MSE = {mse:.2f}")
#     print(f"ARD = {ard:.2f}%")
#     print(f"✅ 误差 ≤ 1% 的数据点数量: {within_1pct}")
#     print(f"✅ 误差 ≤ 5% 的数据点数量: {within_5pct}")
#     print(f"✅ 误差 ≤ 10% 的数据点数量: {within_10pct}")
#
#     # ========= 7. 输出预测结果 =========
#     results = pd.DataFrame({
#         "Material_ID": material_ids,
#         "Temperature (K)": temperatures,
#         "Cp_measured": y_total,
#         "Cp_predicted": y_pred
#     })
#     results.to_excel("Cp预测结果_slopeT特征_相似度回归.xlsx", index=False)
#     print("✅ 已保存预测结果为: Cp预测结果_slopeT特征_相似度回归.xlsx")
#
#     # ========= 8. 输出系数表 =========
#     feature_labels = (
#             list(group_cols) +  # 19 个基团
#             [f"{g}_T" for g in group_cols] +  # 19 个基团 × T
#             ["slope×T"]  # 1 个新特征
#     )
#
#     coefficients = pd.DataFrame({
#         "Feature": feature_labels,
#         "Contribution": model.coef_
#     })
#     coefficients.to_excel("Cp系数表_slopeT特征_相似度回归.xlsx", index=False)
#     print("📈 已保存模型系数为: Cp系数表_slopeT特征_相似度回归.xlsx")
# else:
#     print("❌ 没有有效的数据用于训练总模型")

import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ========= 1. 读取数据 =========
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.dropna(subset=[df.columns[0]])
df[df.columns[0]] = df[df.columns[0]].astype(int)

# ========= 2. 列定义 =========
group_cols = df.columns[11:30]  # 19个基团列
temp_cols = df.columns[30:40]  # 10个温度点
cp_cols = df.columns[40:50]  # 10个 Cp 值
target_column_T1 = 'ASPEN Half Critical T'

# ========= 3. 子模型训练 =========
X_groups = df[group_cols]  # 使用原始基团向量
valid_mask = ~df[target_column_T1].isna()

# 计算相似度时使用 log1p 转换后的基团向量
group_vectors_log = np.log1p(X_groups)


# 计算相似度函数
def compute_msc(target_vector, reference_vector, alpha=np.e):
    target_vector = np.array(target_vector)
    reference_vector = np.array(reference_vector)
    min_vals = np.minimum(target_vector, reference_vector)
    max_vals = np.maximum(target_vector, reference_vector)
    sum_min = np.sum(min_vals)
    sum_max = np.sum(max_vals)
    if sum_max == 0:  # 添加除以零保护
        return 0
    msc = (alpha ** sum_min - 1) / (alpha ** sum_max - 1)
    return msc


# 改用相似度回归预测 T1
y_T1 = df.loc[valid_mask, target_column_T1].values
similarity_threshold = 0.9  # 降低相似度阈值，获得更多样本

print("开始训练T1模型...")
T1_predictions = np.full(len(df), np.nan)  # 为所有分子创建数组，初始为NaN

# 获取有有效T1值的分子索引
valid_indices = valid_mask[valid_mask].index.tolist()

# 只为有有效T1值的分子预测T1，但从所有分子中选择相似分子
for i, orig_i_idx in enumerate(valid_indices):
    if i % 10 == 0:
        print(f"处理T1第 {i}/{len(valid_indices)} 个分子...")

    similarities_i = []
    # 从所有分子中选择相似分子
    for j in range(len(df)):
        if j != orig_i_idx:  # 排除自身
            similarity = compute_msc(group_vectors_log.iloc[orig_i_idx], group_vectors_log.iloc[j])
            similarities_i.append((j, similarity))

    # 选择相似度大于阈值的参考分子
    selected_indices = [j for j, similarity in similarities_i if similarity > similarity_threshold]

    # 如果没有符合条件的分子，则选择相似度最高的一个
    if len(selected_indices) == 0 and len(similarities_i) > 0:
        selected_indices.append(max(similarities_i, key=lambda x: x[1])[0])

    # 只使用有T1值的相似分子进行训练
    valid_selected_indices = [idx for idx in selected_indices if valid_mask[idx]]

    # 确保至少有一定数量的训练样本
    if len(valid_selected_indices) < 5 and len(valid_indices) > 5:
        # 从所有有T1值的分子中补充一些
        additional_indices = [idx for idx in valid_indices if idx != orig_i_idx and idx not in valid_selected_indices]
        if len(additional_indices) > 0:
            valid_selected_indices.extend(additional_indices[:min(5, len(additional_indices))])

    if len(valid_selected_indices) > 0:
        try:
            # 数据标准化
            scaler = StandardScaler()
            X_train = X_groups.iloc[valid_selected_indices].values
            X_test = X_groups.iloc[orig_i_idx:orig_i_idx + 1].values

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 使用合理的Huber回归参数
            model = HuberRegressor(max_iter=10000, epsilon=1.5, alpha=0.0001)
            y_train = df.loc[valid_selected_indices, target_column_T1].values

            model.fit(X_train_scaled, y_train)
            T1_pred = model.predict(X_test_scaled)[0]
            T1_predictions[orig_i_idx] = T1_pred

        except Exception as e:
            print(f"分子 {orig_i_idx} 训练失败: {e}")
            # 备用方案：使用相似分子的平均值
            T1_predictions[orig_i_idx] = np.mean(df.loc[valid_selected_indices, target_column_T1]) if len(
                valid_selected_indices) > 0 else np.mean(y_T1)
    else:
        # 备用方案：使用平均值
        T1_predictions[orig_i_idx] = np.mean(y_T1)

print("开始训练Cp1和Cp2模型...")
Cp1_predictions = np.full(len(df), np.nan)
Cp2_predictions = np.full(len(df), np.nan)

# 为所有分子预测Cp1和Cp2
for i in range(len(df)):
    if i % 10 == 0:
        print(f"处理Cp第 {i}/{len(df)} 个分子...")

    similarities_i = []
    for j in range(len(df)):
        if i != j:  # 排除自身
            similarity = compute_msc(group_vectors_log.iloc[i], group_vectors_log.iloc[j])
            similarities_i.append((j, similarity))

    # 选择相似分子
    selected_indices = [j for j, similarity in similarities_i if similarity > similarity_threshold]
    if len(selected_indices) == 0 and len(similarities_i) > 0:
        selected_indices.append(max(similarities_i, key=lambda x: x[1])[0])

    if len(selected_indices) > 0:
        try:
            # 数据标准化
            scaler = StandardScaler()
            X_train = X_groups.iloc[selected_indices].values
            X_test = X_groups.iloc[i:i + 1].values

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 训练Cp1模型
            model1 = HuberRegressor(max_iter=10000, epsilon=1.5, alpha=0.0001)
            y_train_cp1 = df.iloc[selected_indices, 9].values
            model1.fit(X_train_scaled, y_train_cp1)
            Cp1_pred = model1.predict(X_test_scaled)[0]
            Cp1_predictions[i] = Cp1_pred

            # 训练Cp2模型
            model2 = HuberRegressor(max_iter=10000, epsilon=1.5, alpha=0.0001)
            y_train_cp2 = df.iloc[selected_indices, 50].values
            model2.fit(X_train_scaled, y_train_cp2)
            Cp2_pred = model2.predict(X_test_scaled)[0]
            Cp2_predictions[i] = Cp2_pred

        except Exception as e:
            print(f"分子 {i} 的Cp预测失败: {e}")
            Cp1_predictions[i] = np.nanmean(df.iloc[:, 9])
            Cp2_predictions[i] = np.nanmean(df.iloc[:, 50])
    else:
        # 备用方案：使用平均值
        Cp1_predictions[i] = np.nanmean(df.iloc[:, 9])
        Cp2_predictions[i] = np.nanmean(df.iloc[:, 50])

# ========= 3.1 子模型评估 =========
# 只评估有有效值的部分
valid_T1_mask = ~np.isnan(T1_predictions) & valid_mask
y_pred_T1 = T1_predictions[valid_T1_mask]
y_true_T1 = df.loc[valid_T1_mask, target_column_T1].values

r2_T1 = r2_score(y_true_T1, y_pred_T1) if len(y_true_T1) > 0 else np.nan
mse_T1 = mean_squared_error(y_true_T1, y_pred_T1) if len(y_true_T1) > 0 else np.nan

# Cp1和Cp2评估所有分子 - 修复索引错误
# 方法1：使用 numpy 数组进行布尔索引
cp1_valid_mask = ~np.isnan(Cp1_predictions) & ~np.isnan(df.iloc[:, 9].values)
y_Cp1_true = df.iloc[:, 9].values[cp1_valid_mask]
y_Cp1_pred = Cp1_predictions[cp1_valid_mask]

cp2_valid_mask = ~np.isnan(Cp2_predictions) & ~np.isnan(df.iloc[:, 50].values)
y_Cp2_true = df.iloc[:, 50].values[cp2_valid_mask]
y_Cp2_pred = Cp2_predictions[cp2_valid_mask]

r2_Cp1 = r2_score(y_Cp1_true, y_Cp1_pred) if len(y_Cp1_true) > 0 else np.nan
mse_Cp1 = mean_squared_error(y_Cp1_true, y_Cp1_pred) if len(y_Cp1_true) > 0 else np.nan

r2_Cp2 = r2_score(y_Cp2_true, y_Cp2_pred) if len(y_Cp2_true) > 0 else np.nan
mse_Cp2 = mean_squared_error(y_Cp2_true, y_Cp2_pred) if len(y_Cp2_true) > 0 else np.nan

print("\n📌 子模型评估结果：")
print(f"T1_model ->     样本数: {len(y_true_T1)}, R²: {r2_T1:.4f} | MSE: {mse_T1:.4f}")
print(f"Cp1_model ->    样本数: {len(y_Cp1_true)}, R²: {r2_Cp1:.4f} | MSE: {mse_Cp1:.4f}")
print(f"Cp2_model ->    样本数: {len(y_Cp2_true)}, R²: {r2_Cp2:.4f} | MSE: {mse_Cp2:.4f}")

# ========= 4. 构建训练数据 =========
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    cps = row[cp_cols].values

    # 检查是否有预测值
    if np.isnan(T1_predictions[i]) or np.isnan(Cp1_predictions[i]) or np.isnan(Cp2_predictions[i]):
        continue

    try:
        T1 = T1_predictions[i]
        if T1 <= 0:
            continue
        T2 = T1 * 1.5
        Cp1 = Cp1_predictions[i]
        Cp2 = Cp2_predictions[i]
        slope = (Cp2 - Cp1) / (T2 - T1) if (T2 - T1) != 0 else 0
    except:
        continue

    for T, Cp in zip(temps, cps):
        if np.isnan(T) or np.isnan(Cp):
            continue

        features = np.concatenate([Nk, Nk * T, [slope * T]])

        X_total.append(features)
        y_total.append(Cp)
        material_ids.append(material_id)
        temperatures.append(T)

# ========= 5. 模型拟合（Huber） =========
if len(X_total) > 0:
    X_total = np.array(X_total)
    y_total = np.array(y_total)

    # 对总模型特征也进行标准化
    scaler_total = StandardScaler()
    X_total_scaled = scaler_total.fit_transform(X_total)

    model = HuberRegressor(max_iter=10000, epsilon=1.5, alpha=0.0001).fit(X_total_scaled, y_total)

    # ========= 6. 模型评估 =========
    y_pred = model.predict(X_total_scaled)
    mse = mean_squared_error(y_total, y_pred)
    r2 = r2_score(y_total, y_pred)
    ard = np.mean(np.abs((y_total - y_pred) / y_total)) * 100

    # === 新增误差统计 ===
    relative_error = np.abs((y_pred - y_total) / y_total) * 100
    within_1pct = np.sum(relative_error <= 1)
    within_5pct = np.sum(relative_error <= 5)
    within_10pct = np.sum(relative_error <= 10)

    print("\n📊 总模型评估（含 slope×T 特征）：")
    print(f"R²  = {r2:.4f}")
    print(f"MSE = {mse:.2f}")
    print(f"ARD = {ard:.2f}%")
    print(f"✅ 误差 ≤ 1% 的数据点数量: {within_1pct}")
    print(f"✅ 误差 ≤ 5% 的数据点数量: {within_5pct}")
    print(f"✅ 误差 ≤ 10% 的数据点数量: {within_10pct}")

    # ========= 7. 输出预测结果 =========
    results = pd.DataFrame({
        "Material_ID": material_ids,
        "Temperature (K)": temperatures,
        "Cp_measured": y_total,
        "Cp_predicted": y_pred
    })
    results.to_excel("Cp预测结果_slopeT特征_相似度回归.xlsx", index=False)
    print("✅ 已保存预测结果为: Cp预测结果_slopeT特征_相似度回归.xlsx")

    # ========= 8. 输出系数表 =========
    feature_labels = (
            list(group_cols) +  # 19 个基团
            [f"{g}_T" for g in group_cols] +  # 19 个基团 × T
            ["slope×T"]  # 1 个新特征
    )

    coefficients = pd.DataFrame({
        "Feature": feature_labels,
        "Contribution": model.coef_
    })
    coefficients.to_excel("Cp系数表_slopeT特征_相似度回归.xlsx", index=False)
    print("📈 已保存模型系数为: Cp系数表_slopeT特征_相似度回归.xlsx")
else:
    print("❌ 没有有效的数据用于训练总模型")
