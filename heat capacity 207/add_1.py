# # import pandas as pd
# # import numpy as np
# # from sklearn.linear_model import HuberRegressor
# # from sklearn.preprocessing import PolynomialFeatures
# # from sklearn.metrics import mean_squared_error, r2_score
# #
# # # ========= 1. 读取数据 =========
# # file_path = "heat capacity 207.xlsx"
# # df = pd.read_excel(file_path, sheet_name="Sheet1")
# # df = df.dropna(subset=[df.columns[0]])  # 删除第一列为空的行
# # df[df.columns[0]] = df[df.columns[0]].astype(int)  # 将第一列转换为整数类型
# #
# # # ========= 2. 列定义 =========
# # group_cols = df.columns[11:30]   # 19个基团列
# # temp_cols = df.columns[30:40]    # 10个温度点
# # cp_cols = df.columns[40:50]      # 10个 Cp 值
# # target_column_T1 = 'ASPEN Half Critical T'
# # Tc0 = 138  # 临界温度归一化常数
# #
# # # ========= 3. 子模型训练 =========
# # X_groups = df[group_cols]
# # valid_mask = ~df[target_column_T1].isna()
# #
# # poly = PolynomialFeatures(degree=2, include_bias=False)
# # X_poly = poly.fit_transform(X_groups[valid_mask])
# # y_exp_T1 = np.exp(df.loc[valid_mask, target_column_T1] / Tc0)
# #
# # # 模型拟合
# # T1_model = HuberRegressor(max_iter=9000).fit(X_poly, y_exp_T1)
# # Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 9])
# # Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 50])
# #
# # # ========= 3.1 子模型评估 =========
# # y_pred_T1 = T1_model.predict(X_poly)
# # r2_T1 = r2_score(y_exp_T1, y_pred_T1)
# # mse_T1 = mean_squared_error(y_exp_T1, y_pred_T1)
# #
# # y_Cp1_true = df.iloc[:, 9]
# # y_Cp1_pred = Cp1_model.predict(X_groups)
# # r2_Cp1 = r2_score(y_Cp1_true, y_Cp1_pred)
# # mse_Cp1 = mean_squared_error(y_Cp1_true, y_Cp1_pred)
# #
# # y_Cp2_true = df.iloc[:, 50]
# # y_Cp2_pred = Cp2_model.predict(X_groups)
# # r2_Cp2 = r2_score(y_Cp2_true, y_Cp2_pred)
# # mse_Cp2 = mean_squared_error(y_Cp2_true, y_Cp2_pred)
# #
# # print("\n📌 子模型评估结果：")
# # print(f"T1_model ->     R²: {r2_T1:.4f} | MSE: {mse_T1:.4f}")
# # print(f"Cp1_model ->    R²: {r2_Cp1:.4f} | MSE: {mse_Cp1:.4f}")
# # print(f"Cp2_model ->    R²: {r2_Cp2:.4f} | MSE: {mse_Cp2:.4f}")
# #
# # # ========= 4. 构建训练数据 =========
# # X_total, y_total, material_ids, temperatures = [], [], [], []
# # X_poly_all = poly.transform(X_groups)
# #
# # # 假设 extra_point_indices 是额外点的索引，例如[0, 1, 5, 7, 9]
# # extra_point_indices = [0, 1, 5, 7, 9]  # 这只是示例，根据实际数据设置
# #
# # # 创建权重数组，默认每个点的权重为 1
# # weights = np.ones(len(y_total))  # 默认所有样本的权重为 1
# # weights[extra_point_indices] = 10  # 对额外点赋予较高的权重，例如权重为 10
# #
# # for i, row in df.iterrows():
# #     material_id = row.iloc[0]
# #     Nk = row[group_cols].values
# #     temps = row[temp_cols].values
# #     cps = row[cp_cols].values
# #
# #     Nk_df = pd.DataFrame([Nk], columns=group_cols)
# #     Nk_poly = X_poly_all[i:i+1]
# #
# #     try:
# #         # 新模型：直接预测 T1（无需 log 和 exp）
# #         T1_exp = T1_model.predict(Nk_poly)[0]
# #         if T1_exp <= 0 or np.isnan(T1_exp):
# #             continue
# #         T1 = Tc0 * np.log(T1_exp)
# #         T2 = T1 * 1.5
# #         Cp1 = Cp1_model.predict(Nk_df)[0]
# #         Cp2 = Cp2_model.predict(Nk_df)[0]
# #         slope = (Cp2 - Cp1) / (T2 - T1)
# #     except:
# #         continue
# #
# #     for T, Cp in zip(temps, cps):
# #         if np.isnan(T) or np.isnan(Cp):
# #             continue
# #
# #         features = np.concatenate([
# #             Nk,           # 19 个基团
# #             Nk * T,       # 19 个交互项
# #             [slope * T]   # slope × T
# #         ])
# #
# #         X_total.append(features)
# #         y_total.append(Cp)
# #         material_ids.append(material_id)
# #         temperatures.append(T)
# #
# # # ========= 5. 模型拟合（Huber） =========
# # X_total = np.array(X_total)
# # y_total = np.array(y_total)
# #
# # # 使用加权的损失函数进行训练
# # model = HuberRegressor(max_iter=10000).fit(X_total, y_total, sample_weight=weights)
# #
# # # ========= 6. 模型评估 =========
# # y_pred = model.predict(X_total)
# # mse = mean_squared_error(y_total, y_pred)
# # r2 = r2_score(y_total, y_pred)
# # ard = np.mean(np.abs((y_total - y_pred) / y_total)) * 100
# #
# # print("\n📊 总模型评估（含 slope×T 特征）：")
# # print(f"R²  = {r2:.4f}")
# # print(f"MSE = {mse:.2f}")
# # print(f"ARD = {ard:.2f}%")
# #
# # # ========= 7. 输出预测结果 =========
# # results = pd.DataFrame({
# #     "Material_ID": material_ids,
# #     "Temperature (K)": temperatures,
# #     "Cp_measured": y_total,
# #     "Cp_predicted": y_pred
# # })
# # results.to_excel("Cp预测结果_slopeT特征_β1回归加权.xlsx", index=False)
# # print("✅ 已保存预测结果为: Cp预测结果_slopeT特征_β1回归加权.xlsx")
# #
# # # ========= 8. 输出系数表 =========
# # feature_labels = (
# #     list(group_cols) +               # 19 个基团
# #     [f"{g}_T" for g in group_cols] + # 19 个基团 × T
# #     ["slope×T"]                      # 1 个新特征
# # )
# #
# # coefficients = pd.DataFrame({
# #     "Feature": feature_labels,
# #     "Contribution": model.coef_
# # })
# # coefficients.to_excel("Cp系数表_slopeT特征_β1回归加权.xlsx", index=False)
# # print("📈 已保存模型系数为: Cp系数表_slopeT特征_β1回归加权.xlsx")
#




import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# ========= 1. 读取数据 =========
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 检查数据加载是否成功
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns}")

# 删除第一列为空的行
df = df.dropna(subset=[df.columns[0]])
df[df.columns[0]] = df[df.columns[0]].astype(int)  # 将第一列转换为整数类型

# 确认数据清洗后是否正确
print(f"Data shape after cleaning: {df.shape}")

# ========= 2. 列定义 =========
group_cols = df.columns[11:30]  # 19个基团列
temp_cols = df.columns[30:40]  # 10个温度点
cp_cols = df.columns[40:50]  # 10个 Cp 值
target_column_T1 = 'ASPEN Half Critical T'
Tc0 = 138  # 临界温度归一化常数

# ========= 3. 子模型训练 =========
X_groups = df[group_cols]
valid_mask = ~df[target_column_T1].isna()

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_groups[valid_mask])
y_exp_T1 = np.exp(df.loc[valid_mask, target_column_T1] / Tc0)

# 模型拟合
T1_model = HuberRegressor(max_iter=9000).fit(X_poly, y_exp_T1)
Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 9])
Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 50])

# ========= 3.1 子模型评估 =========
y_pred_T1 = T1_model.predict(X_poly)
r2_T1 = r2_score(y_exp_T1, y_pred_T1)
mse_T1 = mean_squared_error(y_exp_T1, y_pred_T1)

y_Cp1_true = df.iloc[:, 9]
y_Cp1_pred = Cp1_model.predict(X_groups)
r2_Cp1 = r2_score(y_Cp1_true, y_Cp1_pred)
mse_Cp1 = mean_squared_error(y_Cp1_true, y_Cp1_pred)

y_Cp2_true = df.iloc[:, 50]
y_Cp2_pred = Cp2_model.predict(X_groups)
r2_Cp2 = r2_score(y_Cp2_true, y_Cp2_pred)
mse_Cp2 = mean_squared_error(y_Cp2_true, y_Cp2_pred)

print("\n📌 子模型评估结果：")
print(f"T1_model ->     R²: {r2_T1:.4f} | MSE: {mse_T1:.4f}")
print(f"Cp1_model ->    R²: {r2_Cp1:.4f} | MSE: {mse_Cp1:.4f}")
print(f"Cp2_model ->    R²: {r2_Cp2:.4f} | MSE: {mse_Cp2:.4f}")

# ========= 4. 构建训练数据 =========
X_total, y_total, material_ids, temperatures = [], [], [], []
X_poly_all = poly.transform(X_groups)

# 用于存储额外的预测点（T1、Cp1、T2、Cp2）
extra_point_weights = []  # 用于存储每个样本的额外权重

# 检查 X_total 和 y_total 是否为空
print(f"Before filling: X_total size = {len(X_total)}, y_total size = {len(y_total)}")

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    cps = row[cp_cols].values

    Nk_df = pd.DataFrame([Nk], columns=group_cols)
    Nk_poly = X_poly_all[i:i + 1]

    try:
        # 新模型：直接预测 T1（无需 log 和 exp）
        T1_exp = T1_model.predict(Nk_poly)[0]
        if T1_exp <= 0 or np.isnan(T1_exp):
            continue
        T1 = Tc0 * np.log(T1_exp)
        T2 = T1 * 1.5
        Cp1 = Cp1_model.predict(Nk_df)[0]
        Cp2 = Cp2_model.predict(Nk_df)[0]
        slope = (Cp2 - Cp1) / (T2 - T1)

        # 为预测的 T1、Cp1、T2、Cp2 生成权重（用于加权损失）
        extra_point_weights.append(2.2)  # 可以调整权重值（如 10）以强调这些预测点
    except:
        continue

    for T, Cp in zip(temps, cps):
        if np.isnan(T) or np.isnan(Cp):
            continue

        features = np.concatenate([
            Nk,  # 19 个基团
            Nk * T,  # 19 个交互项
            [slope * T]  # slope × T
        ])

        X_total.append(features)
        y_total.append(Cp)
        material_ids.append(material_id)
        temperatures.append(T)

# 检查 X_total 和 y_total 填充后的大小
print(f"After filling: X_total size = {len(X_total)}, y_total size = {len(y_total)}")

# 确保 X_total 和 y_total 不为空，继续训练
if len(X_total) > 0 and len(y_total) > 0:
    # 重新定义权重
    weights = np.ones(len(y_total))  # 默认所有样本的权重为 1

    # 给额外的预测点（T1、Cp1、T2、Cp2）增加更高的权重
    weights[-len(extra_point_weights):] = extra_point_weights  # 将预测点的权重设置为 10

    # 使用加权的损失函数进行训练
    model = HuberRegressor(max_iter=20000).fit(X_total, y_total, sample_weight=weights)

    # ========= 6. 模型评估 =========
    y_pred = model.predict(X_total)
    mse = mean_squared_error(y_total, y_pred)
    r2 = r2_score(y_total, y_pred)
    ard = np.mean(np.abs((y_total - y_pred) / y_total)) * 100

    print("\n📊 总模型评估（含 slope×T 特征）：")
    print(f"R²  = {r2:.4f}")
    print(f"MSE = {mse:.2f}")
    print(f"ARD = {ard:.2f}%")

    # ========= 7. 输出预测结果 =========
    results = pd.DataFrame({
        "Material_ID": material_ids,
        "Temperature (K)": temperatures,
        "Cp_measured": y_total,
        "Cp_predicted": y_pred
    })
    results.to_excel("Cp预测结果_slopeT特征_β1回归加权.xlsx", index=False)
    print("✅ 已保存预测结果为: Cp预测结果_slopeT特征_β1回归加权.xlsx")

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
    coefficients.to_excel("Cp系数表_slopeT特征_β1回归加权.xlsx", index=False)
    print("📈 已保存模型系数为: Cp系数表_slopeT特征_β1回归加权.xlsx")
else:
    print("错误：训练数据为空！")
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
#
# # ========= 1. 读取数据 =========
# file_path = "heat capacity 207.xlsx"
# df = pd.read_excel(file_path, sheet_name="Sheet1")
#
# # 检查数据加载是否成功
# print(f"Data shape: {df.shape}")
# print(f"Columns: {df.columns}")
#
# # 删除第一列为空的行
# df = df.dropna(subset=[df.columns[0]])
# df[df.columns[0]] = df[df.columns[0]].astype(int)  # 将第一列转换为整数类型
#
# # 确认数据清洗后是否正确
# print(f"Data shape after cleaning: {df.shape}")
#
# # ========= 2. 列定义 =========
# group_cols = df.columns[11:30]  # 19个基团列
# temp_cols = df.columns[30:40]  # 10个温度点
# cp_cols = df.columns[40:50]  # 10个 Cp 值
# target_column_T1 = 'ASPEN Half Critical T'
# Tc0 = 138  # 临界温度归一化常数
#
# # ========= 3. 子模型训练 =========
# X_groups = df[group_cols]
# valid_mask = ~df[target_column_T1].isna()
#
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X_groups[valid_mask])
# y_exp_T1 = np.exp(df.loc[valid_mask, target_column_T1] / Tc0)
#
# # 模型拟合
# T1_model = HuberRegressor(max_iter=9000).fit(X_poly, y_exp_T1)
# Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 9])
# Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 50])
#
# # ========= 3.1 子模型评估 =========
# y_pred_T1 = T1_model.predict(X_poly)
# r2_T1 = r2_score(y_exp_T1, y_pred_T1)
# mse_T1 = mean_squared_error(y_exp_T1, y_pred_T1)
#
# y_Cp1_true = df.iloc[:, 9]
# y_Cp1_pred = Cp1_model.predict(X_groups)
# r2_Cp1 = r2_score(y_Cp1_true, y_Cp1_pred)
# mse_Cp1 = mean_squared_error(y_Cp1_true, y_Cp1_pred)
#
# y_Cp2_true = df.iloc[:, 50]
# y_Cp2_pred = Cp2_model.predict(X_groups)
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
# # 存储额外点的索引（T1、Cp1、T2、Cp2对应的点）
# extra_point_indices = []
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
#         T1_exp = T1_model.predict(Nk_poly)[0]
#         if T1_exp <= 0 or np.isnan(T1_exp):
#             continue
#         T1 = Tc0 * np.log(T1_exp)
#         T2 = T1 * 1.5
#         Cp1 = Cp1_model.predict(Nk_df)[0]
#         Cp2 = Cp2_model.predict(Nk_df)[0]
#         slope = (Cp2 - Cp1) / (T2 - T1)
#
#         # 将 T1, Cp1, T2, Cp2 对应的点作为额外点
#         extra_point_indices.append(len(X_total))  # 将当前索引作为额外点的索引
#
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
# # 转换为numpy数组
# X_total = np.array(X_total)
# y_total = np.array(y_total)
#
# print(f"After filling: X_total size = {len(X_total)}, y_total size = {len(y_total)}")
# print(f"Extra point indices: {len(extra_point_indices)}")
#
#
# # ========= 5. 权重自动优化 =========
# def evaluate_model_with_weight(weight_value, X_train, y_train, extra_indices):
#     """使用给定权重评估模型"""
#     weights = np.ones(len(y_train))
#     if len(extra_indices) > 0:
#         # 只调整训练集中存在的额外点索引
#         train_extra_indices = [i for i in extra_indices if i < len(weights)]
#         weights[train_extra_indices] = weight_value
#
#     model = HuberRegressor(max_iter=100000000000).fit(X_train, y_train, sample_weight=weights)
#     y_pred = model.predict(X_train)
#     mse = mean_squared_error(y_train, y_pred)
#     r2 = r2_score(y_train, y_pred)
#     return mse, r2, model
#
#
# # 测试不同的权重值（0-20，间隔0.2）
# weight_values = np.arange(0, 20, 0.2)
# best_mse = np.inf
# best_r2 = -np.inf
# best_weight = 1.0
# best_model = None
# results = []
#
# print("\n🔍 开始权重自动优化（MSE最小化）...")
# print("权重值\tMSE\t\tR²")
#
# for weight in weight_values:
#     mse, r2, model = evaluate_model_with_weight(weight, X_total, y_total, extra_point_indices)
#     results.append((weight, mse, r2))
#
#     if weight % 2 == 0:  # 每2.0打印一次进度
#         print(f"{weight:.1f}\t{mse:.6f}\t{r2:.4f}")
#
#     if mse < best_mse:
#         best_mse = mse
#         best_r2 = r2
#         best_weight = weight
#         best_model = model
#
# print(f"\n🎯 最佳权重: {best_weight:.1f}")
# print(f"最佳 MSE: {best_mse:.6f}")
# print(f"最佳 R²: {best_r2:.4f}")
#
# # ========= 6. 使用最佳权重训练最终模型 =========
# print(f"\n🚀 使用最佳权重 {best_weight:.1f} 训练最终模型...")
#
# # 使用最佳权重训练最终模型，使用所有数据
# final_weights = np.ones(len(y_total))
# if len(extra_point_indices) > 0:
#     final_weights[extra_point_indices] = best_weight
#
# final_model = HuberRegressor(max_iter=1000000000000).fit(X_total, y_total, sample_weight=final_weights)
#
# # ========= 7. 最终模型评估 =========
# y_pred = final_model.predict(X_total)
# mse = mean_squared_error(y_total, y_pred)
# r2 = r2_score(y_total, y_pred)
# ard = np.mean(np.abs((y_total - y_pred) / np.clip(np.abs(y_total), 1e-10, None))) * 100
#
# print("\n📊 最终模型评估结果：")
# print(f"最佳权重: {best_weight:.1f}")
# print(f"R²  = {r2:.4f}")
# print(f"MSE = {mse:.4f}")
# print(f"ARD = {ard:.2f}%")
#
# # ========= 8. 绘制权重优化结果 =========
# weights, mses, r2s = zip(*results)
#
# plt.figure(figsize=(12, 5))
#
# plt.subplot(1, 2, 1)
# plt.plot(weights, mses, 'ro-', linewidth=2, markersize=4)
# plt.axvline(x=best_weight, color='blue', linestyle='--', label=f'最佳权重: {best_weight:.1f}')
# plt.xlabel('权重值')
# plt.ylabel('MSE')
# plt.title('权重优化 - MSE vs 权重值')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.subplot(1, 2, 2)
# plt.plot(weights, r2s, 'go-', linewidth=2, markersize=4)
# plt.axvline(x=best_weight, color='blue', linestyle='--', label=f'最佳权重: {best_weight:.1f}')
# plt.xlabel('权重值')
# plt.ylabel('R²')
# plt.title('权重优化 - R² vs 权重值')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('权重优化结果_MSE最小化.png', dpi=300, bbox_inches='tight')
# plt.show()
