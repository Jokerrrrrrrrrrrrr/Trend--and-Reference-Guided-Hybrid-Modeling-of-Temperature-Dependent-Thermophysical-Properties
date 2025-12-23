# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
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
# temp_cols = df.columns[30:40]    # 10个温度点
# cp_cols = df.columns[40:50]      # 10个 Cp 值
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
# T1_model = HuberRegressor(max_iter=9000).fit(X_poly, y_exp_T1)
# Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 9])
# Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 50])
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
#     Nk_poly = X_poly_all[i:i+1]
#
#     try:
#         # 预测 T1、T2、Cp1、Cp2，并计算斜率
#         T1_exp = T1_model.predict(Nk_poly)[0]
#         if T1_exp <= 0 or np.isnan(T1_exp):
#             continue
#         T1 = Tc0 * np.log(T1_exp)
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
#
#         features = np.concatenate([
#             Nk,           # 12 个基团
#             Nk * T,       # 12 个交互项
#             [slope * T]   # slope × T
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
# print("\n📊 模型评估（含 slope×T 特征）：")
# print(f"R²  = {r2:.4f}")
# print(f"MSE = {mse:.2f}")
# print(f"ARD = {ard:.2f}%")
#
# # ========= 7. 输出预测结果 =========
# results = pd.DataFrame({
#     "Material_ID": material_ids,
#     "Temperature (K)": temperatures,
#     "Cp_measured": y_total,
#     "Cp_predicted": y_pred
# })
# results.to_excel("Cp预测结果_slopeT特征_β1回归.xlsx", index=False)
# print("✅ 已保存预测结果为: Cp预测结果_slopeT特征_β1回归.xlsx")
#
# # ========= 8. 输出系数表 =========
# feature_labels = (
#     list(group_cols) +               # 12 个基团
#     [f"{g}_T" for g in group_cols] + # 12 个基团 × T
#     ["slope×T"]                      # 1 个新特征
# )
#
# coefficients = pd.DataFrame({
#     "Feature": feature_labels,
#     "Contribution": model.coef_
# })
# coefficients.to_excel("Cp系数表_slopeT特征_β1回归.xlsx", index=False)
# print("📈 已保存模型系数为: Cp系数表_slopeT特征_β1回归.xlsx")
# #
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
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
# temp_cols = df.columns[30:40]    # 10个温度点
# cp_cols = df.columns[40:50]      # 10个 Cp 值
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
# # T1 模型评估
# y_pred_T1 = T1_model.predict(X_poly)
# r2_T1 = r2_score(y_exp_T1, y_pred_T1)
# mse_T1 = mean_squared_error(y_exp_T1, y_pred_T1)
#
# # Cp1 模型评估
# y_Cp1_true = df.iloc[:, 9]
# y_Cp1_pred = Cp1_model.predict(X_groups)
# r2_Cp1 = r2_score(y_Cp1_true, y_Cp1_pred)
# mse_Cp1 = mean_squared_error(y_Cp1_true, y_Cp1_pred)
#
# # Cp2 模型评估
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
# for i, row in df.iterrows():
#     material_id = row.iloc[0]
#     Nk = row[group_cols].values
#     temps = row[temp_cols].values
#     cps = row[cp_cols].values
#
#     Nk_df = pd.DataFrame([Nk], columns=group_cols)
#     Nk_poly = X_poly_all[i:i+1]
#
#     try:
#         # 预测 T1、T2、Cp1、Cp2，并计算斜率
#         T1_exp = T1_model.predict(Nk_poly)[0]
#         if T1_exp <= 0 or np.isnan(T1_exp):
#             continue
#         T1 = Tc0 * np.log(T1_exp)
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
#
#         features = np.concatenate([
#             Nk,           # 19 个基团
#             Nk * T,       # 19 个交互项
#             [slope * T]   # slope × T
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
# print("\n📊 总模型评估（含 slope×T 特征）：")
# print(f"R²  = {r2:.4f}")
# print(f"MSE = {mse:.2f}")
# print(f"ARD = {ard:.2f}%")
#
# # ========= 7. 输出预测结果 =========
# results = pd.DataFrame({
#     "Material_ID": material_ids,
#     "Temperature (K)": temperatures,
#     "Cp_measured": y_total,
#     "Cp_predicted": y_pred
# })
# results.to_excel("Cp预测结果_slopeT特征_β1回归.xlsx", index=False)
# print("✅ 已保存预测结果为: Cp预测结果_slopeT特征_β1回归.xlsx")
#
# # ========= 8. 输出系数表 =========
# feature_labels = (
#     list(group_cols) +               # 19 个基团
#     [f"{g}_T" for g in group_cols] + # 19 个基团 × T
#     ["slope×T"]                      # 1 个新特征
# )
#
# coefficients = pd.DataFrame({
#     "Feature": feature_labels,
#     "Contribution": model.coef_
# })
# coefficients.to_excel("Cp系数表_slopeT特征_β1回归.xlsx", index=False)
# print("📈 已保存模型系数为: Cp系数表_slopeT特征_β1回归.xlsx")
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
group_cols = df.columns[11:30]   # 19个基团列
temp_cols = df.columns[30:40]    # 10个温度点
cp_cols = df.columns[40:50]      # 10个 Cp 值
target_column_T1 = 'ASPEN Half Critical T'

# ========= 3. 子模型训练 =========
X_groups = df[group_cols]
valid_mask = ~df[target_column_T1].isna()

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_groups[valid_mask])

# 改用 GradientBoostingRegressor 预测 T1
y_T1 = df.loc[valid_mask, target_column_T1].values
T1_model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
).fit(X_poly, y_T1)

# Cp1, Cp2 使用 HuberRegressor
Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 9])
Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 50])

# ========= 3.1 子模型评估 =========
y_pred_T1 = T1_model.predict(X_poly)
r2_T1 = r2_score(y_T1, y_pred_T1)
mse_T1 = mean_squared_error(y_T1, y_pred_T1)

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

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    cps = row[cp_cols].values

    Nk_df = pd.DataFrame([Nk], columns=group_cols)
    Nk_poly = X_poly_all[i:i+1]

    try:
        # 新模型：直接预测 T1（无需 log 和 exp）
        T1 = T1_model.predict(Nk_poly)[0]
        if T1 <= 0 or np.isnan(T1):
            continue
        T2 = T1 * 1.5
        Cp1 = Cp1_model.predict(Nk_df)[0]
        Cp2 = Cp2_model.predict(Nk_df)[0]
        slope = (Cp2 - Cp1) / (T2 - T1)
        # slope = (y_Cp2_true-y_Cp1_true)/(1.5*target_column_T1-target_column_T1)
    except:
        continue

    for T, Cp in zip(temps, cps):
        if np.isnan(T) or np.isnan(Cp):
            continue

        features = np.concatenate([
            Nk,           # 19 个基团
            Nk * T,       # 19 个交互项
            [slope * T]   # slope × T
        ])

        X_total.append(features)
        y_total.append(Cp)
        material_ids.append(material_id)
        temperatures.append(T)

# ========= 5. 模型拟合（Huber） =========
X_total = np.array(X_total)
y_total = np.array(y_total)

model = HuberRegressor(max_iter=10000).fit(X_total, y_total)

# ========= 6. 模型评估 =========
y_pred = model.predict(X_total)
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
results.to_excel("Cp预测结果_slopeT特征_β1回归.xlsx", index=False)
print("✅ 已保存预测结果为: Cp预测结果_slopeT特征_β1回归.xlsx")

# ========= 8. 输出系数表 =========
feature_labels = (
    list(group_cols) +               # 19 个基团
    [f"{g}_T" for g in group_cols] + # 19 个基团 × T
    ["slope×T"]                      # 1 个新特征
)

coefficients = pd.DataFrame({
    "Feature": feature_labels,
    "Contribution": model.coef_
})
coefficients.to_excel("Cp系数表_slopeT特征_β1回归.xlsx", index=False)
print("📈 已保存模型系数为: Cp系数表_slopeT特征_β1回归.xlsx")
