# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import PolynomialFeatures
#
# # 读取 Gani 数据
# gani_df = pd.read_excel("heat capacity 207.xlsx", sheet_name="Sheet1")
# gani_df = gani_df.dropna(subset=[gani_df.columns[0]])  # 删除第一列为空的行
# gani_df[gani_df.columns[0]] = gani_df[gani_df.columns[0]].astype(int)
#
# # 定义列
# group_cols = gani_df.columns[11:30]  # 19个基团列
# target_column_T1 = 'ASPEN Half Critical T'
#
# # 子模型训练
# X_groups = gani_df[group_cols]
# valid_mask = ~gani_df[target_column_T1].isna()
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X_groups[valid_mask])
# y_T1 = gani_df.loc[valid_mask, target_column_T1]
#
# # ✅ 使用 GradientBoostingRegressor 拟合 T1
# T1_model = GradientBoostingRegressor(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=4,
#     random_state=0
# ).fit(X_poly, y_T1)
#
# # Cp1、Cp2 模型保持不变
# Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, gani_df.iloc[:, 9])  # 预测 Cp1
# Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, gani_df.iloc[:, 50])  # 预测 Cp2
#
# # 计算残差
# X_poly_all = poly.transform(X_groups)
# cp1_residuals = []
# cp2_residuals = []
#
# for i, row in gani_df.iterrows():
#     material_id = row.iloc[0]  # 获取物质 ID
#     Nk = row[group_cols].values  # 获取基团值
#     Nk_df = pd.DataFrame([Nk], columns=group_cols)
#     Nk_poly = X_poly_all[i:i + 1]
#
#     try:
#         # 预测 T1
#         T1 = T1_model.predict(Nk_poly)[0]
#
#         # 跳过无效的 T1
#         if T1 <= 0 or np.isnan(T1):
#             continue
#
#         # 预测 Cp1 和 Cp2
#         Cp1_pred = Cp1_model.predict(Nk_df)[0]
#         Cp2_pred = Cp2_model.predict(Nk_df)[0]
#
#         # 获取实际的 Cp1 和 Cp2 值（来自 gani_df）
#         Cp1_actual = gani_df.iloc[i, 9]  # Cp1 的实际值
#         Cp2_actual = gani_df.iloc[i, 50]  # Cp2 的实际值
#
#         # 计算残差（实际值 - 预测值）
#         Cp1_residual = Cp1_actual - Cp1_pred
#         Cp2_residual = Cp2_actual - Cp2_pred
#
#         # 存储残差
#         cp1_residuals.append(Cp1_residual)
#         cp2_residuals.append(Cp2_residual)
#
#     except Exception as e:
#         # 捕获异常并继续
#         print(f"Error processing material {material_id}: {e}")
#         continue
#
# # 保存为 DataFrame
# residual_df = pd.DataFrame({
#     "Cp1_residual": cp1_residuals,
#     "Cp2_residual": cp2_residuals
# })
#
# # 保存残差结果为 CSV 文件
# residual_df.to_csv("cp_residuals.csv", index=False)
# print("✅ Cp1 和 Cp2 残差已保存为 cp_residuals.csv")


import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures

# 读取 Gani 数据
gani_df = pd.read_excel("heat capacity 207.xlsx", sheet_name="Sheet1")
gani_df = gani_df.dropna(subset=[gani_df.columns[0]])  # 删除第一列为空的行
gani_df[gani_df.columns[0]] = gani_df[gani_df.columns[0]].astype(int)

# 定义列
group_cols = gani_df.columns[11:30]  # 19个基团列
target_column_T1 = 'ASPEN Half Critical T'

# 子模型训练
X_groups = gani_df[group_cols]
valid_mask = ~gani_df[target_column_T1].isna()
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_groups[valid_mask])
y_T1 = gani_df.loc[valid_mask, target_column_T1]

# ✅ 使用 GradientBoostingRegressor 拟合 T1
T1_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=0
).fit(X_poly, y_T1)

# Cp1、Cp2 模型保持不变
Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, gani_df.iloc[:, 9])  # 预测 Cp1
Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, gani_df.iloc[:, 50])  # 预测 Cp2

# 计算残差
X_poly_all = poly.transform(X_groups)
cp1_residuals = []
cp2_residuals = []

for i, row in gani_df.iterrows():
    material_id = row.iloc[0]  # 获取物质 ID
    Nk = row[group_cols].values  # 获取基团值
    Nk_df = pd.DataFrame([Nk], columns=group_cols)
    Nk_poly = X_poly_all[i:i + 1]

    try:
        # 预测 T1
        T1 = T1_model.predict(Nk_poly)[0]

        # 跳过无效的 T1
        if T1 <= 0 or np.isnan(T1):
            continue

        # 预测 Cp1 和 Cp2
        Cp1_pred = Cp1_model.predict(Nk_df)[0]
        Cp2_pred = Cp2_model.predict(Nk_df)[0]

        # 获取实际的 Cp1 和 Cp2 值（来自 gani_df）
        Cp1_actual = gani_df.iloc[i, 9]  # Cp1 的实际值
        Cp2_actual = gani_df.iloc[i, 50]  # Cp2 的实际值

        # 计算残差（实际值 - 预测值）
        Cp1_residual = Cp1_actual - Cp1_pred
        Cp2_residual = Cp2_actual - Cp2_pred

        # 将残差扩展到 10 行，确保每个温度点都有对应的残差
        cp1_residuals.extend([Cp1_residual] * 10)  # 每个物质的残差值复制 10 次
        cp2_residuals.extend([Cp2_residual] * 10)  # 每个物质的残差值复制 10 次

    except Exception as e:
        # 捕获异常并继续
        print(f"Error processing material {material_id}: {e}")
        continue

# 创建物质 ID 列
material_ids = np.tile(gani_df.iloc[:, 0].values, 10)  # 将物质 ID 扩展到 10 行

# 保存为 DataFrame
residual_df = pd.DataFrame({
    "Material_ID": material_ids,
    "Cp1_residual": cp1_residuals,
    "Cp2_residual": cp2_residuals
})

# 保存残差结果为 CSV 文件
residual_df.to_csv("cp_residuals_expanded.csv", index=False)
print("✅ Cp1 和 Cp2 残差（扩展到 10 行）已保存为 cp_residuals_expanded.csv")
