# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
# from sklearn.preprocessing import PolynomialFeatures
#
# # 读取 Gani 数据
# gani_df = pd.read_excel("heat capacity 207.xlsx", sheet_name="Sheet1")
# gani_df = gani_df.dropna(subset=[gani_df.columns[0]])
# gani_df[gani_df.columns[0]] = gani_df[gani_df.columns[0]].astype(int)
#
# # 定义列
# group_cols = gani_df.columns[11:30]  # 19个基团列
# target_column_T1 = 'ASPEN Half Critical T'
# Tc0 = 138
#
# # 子模型训练
# X_groups = gani_df[group_cols]
# valid_mask = ~gani_df[target_column_T1].isna()
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X_groups[valid_mask])
# y_exp_T1 = np.exp(gani_df.loc[valid_mask, target_column_T1] / Tc0)
#
# # 模型拟合
# T1_model = HuberRegressor(max_iter=9000).fit(X_poly, y_exp_T1)
# Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, gani_df.iloc[:, 9])
# Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, gani_df.iloc[:, 50])
#
# # 计算 slope
# X_poly_all = poly.transform(X_groups)
# slope_dict = {}
#
# for i, row in gani_df.iterrows():
#     material_id = row.iloc[0]
#     Nk = row[group_cols].values
#     Nk_df = pd.DataFrame([Nk], columns=group_cols)
#     Nk_poly = X_poly_all[i:i+1]
#
#     try:
#         T1_exp = T1_model.predict(Nk_poly)[0]
#         if T1_exp <= 0 or np.isnan(T1_exp):
#             continue
#         T1 = Tc0 * np.log(T1_exp)
#         T2 = T1 * 1.5
#         Cp1 = Cp1_model.predict(Nk_df)[0]
#         Cp2 = Cp2_model.predict(Nk_df)[0]
#         slope = (Cp2 - Cp1) / (T2 - T1)
#         slope_dict[material_id] = slope
#     except:
#         continue
#
# # 保存为 DataFrame
# slope_df = pd.DataFrame(list(slope_dict.items()), columns=["Material_ID", "slope"])
# slope_df.to_csv("slope_values.csv", index=False)
# print("✅ slope 已保存为 slope_values.csv")
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures

# 读取 Gani 数据
gani_df = pd.read_excel("heat capacity 207.xlsx", sheet_name="Sheet1")
gani_df = gani_df.dropna(subset=[gani_df.columns[0]])
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
Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, gani_df.iloc[:, 9])
Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, gani_df.iloc[:, 50])

# 计算 slope
X_poly_all = poly.transform(X_groups)
slope_dict = {}

for i, row in gani_df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    Nk_df = pd.DataFrame([Nk], columns=group_cols)
    Nk_poly = X_poly_all[i:i+1]

    try:
        T1 = T1_model.predict(Nk_poly)[0]
        if T1 <= 0 or np.isnan(T1):
            continue
        T2 = T1 * 1.5
        Cp1 = Cp1_model.predict(Nk_df)[0]
        Cp2 = Cp2_model.predict(Nk_df)[0]
        slope = (Cp2 - Cp1) / (T2 - T1)
        slope_dict[material_id] = slope
    except:
        continue

# 保存为 DataFrame
slope_df = pd.DataFrame(list(slope_dict.items()), columns=["Material_ID", "slope"])
slope_df.to_csv("slope_values.csv", index=False)
print("✅ slope 已保存为 slope_values.csv")
