#
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ==== 1. 读取数据 ====
# df = pd.read_excel("internal energy 207.xlsx", sheet_name="Sheet1")
#
# # ==== 2. 定义列 ====
# group_cols = df.columns[13:32]   # 第14~25列：基团
# temp_cols = df.columns[32:42]    # 第26~35列：温度
# hvap_cols = df.columns[42:52]    # 第36~45列：Hvap
#
# # ==== 3. 准备 slope 所需模型输入 ====
# df_298 = pd.read_excel("selected_25_descriptors_normal.xlsx")
# X_298 = df_298.drop(columns=["internal energy at normal temperature"])
# rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["internal energy at normal temperature"])
# HVap_298_all = rf_298.predict(X_298)
#
# df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
# X_Tb = df_Tb.drop(columns=["internal energy at boiling temperature"])
# rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["internal energy at boiling temperature"])
# HVap_Tb_all = rf_Tb.predict(X_Tb)
#
# # ==== 4. Tb 模型预测 ====
# from sklearn.linear_model import HuberRegressor
# from sklearn.preprocessing import PolynomialFeatures
#
# Nk_all = df.iloc[:, 13:32].apply(pd.to_numeric, errors='coerce')
# Tb_raw = df.iloc[:, 5].values
# Tb0 = 222.543
# poly = PolynomialFeatures(degree=2, include_bias=False)
# Nk_poly = poly.fit_transform(Nk_all)
#
# mask_tb = ~np.isnan(Tb_raw)
# model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
# Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))
#
# # ==== 5. 计算 slope 并加入主 DataFrame ====
# T_ref = 298.15
# slope_values = (HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)
# df["slope"] = slope_values
#
# # ==== 6. 构建训练数据 ====
# X_total, y_total, material_ids, temperatures = [], [], [], []
#
# for i, row in df.iterrows():
#     material_id = row.iloc[0]
#     Nk = row[group_cols].values
#     temps = row[temp_cols].values
#     hvaps = row[hvap_cols].values
#     slope = row["slope"]
#
#     for T, Hv in zip(temps, hvaps):
#         if np.isnan(T) or np.isnan(Hv) or np.isnan(slope):
#             continue
#         features = np.concatenate([Nk, [T], [slope]])
#         X_total.append(features)
#         y_total.append(Hv)
#         material_ids.append(material_id)
#         temperatures.append(T)
#
# X_total = np.array(X_total)
# y_total = np.array(y_total)
#
# # ==== 7. 拟合模型 ====
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_total, y_total)
#
# # ==== 8. 模型评估 ====
# y_pred = model.predict(X_total)
# r2 = r2_score(y_total, y_pred)
# mse = mean_squared_error(y_total, y_pred)
# ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100  # ARD %
#
# print("\n📊 模型评估（基团 + 温度 + slope 特征）：")
# print(f"R²  = {r2:.4f}")
# print(f"MSE = {mse:.2f}")
# print(f"ARD = {ard:.2f}%")
#
# # ==== 9. 保存结果 ====
# results = pd.DataFrame({
#     "Material_ID": material_ids,
#     "Temperature (K)": temperatures,
#     "Hvap_measured": y_total,
#     "Hvap_predicted": y_pred,
#     "Absolute Error": np.abs(y_total - y_pred),
#     "Relative Error (%)": 100 * np.abs((y_total - y_pred) / y_total)
# })
# results.to_excel("Internal energy预测结果_加slope特征_RF.xlsx", index=False)
# print("✅ 已保存预测结果为: Internal energy预测结果_加slope特征_RF.xlsx")
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ==== 1. 读取数据 ====
# df = pd.read_excel("internal energy 207.xlsx", sheet_name="Sheet1")
#
# # ==== 2. 定义列 ====
# group_cols = df.columns[13:32]   # 第14~25列：基团
# temp_cols = df.columns[32:42]    # 第26~35列：温度
# hvap_cols = df.columns[42:52]    # 第36~45列：Hvap
#
# # ==== 3. 准备 slope 所需模型输入 ====
# df_298 = pd.read_excel("selected_25_descriptors_normal.xlsx")
# X_298 = df_298.drop(columns=["internal energy at normal temperature"])
# rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["internal energy at normal temperature"])
# HVap_298_all = rf_298.predict(X_298)
#
# df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
# X_Tb = df_Tb.drop(columns=["internal energy at boiling temperature"])
# rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["internal energy at boiling temperature"])
# HVap_Tb_all = rf_Tb.predict(X_Tb)
#
# # ==== 4. Tb 模型预测 ====
# from sklearn.linear_model import HuberRegressor
# from sklearn.preprocessing import PolynomialFeatures
#
# Nk_all = df.iloc[:, 13:32].apply(pd.to_numeric, errors='coerce')
# Tb_raw = df.iloc[:, 5].values
# Tb0 = 222.543
# poly = PolynomialFeatures(degree=2, include_bias=False)
# Nk_poly = poly.fit_transform(Nk_all)
#
# mask_tb = ~np.isnan(Tb_raw)
# model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
# Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))
#
# # ==== 5. 计算 slope 并加入主 DataFrame ====
# T_ref = 298.15
# slope_values = (HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)
# df["slope"] = slope_values
#
# # ==== 6. 构建训练数据 ====
# X_total, y_total, material_ids, temperatures = [], [], [], []
#
# for i, row in df.iterrows():
#     material_id = row.iloc[0]
#     Nk = row[group_cols].values
#     temps = row[temp_cols].values
#     hvaps = row[hvap_cols].values
#     slope = row["slope"]
#
#     for T, Hv in zip(temps, hvaps):
#         if np.isnan(T) or np.isnan(Hv) or np.isnan(slope):
#             continue
#         features = np.concatenate([Nk, [T], [slope]])
#         X_total.append(features)
#         y_total.append(Hv)
#         material_ids.append(material_id)
#         temperatures.append(T)
#
# X_total = np.array(X_total)
# y_total = np.array(y_total)
#
# # ==== 7. 拟合模型 ====
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_total, y_total)
#
# # ==== 8. 模型评估 ====
# y_pred = model.predict(X_total)
# r2 = r2_score(y_total, y_pred)
# mse = mean_squared_error(y_total, y_pred)
# ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100  # ARD %
#
# print("\n📊 模型评估（基团 + 温度 + slope 特征）：")
# print(f"R²  = {r2:.4f}")
# print(f"MSE = {mse:.2f}")
# print(f"ARD = {ard:.2f}%")
#
# # 计算相对误差
# relative_error = np.abs((y_total - y_pred) / y_total) * 100
#
# # 计算不同相对误差的数量
# error_1_percent = np.sum(relative_error < 1)
# error_5_percent = np.sum(relative_error < 5)
# error_10_percent = np.sum(relative_error < 10)
#
# print(f"\n📊 统计结果：")
# print(f"数据点相对误差小于1%: {error_1_percent}个")
# print(f"数据点相对误差小于5%: {error_5_percent}个")
# print(f"数据点相对误差小于10%: {error_10_percent}个")
#
# # ==== 9. 保存结果 ====
# results = pd.DataFrame({
#     "Material_ID": material_ids,
#     "Temperature (K)": temperatures,
#     "Hvap_measured": y_total,
#     "Hvap_predicted": y_pred,
#     "Absolute Error": np.abs(y_total - y_pred),
#     "Relative Error (%)": relative_error
# })
# results.to_excel("Internal_energy预测结果_加slope特征_RF.xlsx", index=False)
# print("✅ 已保存预测结果为: Internal_energy预测结果_加slope特征_RF.xlsx")

#
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ==== 1. 读取数据 ====
# df = pd.read_excel("internal energy 207.xlsx", sheet_name="Sheet6")
#
# # ==== 2. 定义列 ====
# group_cols = df.columns[13:32]  # 第14~32列：基团
# temp_cols = df.columns[32:42]  # 第33~42列：温度
# hvap_cols = df.columns[42:52]  # 第43~52列：目标变量 Hvap
#
# # ==== 3. 计算每个物质的目标 slope（中位数） ====
# slope_targets = []
# for i, row in df.iterrows():
#     temps = row[temp_cols].values
#     hvaps = row[hvap_cols].values
#
#     slopes = [(hvaps[t + 1] - hvaps[t]) / (temps[t + 1] - temps[t])
#               for t in range(len(temps) - 1) if not np.isnan(hvaps[t]) and not np.isnan(temps[t])]
#
#     slope_targets.append(np.median(slopes) if slopes else np.nan)
#
# df["slope_target"] = slope_targets
#
# # ==== 4. 用基团训练线性回归预测 slope ====
# X_slope = df[group_cols].values
# y_slope = df["slope_target"].values
#
# slope_model = LinearRegression()
# slope_model.fit(X_slope, y_slope)
#
# # 预测 slope
# slope_pred_all = slope_model.predict(X_slope)
#
# # ==== 5. 构建随机森林训练数据 ====
# X_total, y_total, material_ids, temperatures = [], [], [], []
#
# for i, row in df.iterrows():
#     material_id = row.iloc[0]
#     Nk = row[group_cols].values
#     temps = row[temp_cols].values
#     hvaps = row[hvap_cols].values
#     slope_pred = slope_pred_all[i]
#
#     for T, Hv in zip(temps, hvaps):
#         if np.isnan(T) or np.isnan(Hv) or np.isnan(slope_pred):
#             continue
#         features = np.concatenate([Nk, [T], [slope_pred]])
#         X_total.append(features)
#         y_total.append(Hv)
#         material_ids.append(material_id)
#         temperatures.append(T)
#
# X_total = np.array(X_total)
# y_total = np.array(y_total)
#
# # ==== 6. 拟合随机森林 ====
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_total, y_total)
#
# # ==== 7. 模型评估 ====
# y_pred = rf_model.predict(X_total)
# r2 = r2_score(y_total, y_pred)
# mse = mean_squared_error(y_total, y_pred)
# ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100
#
# print("\n📊 随机森林模型（基团 + 温度 + slope_pred 特征）评估：")
# print(f"R²  = {r2:.4f}")
# print(f"MSE = {mse:.2f}")
# print(f"ARD = {ard:.2f}%")
#
# # ==== 8. 保存结果 ====
# results = pd.DataFrame({
#     "Material_ID": material_ids,
#     "Temperature (K)": temperatures,
#     "Hvap_measured": y_total,
#     "Hvap_predicted": y_pred,
#     "Absolute Error": np.abs(y_total - y_pred),
#     "Relative Error (%)": np.abs((y_total - y_pred) / y_total) * 100
# })
# results.to_excel("Internal_energy_RF_with_slope_pred.xlsx", index=False)
# print("✅ 预测结果已保存为: Internal_energy_RF_with_slope_pred.xlsx")


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==== 1. 读取数据 ====
df = pd.read_excel("internal energy 207.xlsx", sheet_name="Sheet6")

# ==== 2. 定义列 ====
group_cols = df.columns[13:32]  # 第14~32列：基团
temp_cols = df.columns[32:42]   # 第33~42列：温度
hvap_cols = df.columns[42:52]   # 第43~52列：目标变量 Hvap

# ==== 3. 计算每个物质的目标 slope（首末点斜率） ====
slope_targets = []
for i, row in df.iterrows():
    temps = row[temp_cols].values
    hvaps = row[hvap_cols].values

    valid_idx = [j for j in range(len(temps)) if not np.isnan(temps[j]) and not np.isnan(hvaps[j])]
    if len(valid_idx) >= 2:
        first, last = valid_idx[0], valid_idx[-1]
        slope_target = (hvaps[last] - hvaps[first]) / (temps[last] - temps[first])
    else:
        slope_target = np.nan

    slope_targets.append(slope_target)

df["slope_target"] = slope_targets

# ==== 4. 用基团训练线性回归预测 slope ====
X_slope = df[group_cols].values
y_slope = df["slope_target"].values

slope_model = LinearRegression()
slope_model.fit(X_slope, y_slope)

# 预测 slope
slope_pred_all = slope_model.predict(X_slope)

# ==== 4a. 评估 slope 预测精度 ====
r2_slope = r2_score(y_slope, slope_pred_all)
mse_slope = mean_squared_error(y_slope, slope_pred_all)
ard_slope = np.mean(np.abs((slope_pred_all - y_slope) / y_slope)) * 100

print("\n📊 斜率预测线性回归模型评估：")
print(f"R²_slope  = {r2_slope:.4f}")
print(f"MSE_slope = {mse_slope:.4f}")
print(f"ARD_slope = {ard_slope:.2f}%")

# ==== 5. 构建随机森林训练数据 ====
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    hvaps = row[hvap_cols].values
    slope_pred = slope_pred_all[i]

    for T, Hv in zip(temps, hvaps):
        if np.isnan(T) or np.isnan(Hv) or np.isnan(slope_pred):
            continue
        features = np.concatenate([Nk, [T], [slope_pred]])
        X_total.append(features)
        y_total.append(Hv)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)

# ==== 6. 拟合随机森林 ====
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_total, y_total)

# ==== 7. 模型评估 ====
y_pred = rf_model.predict(X_total)
r2_rf = r2_score(y_total, y_pred)
mse_rf = mean_squared_error(y_total, y_pred)
ard_rf = np.mean(np.abs((y_pred - y_total) / y_total)) * 100

print("\n📊 随机森林模型（基团 + 温度 + slope_pred 特征）评估：")
print(f"R²_RF  = {r2_rf:.4f}")
print(f"MSE_RF = {mse_rf:.2f}")
print(f"ARD_RF = {ard_rf:.2f}%")

# ==== 8. 保存结果 ====
results = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Hvap_measured": y_total,
    "Hvap_predicted": y_pred,
    "Absolute Error": np.abs(y_total - y_pred),
    "Relative Error (%)": np.abs((y_total - y_pred) / y_total) * 100
})
results.to_excel("Internal_energy_RF_with_slope_pred_first_last.xlsx", index=False)
print("✅ 预测结果已保存为: Internal_energy_RF_with_slope_pred_first_last.xlsx")
