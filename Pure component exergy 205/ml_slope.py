#
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ==== 1. 读取数据 ====
# df = pd.read_excel("volume208.xlsx", sheet_name="Sheet1")
#
# # ==== 2. 定义列 ====
# group_cols = df.columns[13:32]   # 第14~25列：基团
# temp_cols = df.columns[32:42]    # 第26~35列：温度
# hvap_cols = df.columns[42:52]    # 第36~45列：Hvap
#
# # ==== 3. 准备 slope 所需模型输入 ====
# df_298 = pd.read_excel("selected_25_descriptors_normal.xlsx")
# X_298 = df_298.drop(columns=["volume at normal temperature"])
# rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["volume at normal temperature"])
# HVap_298_all = rf_298.predict(X_298)
#
# df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
# X_Tb = df_Tb.drop(columns=["volume at boiling temperature"])
# rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["volume at boiling temperature"])
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
# results.to_excel("Vol预测结果_加slope特征_RF.xlsx", index=False)
# print("✅ 已保存预测结果为: Vol预测结果_加slope特征_RF.xlsx")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==== 1. 读取数据 ====
df = pd.read_excel("Pure component exergy 205.xlsx", sheet_name="Sheet1")

# ==== 2. 定义列 ====
group_cols = df.columns[12:31]   # 第14~25列：基团
temp_cols = df.columns[31:41]    # 第26~35列：温度
v_cols = df.columns[41:51]       # 第36~45列：Hvap

# ==== 3. 准备 slope 所需模型输入 ====
df_298 = pd.read_excel("selected_25_descriptors_normal.xlsx")
X_298 = df_298.drop(columns=["ASPEN Exergy at 500k Temperature(j/mol)"])
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["ASPEN Exergy at 500k Temperature(j/mol)"])
HVap_298_all = rf_298.predict(X_298)

df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
X_Tb = df_Tb.drop(columns=["ASPEN Exergy at BoilingTemperature(j/mol)"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["ASPEN Exergy at BoilingTemperature(j/mol)"])
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 4. Tb 模型预测 ====
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

Nk_all = df.iloc[:, 12:31].apply(pd.to_numeric, errors='coerce')
Tb_raw = df.iloc[:, 5].values
Tb0 = 222.543
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)

mask_tb = ~np.isnan(Tb_raw)
model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))

# ==== 5. 计算 slope 并加入主 DataFrame ====
T_ref = 500
slope_values = (HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)
df["slope"] = slope_values

# ==== 6. 构建训练数据 ====
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    vols = row[v_cols].values
    slope = row["slope"]

    for T, vol in zip(temps, vols):
        if np.isnan(T) or np.isnan(vol) or np.isnan(slope):
            continue
        features = np.concatenate([Nk, [T], [slope]])
        X_total.append(features)
        y_total.append(vol)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)

# ==== 7. 拟合模型 ====
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_total, y_total)

# ==== 8. 模型评估 ====
y_pred = model.predict(X_total)
r2 = r2_score(y_total, y_pred)
mse = mean_squared_error(y_total, y_pred)
ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100  # ARD %

print("\n📊 模型评估（基团 + 温度 + slope 特征）：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")

# 计算相对误差
relative_error = np.abs((y_pred - y_total) / y_total) * 100

# 统计不同误差阈值内的点数
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print(f"相对误差 ≤ 1% 的点数: {within_1pct}")
print(f"相对误差 ≤ 5% 的点数: {within_5pct}")
print(f"相对误差 ≤ 10% 的点数: {within_10pct}")

# ==== 9. 保存结果 ====
results = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Vol_measured": y_total,
    "Vol_predicted": y_pred,
    "Absolute Error": np.abs(y_total - y_pred),
    "Relative Error (%)": relative_error
})
results.to_excel("Exe预测结果_加slope特征_RF.xlsx", index=False)
print("✅ 已保存预测结果为: Exe预测结果_加slope特征_RF.xlsx")
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # 1. 读取数据
# file_path = "Pure component exergy 205.xlsx"
# df = pd.read_excel(file_path, sheet_name="Sheet1")
# df = df.dropna(subset=[df.columns[0]])
# df[df.columns[0]] = df[df.columns[0]].astype(int)
#
# # 2. 列定义
# group_cols = df.columns[12:31]   # M到AE: 基团浓度
# temp_cols = df.columns[31:41]    # AF到AO: 温度
# target_cols = df.columns[41:51]  # AP到AY: 目标值（exergy）
#
# # === 斜率回归模型训练 ===
# slope_medians = []
# for i, row in df.iterrows():
#     temps = row[temp_cols].values
#     targets = row[target_cols].values
#     slopes = [(targets[t+1]-targets[t])/(temps[t+1]-temps[t]) for t in range(len(temps)-1)]
#     slope_medians.append(np.median(slopes))
#
# target_slopes = np.array(slope_medians)
#
# # 斜率回归模型
# X_slope = df[group_cols].values  # 注意这里用 numpy 数组
# y_slope = target_slopes
#
# slope_model = LinearRegression()
# slope_model.fit(X_slope, y_slope)
#
# # 斜率预测及指标
# predicted_slopes = slope_model.predict(X_slope)  # 直接用 numpy 数组，不带列名
# mse_slope = mean_squared_error(y_slope, predicted_slopes)
# r2_slope = r2_score(y_slope, predicted_slopes)
# print(f"斜率模型 MSE = {mse_slope:.4f}, R² = {r2_slope:.4f}")
#
# # === Exergy 预测随机森林模型 ===
# X_total, y_total, material_ids, temperatures = [], [], [], []
#
# for i, row in df.iterrows():
#     material_id = row.iloc[0]
#     Nk = row[group_cols].values
#     temps = row[temp_cols].values
#     ex_values = row[target_cols].values
#
#     # 用 numpy 数组预测 slope，避免列名警告
#     slope = slope_model.predict(Nk.reshape(1, -1))[0]
#
#     for T, val in zip(temps, ex_values):
#         if np.isnan(T) or np.isnan(val):
#             continue
#         features = np.concatenate([Nk, [T], [slope]])
#         X_total.append(features)
#         y_total.append(val)
#         material_ids.append(material_id)
#         temperatures.append(T)
#
# X_total = np.array(X_total)
# y_total = np.array(y_total)
#
# # 随机森林训练
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_total, y_total)
#
# # 预测 Exergy
# y_pred = rf_model.predict(X_total)
# mse_rf = mean_squared_error(y_total, y_pred)
# r2_rf = r2_score(y_total, y_pred)
# print(f"随机森林模型 MSE = {mse_rf:.4f}, R² = {r2_rf:.4f}")
#
# # 保存预测结果
# results = pd.DataFrame({
#     "Material_ID": material_ids,
#     "Temperature (K)": temperatures,
#     "Ex_measured": y_total,
#     "Ex_predicted": y_pred
# })
# results.to_excel("Ex_predictions_with_slope_model_no_warning.xlsx", index=False)
# print("预测结果已保存为: Ex_predictions_with_slope_model_no_warning.xlsx")
