# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ==== 1. 读取数据 ====
# df = pd.read_excel("pure component isentropic exponent 207.xlsx", sheet_name="Sheet1")
#
# # ==== 2. 定义列 ====
# group_cols = df.columns[12:31]   # 第14~32列：基团
# temp_cols = df.columns[31:41]    # 第33~42列：温度
# v_cols = df.columns[41:51]       # 第43~52列：目标变量 Vol
#
# # ==== 3. 计算每个物质的目标 slope（首末点斜率） ====
# slope_targets = [(row[v_cols].values[-1] - row[v_cols].values[0]) /
#                  (row[temp_cols].values[-1] - row[temp_cols].values[0])
#                  for i, row in df.iterrows()]
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
# # ==== 5. 预测 slope ====
# slope_pred = slope_model.predict(X_slope)
#
# # ==== 6. 评估模型精度 ====
# r2 = r2_score(y_slope, slope_pred)
# mse = mean_squared_error(y_slope, slope_pred)
# ard = np.mean(np.abs((slope_pred - y_slope) / y_slope)) * 100
#
# print("\n📊 基团线性回归预测 slope 评估：")
# print(f"R²  = {r2:.4f}")
# print(f"MSE = {mse:.4f}")
# print(f"ARD = {ard:.2f}%")
#
# # ==== 7. 保存结果（可选） ====
# results = pd.DataFrame({
#     "Material_ID": df.iloc[:, 0],
#     "Slope_measured": y_slope,
#     "Slope_predicted": slope_pred
# })
# results.to_excel("Slope_prediction_linear_regression.xlsx", index=False)
# print("✅ 预测结果已保存为: Slope_prediction_linear_regression.xlsx")

#
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
#
# # ==== 1. 读取数据 ====
# df = pd.read_excel("pure component isentropic exponent 207.xlsx", sheet_name="Sheet1")
#
# # ==== 2. 定义列 ====
# group_cols = df.columns[12:31]   # 第14~32列：基团
# temp_cols = df.columns[31:41]    # 第33~42列：温度
# v_cols = df.columns[41:51]       # 第43~52列：目标变量 Vol
#
# # ==== 3. 计算每个物质的目标 slope（相邻点斜率中位数） ====
# slope_targets = []
# for i, row in df.iterrows():
#     temps = row[temp_cols].values
#     vols = row[v_cols].values
#
#     slopes = [(vols[t+1] - vols[t]) / (temps[t+1] - temps[t]) for t in range(len(temps)-1)]
#     median_slope = np.median(slopes)
#     slope_targets.append(median_slope)
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
# # ==== 5. 预测 slope ====
# slope_pred = slope_model.predict(X_slope)
#
# # ==== 6. 评估模型精度 ====
# r2 = r2_score(y_slope, slope_pred)
# mse = mean_squared_error(y_slope, slope_pred)
# ard = np.mean(np.abs((slope_pred - y_slope) / y_slope)) * 100
#
# print("\n📊 基团线性回归预测 slope（中位数）评估：")
# print(f"R²  = {r2:.4f}")
# print(f"MSE = {mse:.4f}")
# print(f"ARD = {ard:.2f}%")
#
# # ==== 7. 保存结果（可选） ====
# results = pd.DataFrame({
#     "Material_ID": df.iloc[:, 0],
#     "Slope_measured": y_slope,
#     "Slope_predicted": slope_pred
# })
# results.to_excel("Slope_prediction_linear_regression_median.xlsx", index=False)
# print("✅ 预测结果已保存为: Slope_prediction_linear_regression_median.xlsx")



import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==== 1. 读取数据 ====
df = pd.read_excel("pure component isentropic exponent 207.xlsx", sheet_name="Sheet1")

# ==== 2. 定义列 ====
group_cols = df.columns[12:31]   # 第14~32列：基团
temp_cols = df.columns[31:41]    # 第33~42列：温度
v_cols = df.columns[41:51]       # 第43~52列：目标变量 Vol

# ==== 3. 计算每个物质的目标 slope（首末点斜率） ====
slope_targets = [(row[v_cols].values[-1] - row[v_cols].values[0]) /
                 (row[temp_cols].values[-1] - row[temp_cols].values[0])
                 for i, row in df.iterrows()]

df["slope_target"] = slope_targets

# ==== 4. 用基团训练 Huber 回归预测 slope ====
X_slope = df[group_cols].values
y_slope = df["slope_target"].values

slope_model = HuberRegressor(max_iter=10000)
slope_model.fit(X_slope, y_slope)

# ==== 5. 预测 slope ====
slope_pred = slope_model.predict(X_slope)

# ==== 6. 评估模型精度 ====
r2 = r2_score(y_slope, slope_pred)
mse = mean_squared_error(y_slope, slope_pred)
ard = np.mean(np.abs((slope_pred - y_slope) / y_slope)) * 100

print("\n📊 基团 Huber 回归预测 slope 评估：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.4f}")
print(f"ARD = {ard:.2f}%")

# ==== 7. 保存预测结果 ====
results = pd.DataFrame({
    "Material_ID": df.iloc[:, 0],
    "Slope_measured": y_slope,
    "Slope_predicted": slope_pred
})
results.to_excel("Slope_prediction_Huber.xlsx", index=False)
print("✅ 预测结果已保存为: Slope_prediction_Huber.xlsx")
