# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
#
# # ==== 1. 读取数据 ====
# file_path = "heat capacity 207.xlsx"
# df = pd.read_excel(file_path, sheet_name="Sheet1")
# df = df.dropna(subset=[df.columns[0]])
# df[df.columns[0]] = df[df.columns[0]].astype(int)
#
# # ==== 2. 定义列 ====
# group_cols = df.columns[11:30]   # 19个基团列
# temp_cols = df.columns[30:40]    # 10个温度点
# cp_cols = df.columns[40:50]      # 10个 Cp 值
#
# # ==== 3. 计算实际 slope（首末点斜率） ====
# slope_targets = [(row[cp_cols].values[-1] - row[cp_cols].values[0]) /
#                  (row[temp_cols].values[-1] - row[temp_cols].values[0])
#                  for i, row in df.iterrows()]
#
# df["slope_target"] = slope_targets
#
# # ==== 4. 用基团训练线性回归预测 slope ====
# X_slope = df[group_cols].values
# y_slope = df["slope_target"].values
# slope_model = LinearRegression()
# slope_model.fit(X_slope, y_slope)
#
# # 预测 slope
# slope_pred_all = slope_model.predict(X_slope)
#
# # ==== 5. 保存实际 slope 与预测 slope ====
# slope_results = pd.DataFrame({
#     "Material_ID": df.iloc[:, 0],
#     "Slope_actual": y_slope,
#     "Slope_predicted": slope_pred_all
# })
#
# slope_results.to_excel("Slope_prediction_results.xlsx", index=False)
# print("✅ 已保存实际斜率与预测斜率结果为: Slope_prediction_results.xlsx")


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==== 1. 读取数据 ====
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.dropna(subset=[df.columns[0]])
df[df.columns[0]] = df[df.columns[0]].astype(int)

# ==== 2. 定义列 ====
group_cols = df.columns[11:30]   # 19个基团列
temp_cols = df.columns[30:40]    # 10个温度点
cp_cols = df.columns[40:50]      # 10个 Cp 值

# ==== 3. 计算实际 slope（相邻点斜率中位数） ====
slope_targets = []
for i, row in df.iterrows():
    temps = row[temp_cols].values
    cps = row[cp_cols].values
    slopes = [(cps[t+1] - cps[t]) / (temps[t+1] - temps[t]) for t in range(len(temps)-1)]
    median_slope = np.median(slopes)
    slope_targets.append(median_slope)

df["slope_target"] = slope_targets

# ==== 4. 用基团训练线性回归预测 slope ====
X_slope = df[group_cols].values
y_slope = df["slope_target"].values

slope_model = LinearRegression()
slope_model.fit(X_slope, y_slope)

# 预测 slope
slope_pred_all = slope_model.predict(X_slope)

# ==== 5. 评估 slope 预测精度 ====
r2_slope = r2_score(y_slope, slope_pred_all)
mse_slope = mean_squared_error(y_slope, slope_pred_all)
ard_slope = np.mean(np.abs((slope_pred_all - y_slope) / y_slope)) * 100

print("\n📊 基团线性回归预测 slope（中位数）评估：")
print(f"R²_slope  = {r2_slope:.4f}")
print(f"MSE_slope = {mse_slope:.4f}")
print(f"ARD_slope = {ard_slope:.2f}%")

# ==== 6. 保存实际 slope 与预测 slope ====
slope_results = pd.DataFrame({
    "Material_ID": df.iloc[:, 0],
    "Slope_actual_median": y_slope,
    "Slope_predicted": slope_pred_all
})

slope_results.to_excel("Slope_prediction_median_results.xlsx", index=False)
print("✅ 已保存实际斜率与预测斜率（中位数）结果为: Slope_prediction_median_results.xlsx")
