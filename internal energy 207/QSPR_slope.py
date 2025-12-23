# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # 读取包含 slopeT 特征的数据
# df = pd.read_excel("Transformed_internal_energy_with_slope.xlsx")
#
# # 分离特征和目标变量
# X = df.drop(columns=["Internal energy"])
# y = df["Internal energy"]
#
# # 模型训练
# model = RandomForestRegressor(random_state=42)
# model.fit(X, y)
#
# # 模型预测
# y_pred = model.predict(X)
#
# # 评估指标输出
# r2 = r2_score(y, y_pred)
# mse = mean_squared_error(y, y_pred)
# ard = np.mean(np.abs((y_pred - y) / y)) * 100  # 平均相对偏差（百分比）
#
# print("📊 模型评估结果：")
# print(f"R²  = {r2:.4f}")
# print(f"MSE = {mse:.4f}")
# print(f"ARD = {ard:.2f}%")
#
# # 生成对比表并保存为 Excel
# comparison_df = df.copy()
# comparison_df["Predicted_Volume"] = y_pred
# comparison_df["Absolute_Error"] = np.abs(y - y_pred)
# comparison_df["Relative_Error (%)"] = 100 * np.abs((y - y_pred) / y)
#
# comparison_df.to_excel("prediction_vs_actual_Internal_energy_with_slopeT.xlsx", index=False)
# print("✅ 已保存 prediction_vs_actual_Internal_energy_with_slopeT.xlsx")


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 读取包含 slopeT 特征的数据
df = pd.read_excel("Transformed_internal_energy_with_slope.xlsx")

# 分离特征和目标变量
X = df.drop(columns=["Internal energy"])
y = df["Internal energy"]

# 模型训练
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# 模型预测
y_pred = model.predict(X)

# 评估指标输出
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
ard = np.mean(np.abs((y_pred - y) / y)) * 100  # 平均相对偏差（百分比）

print("📊 模型评估结果：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.4f}")
print(f"ARD = {ard:.2f}%")

# 生成相对误差
relative_error = 100 * np.abs((y - y_pred) / y)

# 统计相对误差小于1%, 5%, 10%的数据点数量
error_1_percent = np.sum(relative_error < 1)
error_5_percent = np.sum(relative_error < 5)
error_10_percent = np.sum(relative_error < 10)

print("\n📊 统计结果：")
print(f"数据点相对误差小于1%: {error_1_percent}个")
print(f"数据点相对误差小于5%: {error_5_percent}个")
print(f"数据点相对误差小于10%: {error_10_percent}个")

# 生成对比表并保存为 Excel
comparison_df = df.copy()
comparison_df["Predicted_Volume"] = y_pred
comparison_df["Absolute_Error"] = np.abs(y - y_pred)
comparison_df["Relative_Error (%)"] = relative_error

# 保存结果到 Excel
comparison_df.to_excel("prediction_vs_actual_Internal_energy_with_slopeT.xlsx", index=False)
print("✅ 已保存 prediction_vs_actual_Internal_energy_with_slopeT.xlsx")
