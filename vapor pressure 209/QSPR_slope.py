# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # 读取包含 slopeT 特征的数据
# df = pd.read_excel("Transformed_vp_with_slope.xlsx")
#
# # 分离特征和目标变量
# X = df.drop(columns=["Vapor Pressure"])
# y = df["Vapor Pressure"]
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
# comparison_df.to_excel("prediction_vs_actual_vp_with_slopeT.xlsx", index=False)
# print("✅ 已保存 prediction_vs_actual_vp_with_slopeT.xlsx")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 读取包含 slopeT 特征的数据
df = pd.read_excel("Transformed_vp_with_slope.xlsx")

# 分离特征和目标变量
X = df.drop(columns=["Vapor Pressure"])
y = df["Vapor Pressure"]

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

# 计算相对误差
relative_error = np.abs((y_pred - y) / y) * 100

# 统计误差小于 1%、5%、10% 的数据点数量
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

# 输出误差统计
print(f"✅ 误差 ≤ 1% 的点数: {within_1pct}")
print(f"✅ 误差 ≤ 5% 的点数: {within_5pct}")
print(f"✅ 误差 ≤ 10% 的点数: {within_10pct}")

# 生成对比表并保存为 Excel
comparison_df = df.copy()
comparison_df["Predicted_Volume"] = y_pred
comparison_df["Absolute_Error"] = np.abs(y - y_pred)
comparison_df["Relative_Error (%)"] = relative_error
comparison_df.to_excel("prediction_vs_actual_vp_with_slopeT_and_error_stats.xlsx", index=False)

print("✅ 已保存预测结果为: prediction_vs_actual_vp_with_slopeT_and_error_stats.xlsx")
