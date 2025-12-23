# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # 读取新的包含 slopeT 特征的数据
# df = pd.read_csv("Transformed_Training_Data_with_slopeT.csv")
#
# # 分离特征和目标变量
# X = df.drop(columns=["Heat_Capacity"])
# y = df["Heat_Capacity"]
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
#
# print(f"R²: {r2:.4f}")
# print(f"MSE: {mse:.4f}")
#
# # 生成对比表并保存为 Excel
# comparison_df = df.copy()
# comparison_df["Predicted_Heat_Capacity"] = y_pred
# comparison_df.to_excel("prediction_vs_actual_with_slopeT.xlsx", index=False)
# print("✅ 已保存 prediction_vs_actual_with_slopeT.xlsx")


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 读取包含 slopeT 特征的数据
df = pd.read_csv("Transformed_Training_Data_with_slopeT.csv")

# 分离特征和目标变量
X = df.drop(columns=["Heat_Capacity"])
y = df["Heat_Capacity"]

# 模型训练
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# 模型预测
y_pred = model.predict(X)

# 评估指标输出
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
ard = np.mean(np.abs((y - y_pred) / y)) * 100  # 计算 ARD

print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"ARD: {ard:.2f}%")

# 生成对比表并保存为 Excel
comparison_df = df.copy()
comparison_df["Predicted_Heat_Capacity"] = y_pred
comparison_df.to_excel("prediction_vs_actual_with_slopeT.xlsx", index=False)
print("✅ 已保存 prediction_vs_actual_with_slopeT.xlsx")
