import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 读取包含 slopeT 特征的数据
df = pd.read_excel("Transformed_hp_with_slope.xlsx")

# 分离特征和目标变量
X = df.drop(columns=["Heat Capacity"])
y = df["Heat Capacity"]

# 模型训练
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# 模型预测
y_pred = model.predict(X)

# 评估指标
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
relative_error = np.abs((y - y_pred) / y) * 100
ard = np.mean(relative_error)

# 误差统计
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print("📊 模型评估结果：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.4f}")
print(f"ARD = {ard:.2f}%")
print(f"✅ 误差 ≤ 1% 的点数: {within_1pct}")
print(f"✅ 误差 ≤ 5% 的点数: {within_5pct}")
print(f"✅ 误差 ≤ 10% 的点数: {within_10pct}")

# 保存结果
comparison_df = df.copy()
comparison_df["Predicted_Heat_Capacity"] = y_pred
comparison_df["Absolute_Error"] = np.abs(y - y_pred)
comparison_df["Relative_Error (%)"] = relative_error

comparison_df.to_excel("prediction_vs_actual_hp_with_slopeT.xlsx", index=False)
print("✅ 已保存 prediction_vs_actual_hp_with_slopeT.xlsx")
