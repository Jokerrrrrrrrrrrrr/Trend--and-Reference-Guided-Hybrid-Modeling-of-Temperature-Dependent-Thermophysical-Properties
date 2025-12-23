import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
df = pd.read_excel("selected_25_descriptors_data_298.xlsx")

# 分离特征和目标变量
X = df.drop(columns=["Heat of vaporization at normal temperature"])
y = df["Heat of vaporization at normal temperature"]

# 模型训练
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# 模型预测
y_pred = model.predict(X)

# 评估指标输出
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# 生成对比表并保存为 Excel
comparison_df = X.copy()
comparison_df["Actual_Heat_vaporization_298"] = y
comparison_df["Predicted_Heat_vaporization_298"] = y_pred
comparison_df.to_excel("prediction_vs_actual_298.xlsx", index=False)
