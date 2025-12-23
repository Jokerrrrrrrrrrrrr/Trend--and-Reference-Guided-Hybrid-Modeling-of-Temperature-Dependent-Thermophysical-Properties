import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. 读取数据
file_path = "heat of vaporization 204.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 2. 定义列索引
group_cols = df.columns[13:32]   # 第14~25列，基团
temp_cols = df.columns[32:42]    # 第26~35列，温度
hv_cols = df.columns[42:52]      # 第36~45列，Hvap

# 3. 构建训练数据
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    hvaps = row[hv_cols].values

    for T, Hv in zip(temps, hvaps):
        if np.isnan(T) or np.isnan(Hv):
            continue
        features = np.concatenate([Nk, [T]])
        X_total.append(features)
        y_total.append(Hv)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)

# 4. 拟合模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_total, y_total)

# 5. 评估模型
y_pred = model.predict(X_total)
r2 = r2_score(y_total, y_pred)
mse = mean_squared_error(y_total, y_pred)
ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100  # 平均相对偏差 (%)

print("\n📊 模型评估（基团 + 温度 特征）：")
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

# 6. 保存预测结果
df_result = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Hvap_measured (J/mol)": y_total,
    "Hvap_predicted (J/mol)": y_pred,
    "Absolute Error": np.abs(y_total - y_pred),
    "Relative Error (%)": 100 * np.abs((y_total - y_pred) / y_total)
})

df_result.to_excel("Hvap预测结果_基团加温度_RF.xlsx", index=False)
print("✅ 已保存预测结果为: Hvap预测结果_基团加温度_RF.xlsx")
