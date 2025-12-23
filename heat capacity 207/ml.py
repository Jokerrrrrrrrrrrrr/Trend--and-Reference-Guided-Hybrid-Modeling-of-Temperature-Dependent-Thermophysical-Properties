import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. 读取数据
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.dropna(subset=[df.columns[0]])
df[df.columns[0]] = df[df.columns[0]].astype(int)

# 2. 定义列
group_cols = df.columns[11:30]   # 12个基团列
temp_cols = df.columns[30:40]    # 10个温度点
cp_cols = df.columns[40:50]
# 3. 构建训练数据
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values      # 基团数量
    temps = row[temp_cols].values    # 温度点
    cps = row[cp_cols].values        # Cp 点

    for T, Cp in zip(temps, cps):
        if np.isnan(T) or np.isnan(Cp):
            continue

        features = np.concatenate([Nk, [T]])  # 添加温度为第13列
        X_total.append(features)
        y_total.append(Cp)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)

# 4. 拟合模型
model = RandomForestRegressor(n_estimators=100, random_state=40)
model.fit(X_total, y_total)

# 5. 模型评估（用于所有数据）
y_pred = model.predict(X_total)
mse = mean_squared_error(y_total, y_pred)
r2 = r2_score(y_total, y_pred)
ard = np.mean(np.abs((y_total - y_pred) / y_total)) * 100

# === 新增误差范围统计 ===
relative_error = np.abs((y_total - y_pred) / y_total) * 100
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print("\n📊 模型评估（基团 + 温度 特征）：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")
print(f"✅ 误差 ≤ 1% 的点数: {within_1pct}")
print(f"✅ 误差 ≤ 5% 的点数: {within_5pct}")
print(f"✅ 误差 ≤ 10% 的点数: {within_10pct}")
print("\n📊 模型评估（基团 + 温度 特征）：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")

# 6. 保存预测结果
results = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Cp_measured": y_total,
    "Cp_predicted": y_pred
})
results.to_excel("Cp预测结果_基团加温度_RF.xlsx", index=False)
print("✅ 已保存预测结果为: Cp预测结果_基团加温度_RF.xlsx")
