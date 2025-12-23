import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 数据读取
file_path = "heat capacity 207.xlsx"  # 文件路径
df = pd.read_excel(file_path)

# 分组、温度、热容列索引
group_cols = df.columns[11:30]   # 12个基团列
temp_cols = df.columns[30:40]    # 10个温度点
cp_cols = df.columns[40:50]      # 10个 Cp 值

# 2. 构建全部样本
X_total = []
y_total = []
material_ids = []
temperatures = []

for _, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    cps = row[cp_cols].values

    for T, cp in zip(temps, cps):
        if np.isnan(T) or np.isnan(cp):
            continue
        features = np.concatenate([Nk, Nk * T])
        X_total.append(features)
        y_total.append(cp)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)
material_ids = np.array(material_ids)
temperatures = np.array(temperatures)

# 3. 模型训练（使用全部数据）
model = LinearRegression()
model.fit(X_total, y_total)

# 4. 预测与评估
y_pred = model.predict(X_total)
mse = mean_squared_error(y_total, y_pred)
r2 = r2_score(y_total, y_pred)
ard = np.mean(np.abs((y_total - y_pred) / y_total)) * 100

# ==== 新增：误差统计 ==== #
relative_error = np.abs((y_pred - y_total) / y_total) * 100
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print(f"📊 全数据训练评估结果:")
print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")
print(f"误差 ≤ 1% 的点数: {within_1pct}")
print(f"误差 ≤ 5% 的点数: {within_5pct}")
print(f"误差 ≤ 10% 的点数: {within_10pct}")
print(f"📊 全数据训练评估结果:")
print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")

# 5. 输出预测结果
results = pd.DataFrame({
    'Material_ID': material_ids,
    'Temperature (K)': temperatures,
    'Cp_measured': y_total,
    'Cp_predicted': y_pred
})
results.to_excel("全数据_线性Cp预测结果.xlsx", index=False)
print("✅ Cp 预测结果已保存为: 全数据_线性Cp预测结果.xlsx")

# 6. 输出基团贡献系数（含温度项）
coefficients = pd.DataFrame({
    'Group': list(group_cols) + [f'{group}_T' for group in group_cols],
    'Contribution': model.coef_
})
print("\n📈 基团贡献系数（包含乘温项）:")
print(coefficients.sort_values(by='Contribution', ascending=False).to_string(index=False))
