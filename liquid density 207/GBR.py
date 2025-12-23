import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor  # 改为导入梯度提升
from sklearn.metrics import mean_squared_error, r2_score

# 1. 读取数据
file_path = "liquid density.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 2. 定义列索引
group_cols = df.columns[12:31]   # 第14~25列，基团
temp_cols = df.columns[31:41]    # 第26~35列，温度
v_cols = df.columns[41:51]      # 第36~45列，体积/密度

# 3. 构建训练数据
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    vols = row[v_cols].values

    for T, vol in zip(temps, vols):
        if np.isnan(T) or np.isnan(vol):
            continue
        features = np.concatenate([Nk, [T]])
        X_total.append(features)
        y_total.append(vol)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)

# 4. 拟合模型 - 改为梯度提升回归
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
model.fit(X_total, y_total)

# 5. 评估模型
y_pred = model.predict(X_total)
r2 = r2_score(y_total, y_pred)
mse = mean_squared_error(y_total, y_pred)
ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100  # 平均相对偏差 (%)

print("\n📊 梯度提升回归模型评估（基团 + 温度 特征）：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.8f}")
print(f"ARD = {ard:.2f}%")

# 计算相对误差
relative_error = np.abs((y_pred - y_total) / y_total) * 100

# 统计不同误差阈值内的点数
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print(f"✅ 相对误差 ≤ 1% 的点数: {within_1pct}")
print(f"✅ 相对误差 ≤ 5% 的点数: {within_5pct}")
print(f"✅ 相对误差 ≤ 10% 的点数: {within_10pct}")

# 6. 保存预测结果
df_result = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Density_measured": y_total,  # 修正列名，假设是密度
    "Density_predicted": y_pred,  # 修正列名，假设是密度
    "Absolute Error": np.abs(y_total - y_pred),
    "Relative Error (%)": relative_error
})

df_result.to_excel("Liquid_density预测结果_基团加温度_GBR.xlsx", index=False)
print("✅ 已保存预测结果为: Liquid_density预测结果_基团加温度_GBR.xlsx")