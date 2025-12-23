import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==== 1. 读取数据 ====
file_path = "Pure component exergy 205.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# ==== 2. 定义列索引 ====
group_cols = df.columns[12:31]   # 第14~25列：基团
temp_cols = df.columns[31:41]    # 第26~35列：温度
v_cols = df.columns[41:51]       # 第36~45列：目标变量 Hvap

# ==== 3. 构建训练数据（基团 + 基团×温度） ====
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    vols = row[v_cols].values

    for T, vol in zip(temps, vols):
        if np.isnan(T) or np.isnan(vol):
            continue
        # 特征：基团 + 基团×温度
        features = np.concatenate([Nk, Nk * T])
        X_total.append(features)
        y_total.append(vol)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)
material_ids = np.array(material_ids)
temperatures = np.array(temperatures)

# ==== 4. 拟合线性回归模型 ====
model = LinearRegression()
model.fit(X_total, y_total)

# ==== 5. 预测与评估 ====
y_pred = model.predict(X_total)
r2 = r2_score(y_total, y_pred)
mse = mean_squared_error(y_total, y_pred)
ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100

# 统计不同误差阈值内的点数
relative_error = np.abs((y_pred - y_total) / y_total) * 100
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print("\n📊 模型评估（基团 + 基团×温度交互项）：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")
print(f"相对误差 ≤ 1% 的点数: {within_1pct}")
print(f"相对误差 ≤ 5% 的点数: {within_5pct}")
print(f"相对误差 ≤ 10% 的点数: {within_10pct}")

# ==== 6. 保存预测结果 ====
df_result = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Vol_measured (J/mol)": y_total,
    "Vol_predicted (J/mol)": y_pred,
    "Absolute Error": np.abs(y_total - y_pred),
    "Relative Error (%)": relative_error
})

df_result.to_excel("exe预测结果_基团加温度_线性回归.xlsx", index=False)
print("✅ 已保存预测结果为: exe预测结果_基团加温度_线性回归.xlsx")

# ==== 7. 输出基团贡献系数（含交互项） ====
feature_labels = list(group_cols) + [f"{g}_T" for g in group_cols]
coefficients = pd.DataFrame({
    "Feature": feature_labels,
    "Coefficient": model.coef_
})
coefficients.to_excel("exe基团系数_线性回归.xlsx", index=False)
print("✅ 已保存模型系数为: exe基团系数_线性回归.xlsx")
