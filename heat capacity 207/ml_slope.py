import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. 读取数据
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.dropna(subset=[df.columns[0]])
df[df.columns[0]] = df[df.columns[0]].astype(int)

# 2. 列定义
group_cols = df.columns[11:30]   # 12个基团列
temp_cols = df.columns[30:40]    # 10个温度点
cp_cols = df.columns[40:50]
target_column_T1 = 'ASPEN Half Critical T'
Tc0 = 138

# 3. 子模型训练：用于估算 T1, Cp1, Cp2 → 计算 slope
X_groups = df[group_cols]
valid_mask = ~df[target_column_T1].isna()
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_groups[valid_mask])
y_exp_T1 = np.exp(df.loc[valid_mask, target_column_T1] / Tc0)

T1_model = HuberRegressor(max_iter=9000).fit(X_poly, y_exp_T1)
Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 9])
Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 50])

# 4. 构建训练数据
X_total, y_total, material_ids, temperatures = [], [], [], []
X_poly_all = poly.transform(X_groups)

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    cps = row[cp_cols].values

    Nk_df = pd.DataFrame([Nk], columns=group_cols)
    Nk_poly = X_poly_all[i:i+1]

    try:
        T1_exp = T1_model.predict(Nk_poly)[0]
        if T1_exp <= 0 or np.isnan(T1_exp):
            continue
        T1 = Tc0 * np.log(T1_exp)
        T2 = T1 * 1.5
        Cp1 = Cp1_model.predict(Nk_df)[0]
        Cp2 = Cp2_model.predict(Nk_df)[0]
        slope = (Cp2 - Cp1) / (T2 - T1)
    except:
        continue

    for T, Cp in zip(temps, cps):
        if np.isnan(T) or np.isnan(Cp):
            continue

        features = np.concatenate([
            Nk,            # 12 个基团
            [T],           # 温度
            [slope*T]    # slope × T
        ])
        X_total.append(features)
        y_total.append(Cp)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)

# 5. 拟合机器学习模型（随机森林）
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_total, y_total)

# 6. 评估模型
y_pred = model.predict(X_total)
mse = mean_squared_error(y_total, y_pred)
r2 = r2_score(y_total, y_pred)
ard = np.mean(np.abs((y_total - y_pred) / y_total)) * 100  # <-- 新增 ARD

# === 新增误差范围统计 ===
relative_error = np.abs((y_total - y_pred) / y_total) * 100
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print("\n📊 模型评估（含 slope×T 特征）：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")
print(f"✅ 误差 ≤ 1% 的点数: {within_1pct}")
print(f"✅ 误差 ≤ 5% 的点数: {within_5pct}")
print(f"✅ 误差 ≤ 10% 的点数: {within_10pct}")

print("\n📊 模型评估（含 slope×T 特征）：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")  # <-- 输出 ARD

# 7. 保存预测结果
results = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Cp_measured": y_total,
    "Cp_predicted": y_pred
})
results.to_excel("Cp预测结果_slopeT特征_RF模型.xlsx", index=False)
print("✅ 已保存预测结果为: Cp预测结果_slopeT特征_RF模型.xlsx")
