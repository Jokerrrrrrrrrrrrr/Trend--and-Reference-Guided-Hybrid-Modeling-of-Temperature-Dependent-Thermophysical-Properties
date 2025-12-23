import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# ========= 1. 读取数据 =========
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.dropna(subset=[df.columns[0]])
df[df.columns[0]] = df[df.columns[0]].astype(int)

# ========= 2. 列定义 =========
group_cols = df.columns[11:30]   # 19个基团列
temp_cols = df.columns[30:40]    # 10个温度点
cp_cols = df.columns[40:50]      # 10个 Cp 值
target_column_T1 = 'ASPEN Half Critical T'

# ========= 3. 子模型训练 =========
X_groups = df[group_cols]
valid_mask = ~df[target_column_T1].isna()

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_groups[valid_mask])

# 改用 GradientBoostingRegressor 预测 T1
y_T1 = df.loc[valid_mask, target_column_T1].values
T1_model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
).fit(X_poly, y_T1)

# ========= 3a. 用基团训练线性回归预测 slope =========
# 计算每个物质的目标 slope（首末点斜率）
slope_targets = [(row[cp_cols].values[-1] - row[cp_cols].values[0]) /
                 (row[temp_cols].values[-1] - row[temp_cols].values[0])
                 for i, row in df.iterrows()]

df["slope_target"] = slope_targets

# 用基团训练线性回归预测 slope
X_slope = df[group_cols].values
y_slope = df["slope_target"].values
slope_model = LinearRegression()
slope_model.fit(X_slope, y_slope)

# 预测 slope
slope_pred_all = slope_model.predict(X_slope)

# ========= 3b. 输出 slope 预测效果 =========
r2_slope = r2_score(y_slope, slope_pred_all)
mse_slope = mean_squared_error(y_slope, slope_pred_all)
ard_slope = np.mean(np.abs((slope_pred_all - y_slope) / y_slope)) * 100

print("\n📊 基团线性回归预测 slope 评估：")
print(f"R²_slope  = {r2_slope:.4f}")
print(f"MSE_slope = {mse_slope:.4f}")
print(f"ARD_slope = {ard_slope:.2f}%")

# ========= 4. 构建 Cp 主模型训练数据 =========
X_total, y_total, material_ids, temperatures = [], [], [], []

X_poly_all = poly.transform(X_groups)
for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    cps = row[cp_cols].values
    slope_pred = slope_pred_all[i]

    for T, Cp in zip(temps, cps):
        if np.isnan(T) or np.isnan(Cp) or np.isnan(slope_pred):
            continue

        features = np.concatenate([
            Nk,           # 19 个基团
            Nk * T,       # 19 个交互项
            [slope_pred * T]   # slope × T
        ])
        X_total.append(features)
        y_total.append(Cp)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)

# ========= 5. 拟合 Cp 主模型（Huber） =========
model = HuberRegressor(max_iter=10000).fit(X_total, y_total)

# ========= 6. Cp 模型评估 =========
y_pred = model.predict(X_total)
mse = mean_squared_error(y_total, y_pred)
r2 = r2_score(y_total, y_pred)
ard = np.mean(np.abs((y_total - y_pred) / y_total)) * 100

relative_error = np.abs((y_pred - y_total) / y_total) * 100
within_1pct = np.sum(relative_error <= 1)
within_5pct = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)

print("\n📊 Cp 主模型评估（含 slope×T 特征）：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")
print(f"✅ 误差 ≤ 1% 的数据点数量: {within_1pct}")
print(f"✅ 误差 ≤ 5% 的数据点数量: {within_5pct}")
print(f"✅ 误差 ≤ 10% 的数据点数量: {within_10pct}")

# ========= 7. 保存 Cp 预测结果 =========
results = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Cp_measured": y_total,
    "Cp_predicted": y_pred
})
results.to_excel("Cp预测结果_slopeT_linear回归.xlsx", index=False)
print("✅ 已保存预测结果为: Cp预测结果_slopeT_linear回归.xlsx")

# ========= 8. 保存 Cp 模型系数 =========
feature_labels = list(group_cols) + [f"{g}_T" for g in group_cols] + ["slope×T"]
coefficients = pd.DataFrame({
    "Feature": feature_labels,
    "Contribution": model.coef_
})
coefficients.to_excel("Cp系数表_slopeT_linear回归.xlsx", index=False)
print("📈 已保存模型系数为: Cp系数表_slopeT_linear回归.xlsx")
