import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

# ==== 1. 读取数据 ====
df = pd.read_excel("heat of vaporization 204.xlsx", sheet_name="Sheet1")

# ==== 2. 定义列 ====
group_cols = df.columns[13:32]   # 第14~25列：基团
temp_cols = df.columns[32:42]    # 第26~35列：温度
hvap_cols = df.columns[42:52]    # 第36~45列：Hvap

# ==== 3. 准备 slope 所需模型输入 ====
df_298 = pd.read_excel("selected_25_descriptors_data_298.xlsx")
X_298 = df_298.drop(columns=["Heat of vaporization at normal temperature"])
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["Heat of vaporization at normal temperature"])
HVap_298_all = rf_298.predict(X_298)

df_Tb = pd.read_excel("selected_25_descriptors_data_boiling_point.xlsx")
X_Tb = df_Tb.drop(columns=["Heat of vaporization at boiling temperature"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["Heat of vaporization at boiling temperature"])
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 4. Tb 模型预测 ====
Nk_all = df.iloc[:, 13:25].apply(pd.to_numeric, errors='coerce')
Tb_raw = df.iloc[:, 5].values
Tb0 = 222.543
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)

mask_tb = ~np.isnan(Tb_raw)
model_Tb = HuberRegressor(max_iter=5000).fit(Nk_poly[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))

# ==== 5. 计算 slope 并加入主 DataFrame ====
T_ref = 298.15
slope_values = (HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)
df["slope"] = slope_values

# ==== 6. 计算残差 ====
Cp1_residual = HVap_Tb_all - df_Tb["Heat of vaporization at boiling temperature"].values
Cp2_residual = HVap_298_all - df_298["Heat of vaporization at normal temperature"].values

# ==== 7. 扩展 Cp1_residual 和 Cp2_residual ====
Cp1_residual_2d = Cp1_residual.repeat(10, axis=0).reshape(-1, 1)  # 将 Cp1_residual 扩展为 2040 行
Cp2_residual_2d = Cp2_residual.repeat(10, axis=0).reshape(-1, 1)  # 将 Cp2_residual 扩展为 2040 行

# ==== 8. 构建训练数据 ====
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    hvaps = row[hvap_cols].values
    slope = row["slope"]

    # 重复特征构建，确保所有数据行数一致
    for T, Hv in zip(temps, hvaps):
        if np.isnan(T) or np.isnan(Hv) or np.isnan(slope):
            continue
        features = np.concatenate([Nk, [T], [slope], Cp1_residual_2d[i], Cp2_residual_2d[i]])  # 加入扩展的残差
        X_total.append(features)
        y_total.append(Hv)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)

# ==== 9. 拟合模型 ====
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_total, y_total)

# ==== 10. 模型评估 ====
y_pred = model.predict(X_total)
r2 = r2_score(y_total, y_pred)
mse = mean_squared_error(y_total, y_pred)
ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100  # ARD %

print("\n📊 模型评估（基团 + 温度 + slope 特征）：")
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

# ==== 11. 保存结果 ====
results = pd.DataFrame({
    "Material_ID": material_ids,                  # 化合物ID
    "Temperature (K)": temperatures,              # 温度
    "Hvap_measured": y_total,                     # 真实的蒸发热
    "Hvap_predicted": y_pred,                     # 预测的蒸发热
    "Absolute Error": np.abs(y_total - y_pred),   # 绝对误差
    "Relative Error (%)": 100 * np.abs((y_total - y_pred) / y_total),  # 相对误差
    "Cp1_residual": Cp1_residual_2d.flatten(),    # 使用计算得到的 Cp1 残差
    "Cp2_residual": Cp2_residual_2d.flatten()     # 使用计算得到的 Cp2 残差
})

# 输出结果并保存
results.to_excel("Hvap_prediction_with_slopeT_and_intercept_19group_with_residuals.xlsx", index=False)
print("✅ 预测结果已保存为 Hvap_prediction_with_slopeT_and_intercept_19group_with_residuals.xlsx")
