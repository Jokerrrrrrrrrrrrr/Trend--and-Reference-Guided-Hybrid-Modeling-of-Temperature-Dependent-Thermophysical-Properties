import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==== 1. 数据加载 ====
df = pd.read_excel("liquid density.xlsx", sheet_name="Sheet1")

group_cols = df.columns[12:31]
temp_cols = df.columns[31:41]
v_cols = df.columns[41:51]

# ==== 2. Hvap 模型（298 K） ====
df_298 = pd.read_excel("selected_25_descriptors_normal.xlsx")
X_298 = df_298.drop(columns=["ASPEN Liquid Density at Normal Temperature(g/cc)"])
y_298 = df_298["ASPEN Liquid Density at Normal Temperature(g/cc)"]
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, y_298)
HVap_298_all = rf_298.predict(X_298)

# ==== 3. Hvap 模型（Tb） ====
df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
X_Tb = df_Tb.drop(columns=["ASPEN Liquid Density at BoilingTemperature(g/cc)"])
y_Tb = df_Tb["ASPEN Liquid Density at BoilingTemperature(g/cc)"]
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, y_Tb)
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 4. Tb 模型预测（标准化 + 多项式） ====
Nk_all = df[group_cols].apply(pd.to_numeric, errors='coerce')
Tb_raw = df.iloc[:, 5].values
Tb0 = 222.543

poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)
scaler = StandardScaler()
Nk_scaled = scaler.fit_transform(Nk_poly)

mask_tb = ~np.isnan(Tb_raw)
model_Tb = HuberRegressor(max_iter=10000)
model_Tb.fit(Nk_scaled[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_scaled), 1e-6, None))

# ==== 5. 计算 slope 并加入主 DataFrame ====
T_ref = 298.15
slope_values = (HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)
df["slope"] = slope_values

# ==== 6. 计算残差 ====
# 计算 Cp1 残差（Density_Tb_all - 真实的 Density_Tb）
Cp1_residual = HVap_Tb_all - df_Tb["ASPEN Liquid Density at BoilingTemperature(g/cc)"].values

# 计算 Cp2 残差（Density_298_all - 真实的 Density_298）
Cp2_residual = HVap_298_all - df_298["ASPEN Liquid Density at Normal Temperature(g/cc)"].values

# ==== 7. 扩展残差数据，确保每个物质对应 10 行 ====
Cp1_residual_expanded = Cp1_residual.repeat(10)  # 每个残差扩展 10 行
Cp2_residual_expanded = Cp2_residual.repeat(10)  # 每个残差扩展 10 行

# ==== 8. 构建训练数据 ====
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    vols = row[v_cols].values
    slope = row["slope"]

    # 重复特征构建，确保所有数据行数一致
    for T, vol in zip(temps, vols):
        if np.isnan(T) or np.isnan(vol) or np.isnan(slope):
            continue
        # 加入扩展的残差特征
        features = np.concatenate([Nk, [T], [slope], [Cp1_residual_expanded[i]], [Cp2_residual_expanded[i]]])
        X_total.append(features)
        y_total.append(vol)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)

# ==== 9. 随机森林模型训练 ====
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_total, y_total)

# ==== 10. 模型评估 ====
y_pred = model.predict(X_total)
r2 = r2_score(y_total, y_pred)
mse = mean_squared_error(y_total, y_pred)
ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100

print("\n📊 模型评估：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")

relative_error = np.abs((y_pred - y_total) / y_total) * 100
print(f"相对误差 ≤ 1% 的点数: {np.sum(relative_error <= 1)}")
print(f"相对误差 ≤ 5% 的点数: {np.sum(relative_error <= 5)}")
print(f"相对误差 ≤ 10% 的点数: {np.sum(relative_error <= 10)}")

# ==== 11. 保存预测结果 ====
results = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Density_measured": y_total,
    "Density_predicted": y_pred,
    "Absolute Error": np.abs(y_total - y_pred),
    "Relative Error (%)": relative_error,
    "Cp1_residual": Cp1_residual_expanded,   # 添加残差特征
    "Cp2_residual": Cp2_residual_expanded    # 添加残差特征
})

results.to_excel("Density预测结果_加slope与残差特征_RF.xlsx", index=False)
print("✅ 已保存预测结果为：Density预测结果_加slope与残差特征_RF.xlsx")

# ==== 12. 模型精度对比 ====
r2_298 = r2_score(y_298, HVap_298_all)
mse_298 = mean_squared_error(y_298, HVap_298_all)
r2_Tb = r2_score(y_Tb, HVap_Tb_all)
mse_Tb = mean_squared_error(y_Tb, HVap_Tb_all)

Tb_true = Tb_raw[mask_tb]
Tb_pred = Tb_pred_all[mask_tb]
r2_Tb_pred = r2_score(Tb_true, Tb_pred)
mse_Tb_pred = mean_squared_error(Tb_true, Tb_pred)

print("\n📊 各子模型精度：")
print(f"🔥 Density@298K  — R² = {r2_298:.4f}, MSE = {mse_298:.2f}")
print(f"🔥 Density@Tb    — R² = {r2_Tb:.4f}, MSE = {mse_Tb:.2f}")
print(f"🌡️  Tb预测     — R² = {r2_Tb_pred:.4f}, MSE = {mse_Tb_pred:.2f}")

hvap_compare = pd.DataFrame({
    "Density_298_True": y_298,
    "Density_298_Pred": HVap_298_all,
    "Density_Tb_True": y_Tb,
    "Density_Tb_Pred": HVap_Tb_all,
    "Density_True": Tb_true,
    "Density_Pred": Tb_pred
})

hvap_compare.to_excel("Density_Tb_模型精度对比.xlsx", index=False)
print("✅ 已保存模型精度对比文件为：Density_Tb_模型精度对比.xlsx")
