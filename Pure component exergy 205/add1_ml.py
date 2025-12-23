import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==== 1. 数据加载 ====
df = pd.read_excel("Pure component exergy 205.xlsx", sheet_name="Sheet1")

# 定义列
group_cols = df.columns[12:31]   # 基团列
temp_cols = df.columns[31:41]    # 温度列
v_cols = df.columns[41:51]       # 能量列

# ==== 2. HVap 模型（298 K） ====
df_298 = pd.read_excel("selected_25_descriptors_normal.xlsx")
X_298 = df_298.drop(columns=["ASPEN Exergy at 500k Temperature(j/mol)"])
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["ASPEN Exergy at 500k Temperature(j/mol)"])
HVap_298_all = rf_298.predict(X_298)

# ==== 3. HVap 模型（Tb） ====
df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
X_Tb = df_Tb.drop(columns=["ASPEN Exergy at BoilingTemperature(j/mol)"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["ASPEN Exergy at BoilingTemperature(j/mol)"])
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 4. 拟合 Tb 模型 ====
Tb_raw = df.iloc[:, 5].values  # 原始 Tb 列
Tb0 = 222.543
mask_tb = ~np.isnan(Tb_raw)

poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(df[group_cols].apply(pd.to_numeric, errors='coerce'))
scaler = StandardScaler()
Nk_scaled = scaler.fit_transform(Nk_poly)

# 使用标准化后的特征进行拟合
model_Tb = HuberRegressor(max_iter=10000)  # 默认优化器 lbfgs，稳定收敛
model_Tb.fit(Nk_scaled[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_scaled), 1e-6, None))

# ==== 5. 计算 slope 并加入主 DataFrame ====
T_ref = 500
slope_values = (HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)
df["slope"] = slope_values

# ==== 6. 计算残差 ====
Cp1_residual = HVap_Tb_all - df_Tb["ASPEN Exergy at BoilingTemperature(j/mol)"].values  # Tb模型残差
Cp2_residual = HVap_298_all - df_298["ASPEN Exergy at 500k Temperature(j/mol)"].values  # 298K模型残差

# ==== 7. 扩展残差数据，确保每个物质对应 10 行 ====
Cp1_residual_expanded = Cp1_residual.repeat(10)  # 每个残差扩展 10 行
Cp2_residual_expanded = Cp2_residual.repeat(10)  # 每个残差扩展 10 行

# ==== 8. 构造训练集 ====
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    vols = row[v_cols].values
    slope = row["slope"]

    for T, vol in zip(temps, vols):
        if np.isnan(T) or np.isnan(vol) or np.isnan(slope):
            continue
        features = np.concatenate([Nk, [T], [slope]])
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
ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100  # ARD %

print("\n📊 模型评估：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")

relative_error = np.abs((y_pred - y_total) / y_total) * 100
print(f"相对误差 ≤ 1% 的点数: {np.sum(relative_error <= 1)}")
print(f"相对误差 ≤ 5% 的点数: {np.sum(relative_error <= 5)}")
print(f"相对误差 ≤ 10% 的点数: {np.sum(relative_error <= 10)}")

# ==== 11. 保存预测结果和残差 ====
results = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Exergy_measured": y_total,
    "Exergy_predicted": y_pred,
    "Cp1_residual": Cp1_residual_expanded,  # 添加Cp1残差
    "Cp2_residual": Cp2_residual_expanded,  # 添加Cp2残差
    "Absolute Error": np.abs(y_total - y_pred),
    "Relative Error (%)": relative_error
})

# 保存为 Excel 文件
results.to_excel("Exergy预测结果_加slope与残差特征_RF.xlsx", index=False)
print("✅ 已保存预测结果为：Exergy预测结果_加slope与残差特征_RF.xlsx")
