import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ==== 1. 读取主数据 ====
df = pd.read_excel("liquid density.xlsx", sheet_name="Sheet1")
material_ids = df.iloc[:, 0].values
Nk_all = df.iloc[:, 12:31].apply(pd.to_numeric, errors='coerce')

# ==== 2. HVap_298 模型 ====
df_298 = pd.read_excel("selected_25_descriptors_normal.xlsx")
X_298 = df_298.drop(columns=["ASPEN Liquid Density at Normal Temperature(g/cc)"])
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["ASPEN Liquid Density at Normal Temperature(g/cc)"])
HVap_298_all = rf_298.predict(X_298)

# ==== 3. HVap_Tb 模型 ====
df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
X_Tb = df_Tb.drop(columns=["ASPEN Liquid Density at BoilingTemperature(g/cc)"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["ASPEN Liquid Density at BoilingTemperature(g/cc)"])
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 4. 拟合 Tb 模型 ====
Tb_raw = df.iloc[:, 5].values  # 原始 Tb 列
Tb0 = 222.543
mask_tb = ~np.isnan(Tb_raw)

poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)
scaler = StandardScaler()
Nk_scaled = scaler.fit_transform(Nk_poly)

# 使用标准化后的特征进行拟合
model_Tb = HuberRegressor(max_iter=10000)  # 默认优化器 lbfgs，稳定收敛
model_Tb.fit(Nk_scaled[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_scaled), 1e-6, None))

# ==== 5. 计算 slope ====
T_ref = 298.15
slope_all = (HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)

# ==== 6. 计算残差 ====
Cp1_residual = HVap_Tb_all - df_Tb["ASPEN Liquid Density at BoilingTemperature(g/cc)"].values
Cp2_residual = HVap_298_all - df_298["ASPEN Liquid Density at Normal Temperature(g/cc)"].values

# ==== 7. 扩展残差数据，确保每个物质对应 10 行 ====
Cp1_residual_expanded = Cp1_residual.repeat(10)  # 每个残差扩展 10 行
Cp2_residual_expanded = Cp2_residual.repeat(10)  # 每个残差扩展 10 行

# ==== 8. 保存为 CSV ====
residuals_df = pd.DataFrame({
    "Material_ID": np.tile(material_ids, 10),  # 复制每个物质的 ID 10 次
    "Cp1_residual": Cp1_residual_expanded,    # 扩展后的 Cp1 残差
    "Cp2_residual": Cp2_residual_expanded     # 扩展后的 Cp2 残差
})

residuals_df.to_csv("residuals_values.csv", index=False)

print("✅ 残差已保存为 residuals_values.csv")
