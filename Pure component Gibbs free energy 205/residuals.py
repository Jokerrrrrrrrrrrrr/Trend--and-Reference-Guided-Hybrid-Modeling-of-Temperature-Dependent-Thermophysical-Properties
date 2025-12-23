import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ==== 1. 读取主数据 ====
df = pd.read_excel("Gibbs free energy 205.xlsx", sheet_name="Sheet1")
material_ids = df.iloc[:, 0].values
Nk_all = df.iloc[:, 12:31].apply(pd.to_numeric, errors='coerce')  # 第13到31列为基团

# ==== 2. HVap_298 模型 ====
df_298 = pd.read_excel("selected_25_descriptors_normal.xlsx")
X_298 = df_298.drop(columns=["ASPEN Vapor pressure at Normal Temperature(bar)"])
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["ASPEN Vapor pressure at Normal Temperature(bar)"])
HVap_298_all = rf_298.predict(X_298)

# ==== 3. HVap_Tb 模型 ====
df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
X_Tb = df_Tb.drop(columns=["ASPEN Vapor pressure at BoilingTemperature(bar)"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["ASPEN Vapor pressure at BoilingTemperature(bar)"])
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
# 残差1：HVap_Tb 模型的实际值与预测值之间的差
Cp1_residual = HVap_Tb_all - df_Tb["ASPEN Vapor pressure at BoilingTemperature(bar)"].values

# 残差2：HVap_298 模型的实际值与预测值之间的差
Cp2_residual = HVap_298_all - df_298["ASPEN Vapor pressure at Normal Temperature(bar)"].values

# ==== 7. 创建残差 DataFrame ====
residual_df = pd.DataFrame({
    "Material_ID": material_ids,
    "Cp1_residual": Cp1_residual,
    "Cp2_residual": Cp2_residual
})

# 保存残差为 Excel 文件
residual_df.to_excel("residual_values.xlsx", index=False)

print("✅ 残差已保存为 residual_values.xlsx")

# ==== 8. 保存 slope ====
slope_df = pd.DataFrame({
    "Material_ID": material_ids,
    "slope": slope_all
})
slope_df.to_excel("slope_values.xlsx", index=False)

print("✅ slope 已保存为 slope_values.xlsx")
