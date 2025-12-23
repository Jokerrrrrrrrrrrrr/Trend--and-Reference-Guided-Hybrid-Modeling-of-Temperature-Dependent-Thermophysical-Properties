import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

# ==== 1. 读取主数据表（包含基团和物质 ID） ====
df = pd.read_excel("pure component isentropic exponent 207.xlsx", sheet_name="Sheet1")
material_ids = df.iloc[:, 0].values  # 假设第一列是 Material_ID
Nk_all = df.iloc[:, 12:31].apply(pd.to_numeric, errors='coerce')  # 第14~25列为基团

# ==== 2. 读取并训练 HVap_298 模型 ====
df_298 = pd.read_excel("selected_25_descriptors_normal.xlsx")
X_298 = df_298.drop(columns=["ASPEN isentropic exponent at normal Temperature(bar)"])
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["ASPEN isentropic exponent at normal Temperature(bar)"])
HVap_298_all = rf_298.predict(X_298)

# ==== 3. 读取并训练 HVap_Tb 模型 ====
df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
X_Tb = df_Tb.drop(columns=["ASPEN isentropic exponent at boiling Temperature(bar)"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["ASPEN isentropic exponent at boiling Temperature(bar)"])
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 4. 拟合 Tb 模型 ====
Tb_raw = df.iloc[:, 5].values  # 原始 Tb 列
Tb0 = 222.543
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)
mask_tb = ~np.isnan(Tb_raw)

model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))

# ==== 5. 计算 slope ====
T_ref = 298.15
slope_all = (HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)

# ==== 6. 计算残差（HVap_Tb 与实际值的差） ====
Cp1_residual = HVap_Tb_all - df_Tb["ASPEN isentropic exponent at boiling Temperature(bar)"].values
Cp2_residual = HVap_298_all - df_298["ASPEN isentropic exponent at normal Temperature(bar)"].values

# ==== 7. 创建残差 DataFrame ====
residual_df = pd.DataFrame({
    "Material_ID": df.iloc[:, 0].values,  # 假设第一列是 Material_ID
    "Cp1_residual": Cp1_residual,
    "Cp2_residual": Cp2_residual
})

# 保存残差为 Excel 文件
residual_df.to_excel("residual_values.xlsx", index=False)

print("✅ 残差已保存为 residual_values.xlsx")

# ==== 8. 保存 slope 数据 ====
slope_df = pd.DataFrame({
    "Material_ID": material_ids,
    "slope": slope_all
})
slope_df.to_excel("slope_values.xlsx", index=False)

print("✅ slope 已保存为 slope_values.xlsx")
