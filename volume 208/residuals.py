import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

# ==== 1. 读取主数据表（包含基团和物质 ID） ====
df = pd.read_excel("volume208.xlsx", sheet_name="Sheet1")
material_ids = df.iloc[:, 0].values  # 假设第一列是 Material_ID
Nk_all = df.iloc[:, 13:32].apply(pd.to_numeric, errors='coerce')  # 第14~25列为基团

# ==== 2. 读取并训练 HVap_298 模型 ====
df_298 = pd.read_excel("selected_25_descriptors_normal.xlsx")
X_298 = df_298.drop(columns=["volume at normal temperature"])
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["volume at normal temperature"])
HVap_298_all = rf_298.predict(X_298)

# ==== 3. 读取并训练 HVap_Tb 模型 ====
df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
X_Tb = df_Tb.drop(columns=["volume at boiling temperature"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["volume at boiling temperature"])
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 4. 拟合 Tb 模型 ====
Tb_raw = df.iloc[:, 5].values  # 原始 Tb 列
Tb0 = 222.543
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)
mask_tb = ~np.isnan(Tb_raw)

model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))

# ==== 5. 计算残差 ====
# 计算 Cp1 残差（HVap_Tb_all - 真实的 HVap_Tb）
Cp1_residual = HVap_Tb_all - df_Tb["volume at boiling temperature"].values

# 计算 Cp2 残差（HVap_298_all - 真实的 HVap_298）
Cp2_residual = HVap_298_all - df_298["volume at normal temperature"].values

# ==== 6. 扩展残差数据，确保每个物质对应 10 行 ====
Cp1_residual_expanded = Cp1_residual.repeat(10)  # 每个残差扩展 10 行
Cp2_residual_expanded = Cp2_residual.repeat(10)  # 每个残差扩展 10 行

# ==== 7. 保存为 CSV ====
residuals_df = pd.DataFrame({
    "Material_ID": np.tile(material_ids, 10),  # 复制每个物质的 ID 10 次
    "Cp1_residual": Cp1_residual_expanded,    # 扩展后的 Cp1 残差
    "Cp2_residual": Cp2_residual_expanded     # 扩展后的 Cp2 残差
})

residuals_df.to_csv("residuals_values.csv", index=False)

print("✅ 残差已保存为 residuals_values.csv")
