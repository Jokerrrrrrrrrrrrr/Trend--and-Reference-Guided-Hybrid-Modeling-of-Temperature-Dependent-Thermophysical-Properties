import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==== 1. 读取数据 ====
df = pd.read_excel("pure component isentropic exponent 207.xlsx", sheet_name="Sheet1")

# ==== 2. 定义列 ====
group_cols = df.columns[12:31]   # 第14~25列：基团
temp_cols = df.columns[31:41]    # 第26~35列：温度
v_cols = df.columns[41:51]       # 第36~45列：Hvap

# ==== 3. 准备 slope 所需模型输入 ====
df_298 = pd.read_excel("selected_25_descriptors_normal.xlsx")
X_298 = df_298.drop(columns=["ASPEN isentropic exponent at normal Temperature(bar)"])
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, df_298["ASPEN isentropic exponent at normal Temperature(bar)"])
HVap_298_all = rf_298.predict(X_298)

df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
X_Tb = df_Tb.drop(columns=["ASPEN isentropic exponent at boiling Temperature(bar)"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["ASPEN isentropic exponent at boiling Temperature(bar)"])
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 4. 拟合 Tb 模型 ====
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

Nk_all = df.iloc[:, 12:31].apply(pd.to_numeric, errors='coerce')
Tb_raw = df.iloc[:, 5].values
Tb0 = 222.543
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)

mask_tb = ~np.isnan(Tb_raw)
model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))

# ==== 5. 计算 slope 并加入主 DataFrame ====
T_ref = 298.15
slope_values = (HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)
df["slope"] = slope_values

# ==== 6. 计算残差（Cp1_residual 和 Cp2_residual） ====
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

# ==== 8. 构造训练数据（加上残差特征） ====
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    vols = row[v_cols].values
    slope = row["slope"]
    Cp1_res = Cp1_residual[i]  # 获取对应的Cp1残差
    Cp2_res = Cp2_residual[i]  # 获取对应的Cp2残差

    for T, vol in zip(temps, vols):
        if np.isnan(T) or np.isnan(vol) or np.isnan(slope):
            continue
        # 添加残差特征
        features = np.concatenate([Nk, [T], [slope], [Cp1_res], [Cp2_res]])
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

print("\n📊 模型评估（基团 + 温度 + slope + 残差特征）：")
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
    "Enthalpy_measured": y_total,
    "Enthalpy_predicted": y_pred,
    "Absolute Error": np.abs(y_total - y_pred),
    "Relative Error (%)": relative_error
})
results.to_excel("Enthalpy预测结果_加slope_残差特征_RF.xlsx", index=False)
print("✅ 已保存预测结果为：Enthalpy预测结果_加slope_残差特征_RF.xlsx")

