import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

# ==== 1. 读取数据 ====
df = pd.read_excel("heat of vaporization 204.xlsx", sheet_name="Sheet1")

# ==== 2. 定义列 ====
group_cols = df.columns[13:32]   # 基团
temp_cols  = df.columns[32:42]   # 温度
hvap_cols  = df.columns[42:52]   # Hvap

# ==== 3. 准备 slope 所需模型输入 ====
# 3.1 298 K 子模型（RF）
df_298 = pd.read_excel("selected_25_descriptors_data_298.xlsx")
y_298  = df_298["Heat of vaporization at normal temperature"].to_numpy()
X_298  = df_298.drop(columns=["Heat of vaporization at normal temperature"])
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, y_298)
HVap_298_all = rf_298.predict(X_298)

# 3.2 沸点温度子模型（RF）
df_Tb = pd.read_excel("selected_25_descriptors_data_boiling_point.xlsx")
y_TbHvap = df_Tb["Heat of vaporization at boiling temperature"].to_numpy()
X_Tb     = df_Tb.drop(columns=["Heat of vaporization at boiling temperature"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, y_TbHvap)
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 4. Tb 子模型预测 ====
Nk_all = df.iloc[:, 13:32].apply(pd.to_numeric, errors='coerce')
Tb_raw = df.iloc[:, 5].values  # 实际 Tb（K）
Tb0 = 222.543
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)

mask_tb = ~np.isnan(Tb_raw)
model_Tb = HuberRegressor(max_iter=5000).fit(Nk_poly[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))  # 还原到 K

# ==== 4b. 子模型评估（新增）====
# RF@298 K
r2_298  = r2_score(y_298, HVap_298_all)
mse_298 = mean_squared_error(y_298, HVap_298_all)

# RF@Tb
r2_TbHvap  = r2_score(y_TbHvap, HVap_Tb_all)
mse_TbHvap = mean_squared_error(y_TbHvap, HVap_Tb_all)

# Tb（在 K 空间评估）
r2_Tb  = r2_score(Tb_raw[mask_tb], Tb_pred_all[mask_tb])
mse_Tb = mean_squared_error(Tb_raw[mask_tb], Tb_pred_all[mask_tb])

print("📌 子模型评估：")
print(f"  RF@298K    -> R²={r2_298:.4f},  MSE={mse_298:.3f}")
print(f"  RF@Tb      -> R²={r2_TbHvap:.4f}, MSE={mse_TbHvap:.3f}")
print(f"  Tb model   -> R²={r2_Tb:.4f},  MSE={mse_Tb:.3f}")

# ==== 5. 计算 slope 并加入主 DataFrame ====
T_ref = 298.15
slope_values = (HVap_Tb_all - HVap_298_all) / (Tb_pred_all - T_ref)
df["slope"] = slope_values

# ==== 6. 构建训练数据 ====
X_total, y_total, material_ids, temperatures = [], [], [], []

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values
    temps = row[temp_cols].values
    hvaps = row[hvap_cols].values
    slope = row["slope"]

    for T, Hv in zip(temps, hvaps):
        if np.isnan(T) or np.isnan(Hv) or np.isnan(slope):
            continue
        features = np.concatenate([Nk, [T], [slope]])
        X_total.append(features)
        y_total.append(Hv)
        material_ids.append(material_id)
        temperatures.append(T)

X_total = np.array(X_total)
y_total = np.array(y_total)

# ==== 7. 拟合主模型 ====
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_total, y_total)

# ==== 8. 主模型评估 ====
y_pred = model.predict(X_total)
r2 = r2_score(y_total, y_pred)
mse = mean_squared_error(y_total, y_pred)
ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100  # ARD %

print("\n📊 主模型评估（基团 + 温度 + slope 特征）：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.2f}")
print(f"ARD = {ard:.2f}%")

# 误差阈值统计
relative_error = np.abs((y_pred - y_total) / y_total) * 100
within_1pct  = np.sum(relative_error <= 1)
within_5pct  = np.sum(relative_error <= 5)
within_10pct = np.sum(relative_error <= 10)
print(f"相对误差 ≤ 1% 的点数: {within_1pct}")
print(f"相对误差 ≤ 5% 的点数: {within_5pct}")
print(f"相对误差 ≤ 10% 的点数: {within_10pct}")

# ==== 9. 保存结果 ====
results = pd.DataFrame({
    "Material_ID": material_ids,
    "Temperature (K)": temperatures,
    "Hvap_measured": y_total,
    "Hvap_predicted": y_pred,
    "Absolute Error": np.abs(y_total - y_pred),
    "Relative Error (%)": 100 * np.abs((y_total - y_pred) / y_total)
})
results.to_excel("Hvap预测结果_加slope特征_RF.xlsx", index=False)
print("✅ 已保存预测结果为: Hvap预测结果_加slope特征_RF.xlsx")
