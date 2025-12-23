import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
df = pd.read_excel("liquid density.xlsx")


# 计算液体密度（Modified Rackett correlation）
def calculate_density(row):
    omega = row.iloc[57]  # BF 列 (ω)
    Tc = row.iloc[6]  # G 列 (Tc)
    Pc = row.iloc[58]  # BG 列 (Pc, bar)

    temps = row.iloc[31:41].values  # AF到AO列 (温度数据)
    densities = []

    for T in temps:
        Zra = 0.29056 - 0.08775 * omega
        temp_value = 1 + (1 - T / Tc) ** 0.285714
        dens = (83.14 * Tc * (Zra ** temp_value)) / Pc
        densities.append(dens)
    return densities


# 逐行计算模型密度
calculated_densities = [calculate_density(row) for _, row in df.iterrows()]

# 原始密度
original_densities = df.iloc[:, 41:51].values  # AP到AY列

# 展开成一维
calculated_densities_flat = np.array(calculated_densities).flatten()
original_densities_flat = original_densities.flatten()

# 计算评估指标
absolute_error = np.abs(calculated_densities_flat - original_densities_flat)
relative_error = 100 * absolute_error / original_densities_flat
r2 = r2_score(original_densities_flat, calculated_densities_flat)
mse = mean_squared_error(original_densities_flat, calculated_densities_flat)
ard = np.mean(relative_error)

# 输出结果
print("\n📊 模型评估（计算密度 vs 原始密度）：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.4f}")
print(f"ARD = {ard:.2f}%")

# 保存对比结果
results_df = pd.DataFrame({
    "Temperature (K)": np.tile(df.iloc[:, 31:41].values, (1, 1)).flatten(),
    "Original_Density": original_densities_flat,
    "Calculated_Density": calculated_densities_flat,
    "Absolute_Error": absolute_error,
    "Relative_Error (%)": relative_error
})
results_df.to_excel("density_comparison_results.xlsx", index=False)
print("✅ 已保存: density_comparison_results.xlsx")
