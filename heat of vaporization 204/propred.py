import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_hvap(T, Tc, omega):
    """
    Reid et al. (1987) 方法计算汽化焓 Hv (kJ/mol)
    """
    tr = 1.0 - (T / Tc)

    # 检查有效范围
    # if not np.all((tr >= 0.2) & (tr <= 0.9)):
    #     return np.nan  # 超出范围的点返回 NaN

    W = (omega - 0.21) / 0.25

    tr_p033 = tr ** 0.333
    tr_p083 = tr ** 0.833
    tr_p1208 = tr ** 1.208

    R1 = (6.537 * tr_p033
          - 2.467 * tr_p083
          - 77.521 * tr_p1208
          + 59.634 * tr
          + 36.009 * tr**2
          - 14.606 * tr**3)

    R2 = (-0.133 * tr_p033
          - 28.215 * tr_p083
          - 82.958 * tr_p1208
          + 99.000 * tr
          + 19.105 * tr**2
          - 2.796 * tr**3)

    Hv = (R1 + W * R2) * Tc * 0.008314  # kJ/mol
    return Hv

# 读取 Excel（Sheet6）
df = pd.read_excel("heat of vaporization 204.xlsx", sheet_name="Sheet6")

# 获取需要的列（使用列号）
Tc = df.iloc[:, 6].to_numpy()  # G 列（列号6）
omega = df.iloc[:, 54].to_numpy()  # BC 列（列号55）
T_values = df.iloc[:, 32:42].to_numpy()  # AG 到 AP 列（列号32到41），每行10个温度点

# 获取实际的汽化焓值（AQ 到 BB 列，列号 43 到 54）
Hv_actual = df.iloc[:, 42:52].to_numpy()

# 结果数组
Hv_results = np.zeros_like(T_values, dtype=float)

# 循环计算 Hv
for i in range(len(df)):
    for j in range(T_values.shape[1]):
        Hv_results[i, j] = calculate_hvap(T_values[i, j], Tc[i], omega[i])
Hv_results=Hv_results*1000
# 计算模型输出值与实际值的对比
# 评估模型精度：R², MSE, ARD
r2 = r2_score(Hv_actual.flatten(), Hv_results.flatten())
mse = mean_squared_error(Hv_actual.flatten(), Hv_results.flatten())
ard = np.mean(np.abs((Hv_results.flatten() - Hv_actual.flatten()) / Hv_actual.flatten())) * 100  # 平均相对误差（百分比）

print("📊 模型评估结果：")
print(f"R²  = {r2:.4f}")
print(f"MSE = {mse:.4f}")
print(f"ARD = {ard:.2f}%")

# 生成对比表并保存为 Excel
Hv_columns = [f"Hv_{j+1}" for j in range(Hv_results.shape[1])]
Hv_df = pd.DataFrame(Hv_results, columns=Hv_columns)

# 合并实际值和模型计算的值
Hv_actual_df = pd.DataFrame(Hv_actual, columns=[f"Actual_Hv_{j+1}" for j in range(Hv_actual.shape[1])])

# 合并数据框
df_with_Hv_comparison = pd.concat([df, Hv_df, Hv_actual_df], axis=1)

# 保存结果
df_with_Hv_comparison.to_excel("Hv_comparison_results_Sheet6.xlsx", index=False)
print("✅ 已保存: Hv_comparison_results_Sheet6.xlsx")
