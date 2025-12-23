import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ========================
# 1) 读取数据
# ========================
df = pd.read_excel("volume208.xlsx", sheet_name="Sheet5")

# slope：从 /mnt/data/slope_values.xlsx 的 Sheet2 读取
# 假设一列为 slope 值（列名可能是 'slope' 或 'slope_value'），行数与 df 对齐（每行对应一个物质）
df_slope = pd.read_excel("slope_values.xlsx", sheet_name="Sheet2")

# 尝试常见列名
slope_col_candidates = [c for c in df_slope.columns if str(c).strip().lower() in ("slope", "slope_value", "slopeval", "slp")]
if not slope_col_candidates:
    # 如果没有常见列名，默认取第一列
    slope_col = df_slope.columns[0]
else:
    slope_col = slope_col_candidates[0]

slope = df_slope[slope_col].to_numpy().reshape(-1)
if len(slope) != len(df):
    raise ValueError(f"slope 行数({len(slope)})与物质数量({len(df)})不一致，请检查 slope 表。")

# ========================
# 2) 原显示模型：按你原代码（注意统一温度切片）
#    - 温度：AF:AO -> iloc[32:42]
#    - 原始密度：AP:AY -> iloc[:, 42:52]
# ========================
def calculate_base_model(row):
    omega = row.iloc[54]  # BF 列 (ω)
    Tc = row.iloc[6]      # G  列 (Tc, K)
    Pc = row.iloc[55]     # BG 列 (Pc, bar) — 请确认单位

    temps = row.iloc[32:42].values  # AF 到 AO：10 个温度点
    base_vals = []

    for T in temps:
        Zra = 0.29056 - 0.08775 * omega
        temp_value = 1 + (1 - T / Tc) ** 0.285714  # 2/7 ≈ 0.285714
        # 83.14 单位为 cm3·bar/(mol·K)，此表达式更像摩尔体积而非密度
        val = (83.14 * Tc * (Zra ** temp_value)) / Pc
        base_vals.append(val)
    return base_vals

base_model = np.array([calculate_base_model(row) for _, row in df.iterrows()])  # (n,10)

# 原始“密度”（AP:AY）
y_exp = df.iloc[:, 42:52].to_numpy()  # (n,10)

# 温度矩阵（与 base_model 使用的温度保持一致）
T = df.iloc[:, 32:42].to_numpy()      # (n,10)

# ========================
# 3) 构建设计矩阵并回归
#     y ≈ base + w1 * slope * T + intercept
#     => (y - base) ≈ w1 * (slope*T) + intercept
# ========================
n, m = T.shape  # m=10
base_flat = base_model.ravel()
y_flat = y_exp.ravel()
T_flat = T.ravel()
slope_flat = np.repeat(slope, m)   # 每个物质重复 10 次

# 特征：slope * T
X_feature = (slope_flat * T_flat).reshape(-1, 1)
target = (y_flat - base_flat)

reg = LinearRegression(fit_intercept=True)
reg.fit(X_feature, target)

w1 = reg.coef_[0]
intercept = reg.intercept_

# 预测（优化后）
y_pred_before = base_flat
y_pred_after = base_flat + (w1 * slope_flat * T_flat + intercept)

# ========================
# 4) 评估
# ========================
def metrics(y_true, y_pred):
    abs_err = np.abs(y_pred - y_true)
    # 相对误差：避免被 0 除
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = 100 * abs_err / y_true
        rel_err = np.where(np.isfinite(rel_err), rel_err, np.nan)
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "ARD": np.nanmean(rel_err)
    }

m_before = metrics(y_flat, y_pred_before)
m_after  = metrics(y_flat, y_pred_after)

print("🔧 回归得到的参数：")
print(f"  intercept = {intercept:.8f}")
print(f"  w1        = {w1:.8f}")

print("\n📊 优化前/后对比：")
print(f"  优化前  R²  = {m_before['R2']:.4f}, MSE = {m_before['MSE']:.6f}, ARD = {m_before['ARD']:.2f}%")
print(f"  优化后  R²  = {m_after['R2']:.4f}, MSE = {m_after['MSE']:.6f}, ARD = {m_after['ARD']:.2f}%")

# ========================
# 5) 导出详细结果
# ========================
results_df = pd.DataFrame({
    "Material_ID": np.repeat(np.arange(n), m),
    "Temperature (K)": T_flat,
    "Original_Value": y_flat,                 # 你表中 AP:AY 的“原始密度”
    "Base_Model": y_pred_before,
    "Correction": (w1 * slope_flat * T_flat + intercept),
    "After_Regression": y_pred_after,
    "Abs_Error_Before": np.abs(y_pred_before - y_flat),
    "Abs_Error_After":  np.abs(y_pred_after  - y_flat)
})

# 相对误差列（安全处理）
with np.errstate(divide='ignore', invalid='ignore'):
    results_df["Rel_Error_Before (%)"] = 100 * results_df["Abs_Error_Before"] / results_df["Original_Value"]
    results_df["Rel_Error_After (%)"]  = 100 * results_df["Abs_Error_After"]  / results_df["Original_Value"]

results_df.to_excel("density_with_slope_regression.xlsx", index=False)
print("✅ 已保存: density_with_slope_regression.xlsx")
