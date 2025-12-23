import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
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

# ==== 5. 定义指数函数并进行拟合 ====

# 指数函数形式: A * exp(B * T)
def exp_func(T, A, B):
    return A * np.exp(B * T)


# 结果保存列表
A_all, B_all, exp_B_all = [], [], []

# 针对每个物质拟合指数函数
for i in range(len(material_ids)):
    # 对每个物质，使用 HVap_298_all[i] 和 HVap_Tb_all[i] 进行拟合
    T_vals = [298.15, Tb_pred_all[i]]  # 选择 (298.15, HVap_298_all[i]) 和 (Tb_pred_all[i], HVap_Tb_all[i]) 作为拟合点
    HVap_vals = [HVap_298_all[i], HVap_Tb_all[i]]  # 对应的 y 值是 HVap_298_all[i] 和 HVap_Tb_all[i]]

    try:
        # 设置初始猜测值
        initial_guess = [HVap_vals[0], 0.1]  # 假设 A 初始为 HVap_vals[0]，B 初始为 0.1

        # 使用 curve_fit 进行拟合
        params, covariance = curve_fit(exp_func, T_vals, HVap_vals, p0=initial_guess, maxfev=10000)

        # 提取拟合的参数 A 和 B
        A, B = params
        A_all.append(A)
        B_all.append(B)

        # 计算 exp(B) 来保存
        exp_B_all.append(np.exp(B))  # 计算 exp(B)

    except Exception as e:
        print(f"物质 {material_ids[i]} 拟合失败: {e}")
        A_all.append(np.nan)
        B_all.append(np.nan)
        exp_B_all.append(np.nan)

# ==== 6. 保存拟合结果为 Excel 文件 ====
# 将拟合结果（A 和 exp(B)）存入 DataFrame
slope_df = pd.DataFrame({
    "Index": range(1, len(material_ids) + 1),  # 添加序号列
    "A": A_all,
    "exp_B": exp_B_all  # 使用 exp(B) 作为结果
})

slope_df.to_excel("slope_values_exp.xlsx", index=False)

print("✅ 每个物质的 A 和 exp(B) 已保存为 slope_values_exp.xlsx")
