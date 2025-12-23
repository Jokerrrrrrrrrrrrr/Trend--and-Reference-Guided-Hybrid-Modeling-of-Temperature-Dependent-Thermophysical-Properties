import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==== 1. 数据加载 ====
df = pd.read_excel("Gibbs free energy 205.xlsx", sheet_name="Sheet6")

group_cols = df.columns[12:31]  # 第13~31列：基团
temp_cols = df.columns[31:41]   # 第32~41列：温度
v_cols = df.columns[41:51]      # 第42~51列：目标变量（吉布斯自由能）

# ==== 2. 数据预处理 ====
# 确保数值列正确转换
for col in temp_cols.tolist() + v_cols.tolist():
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 基团数据
Nk_all = df[group_cols].apply(pd.to_numeric, errors='coerce')

# ==== 3. Hvap 模型（Tb） ====
df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
X_Tb = df_Tb.drop(columns=["ASPEN Vapor pressure at BoilingTemperature(bar)"])
y_Tb = df_Tb["ASPEN Vapor pressure at BoilingTemperature(bar)"]
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, y_Tb)
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 4. Tb 模型预测（标准化 + 多项式）====
Tb_raw = df.iloc[:, 5].values
Tb0 = 222.543

poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)
scaler = StandardScaler()
Nk_scaled = scaler.fit_transform(Nk_poly)

mask_tb = ~np.isnan(Tb_raw)
model_Tb = HuberRegressor(max_iter=10000)
model_Tb.fit(Nk_scaled[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_scaled), 1e-6, None))

# ==== 5. A_k 系数训练 ====
# 使用原始19个基团
G = Nk_all.values  # (n, 19) 原始基团数据
X_rows, y_rows = [], []
temp_eval = []  # 保存温度点信息用于评估

# 构建训练集
for i in range(len(df)):  # 遍历所有物质
    for j, (tcol, vcol) in enumerate(zip(temp_cols, v_cols)):
        Tj = df.at[i, tcol]  # 温度值
        Vj = df.at[i, vcol]  # 吉布斯自由能值

        # 跳过NaN值
        if np.isnan(Tj) or np.isnan(Vj):
            continue

        Tb_i = Tb_pred_all[i]  # 物质i的参考温度
        HVap_Tb_i = HVap_Tb_all[i]  # 物质i的参考吉布斯自由能值

        # 特征：(T - T_ref) × G (使用原始19个基团)
        Xj = (Tj - Tb_i) * G[i]  # 形状: (19,)

        # 目标：V - V_ref
        yj = Vj - HVap_Tb_i

        X_rows.append(Xj)
        y_rows.append(yj)
        temp_eval.append((tcol, vcol, i, j))

X_A = np.array(X_rows)  # (n_samples, 19)
y_A = np.array(y_rows)  # (n_samples,)

# 训练 A_k 系数模型
A_solver = HuberRegressor(fit_intercept=False, max_iter=5000)
A_solver.fit(X_A, y_A)
A_vec = A_solver.coef_  # 长度19，对应19个基团

# ==== 6. 生成吉布斯自由能预测 ====
V_pred = pd.DataFrame(index=df.index, columns=v_cols, dtype=float)

for i in range(len(df)):  # 遍历所有物质
    Tb_i = Tb_pred_all[i]  # 物质i的参考温度
    HVap_Tb_i = HVap_Tb_all[i]  # 物质i的参考吉布斯自由能值

    for j, (tcol, vcol) in enumerate(zip(temp_cols, v_cols)):
        Tj = df.at[i, tcol]  # 温度值

        if np.isnan(Tj):
            V_pred.at[i, vcol] = np.nan
            continue

        # 特征：(T - T_ref) × G (使用原始19个基团)
        Xj = (Tj - Tb_i) * G[i]

        # 预测：V_ref + A_k × (T - T_ref) × G
        V_pred_j = HVap_Tb_i + Xj @ A_vec
        V_pred.at[i, vcol] = V_pred_j

# ==== 7. 评估模型性能 ====
y_true_all, y_pred_all = [], []
for vcol in v_cols:
    m = (~df[vcol].isna()) & (~V_pred[vcol].isna())
    if m.any():
        y_true_all.append(df.loc[m, vcol].to_numpy())
        y_pred_all.append(V_pred.loc[m, vcol].to_numpy())

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

mse_v = mean_squared_error(y_true_all, y_pred_all)
r2_v = r2_score(y_true_all, y_pred_all)
print(f"MSE (on Gibbs Free Energy): {mse_v:.6f}")
print(f"R2  (on Gibbs Free Energy): {r2_v:.6f}")

# ==== 8. 分温度点评估 ====
print("\n分温度点评估:")
for tcol, vcol in zip(temp_cols, v_cols):
    m = (~df[tcol].isna()) & (~df[vcol].isna()) & (~V_pred[vcol].isna())
    if m.any():
        v_true = df.loc[m, vcol].to_numpy()
        v_pred = V_pred.loc[m, vcol].to_numpy()
        mse_temp = mean_squared_error(v_true, v_pred)
        r2_temp = r2_score(v_true, v_pred)
        print(f"  {tcol}: MSE = {mse_temp:.6f}, R2 = {r2_temp:.6f}")

# ==== 9. 保存结果 ====
id_col = df.columns[0]  # 物质ID/名称所在列
out_path = "gibbs_free_energy_actual_vs_pred_long.xlsx"

rows = []
for idx, _ in df.iterrows():
    ID = df.at[idx, id_col]
    for j, (tcol, vcol) in enumerate(zip(temp_cols, v_cols), start=1):
        T = df.at[idx, tcol]
        V_act = df.at[idx, vcol]
        V_prd = V_pred.at[idx, vcol]
        err = (V_prd - V_act) if (pd.notna(V_prd) and pd.notna(V_act)) else np.nan

        rows.append({
            id_col: ID,
            "temp_index": j,
            "temp_col": tcol,
            "T": T,
            "Gibbs_Free_Energy_actual": V_act,
            "Gibbs_Free_Energy_pred": V_prd,
            "error = pred - actual": err,
            "T_ref": Tb_pred_all[idx],
            "Gibbs_Free_Energy_ref": HVap_Tb_all[idx]
        })

long_compare = pd.DataFrame(rows).sort_values([id_col, "temp_index"])

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    long_compare.to_excel(writer, sheet_name="compare_long", index=False)

print(f"\n✅ 结果已保存到: {out_path}")