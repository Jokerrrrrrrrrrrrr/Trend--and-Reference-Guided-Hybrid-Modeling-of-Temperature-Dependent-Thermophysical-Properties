import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# ==== 1. 读取主数据表（包含基团和物质 ID） ====
df = pd.read_excel("internal energy 207.xlsx", sheet_name="Sheet1")
material_ids = df.iloc[:, 0].values  # 假设第一列是 Material_ID
Nk_all = df.iloc[:, 13:32].apply(pd.to_numeric, errors='coerce')  # 第14~32列为基团

# ==== 2. 定义列索引 ====
group_cols = df.columns[13:32]  # 第14~32列：基团
temp_cols = df.columns[32:42]   # 第33~42列：温度
hvap_cols = df.columns[42:52]   # 第43~52列：目标变量 Hvap

# ==== 3. 读取并训练 HVap_Tb 模型 ====
df_Tb = pd.read_excel("selected_25_descriptors_boiling.xlsx")
X_Tb = df_Tb.drop(columns=["internal energy at boiling temperature"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["internal energy at boiling temperature"])
HVap_Tb_all = rf_Tb.predict(X_Tb)

# ==== 4. 拟合 Tb 模型 ====
Tb_raw = df.iloc[:, 5].values  # 原始 Tb 列
Tb0 = 222.543
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)
mask_tb = ~np.isnan(Tb_raw)

model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))

# ==== 5. A_k 系数训练 ====
# 使用原始19个基团（不是扩展后的）
G = Nk_all.values  # (n, 19) 原始基团数据
X_rows, y_rows = [], []
temp_eval = []  # 保存温度点信息用于评估

# 构建训练集
for i in range(len(df)):  # 遍历所有物质
    for j, (tcol, hvcol) in enumerate(zip(temp_cols, hvap_cols)):
        Tj = df.at[i, tcol]  # 温度值
        Hvapj = df.at[i, hvcol]  # 目标变量值

        # 跳过NaN值
        if np.isnan(Tj) or np.isnan(Hvapj):
            continue

        Tb_i = Tb_pred_all[i]  # 物质i的参考温度
        HVap_Tb_i = HVap_Tb_all[i]  # 物质i的参考目标变量值

        # 特征：(T - T_ref) × G (使用原始19个基团)
        Xj = (Tj - Tb_i) * G[i]  # 形状: (19,)

        # 目标：Hvap - Hvap_ref
        yj = Hvapj - HVap_Tb_i

        X_rows.append(Xj)
        y_rows.append(yj)
        temp_eval.append((tcol, hvcol, i, j))

X_A = np.array(X_rows)  # (n_samples, 19)
y_A = np.array(y_rows)  # (n_samples,)

# 训练 A_k 系数模型
A_solver = HuberRegressor(fit_intercept=False, max_iter=5000)
A_solver.fit(X_A, y_A)
A_vec = A_solver.coef_  # 长度19，对应19个基团

# ==== 6. 生成预测 ====
HVap_pred = pd.DataFrame(index=df.index, columns=hvap_cols, dtype=float)

for i in range(len(df)):  # 遍历所有物质
    Tb_i = Tb_pred_all[i]  # 物质i的参考温度
    HVap_Tb_i = HVap_Tb_all[i]  # 物质i的参考目标变量值

    for j, (tcol, hvcol) in enumerate(zip(temp_cols, hvap_cols)):
        Tj = df.at[i, tcol]  # 温度值

        if np.isnan(Tj):
            HVap_pred.at[i, hvcol] = np.nan
            continue

        # 特征：(T - T_ref) × G (使用原始19个基团)
        Xj = (Tj - Tb_i) * G[i]

        # 预测：Hvap_ref + A_k × (T - T_ref) × G
        HVap_pred_j = HVap_Tb_i + Xj @ A_vec
        HVap_pred.at[i, hvcol] = HVap_pred_j

# ==== 7. 评估模型性能 ====
y_true_all, y_pred_all = [], []
for hvcol in hvap_cols:
    m = (~df[hvcol].isna()) & (~HVap_pred[hvcol].isna())
    if m.any():
        y_true_all.append(df.loc[m, hvcol].to_numpy())
        y_pred_all.append(HVap_pred.loc[m, hvcol].to_numpy())

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

mse_hvap = mean_squared_error(y_true_all, y_pred_all)
r2_hvap = r2_score(y_true_all, y_pred_all)
print(f"MSE (on HVap): {mse_hvap:.6f}")
print(f"R2  (on HVap): {r2_hvap:.6f}")

# ==== 8. 分温度点评估 ====
print("\n分温度点评估:")
for tcol, hvcol in zip(temp_cols, hvap_cols):
    m = (~df[tcol].isna()) & (~df[hvcol].isna()) & (~HVap_pred[hvcol].isna())
    if m.any():
        hvap_true = df.loc[m, hvcol].to_numpy()
        hvap_pred = HVap_pred.loc[m, hvcol].to_numpy()
        mse_temp = mean_squared_error(hvap_true, hvap_pred)
        r2_temp = r2_score(hvap_true, hvap_pred)
        print(f"  {tcol}: MSE = {mse_temp:.6f}, R2 = {r2_temp:.6f}")

# ==== 9. 保存结果 ====
id_col = df.columns[0]  # 物质ID/名称所在列
out_path = "internal_energy_actual_vs_pred_long.xlsx"

rows = []
for idx, _ in df.iterrows():
    ID = df.at[idx, id_col]
    for j, (tcol, hvcol) in enumerate(zip(temp_cols, hvap_cols), start=1):
        T = df.at[idx, tcol]
        HVap_act = df.at[idx, hvcol]
        HVap_prd = HVap_pred.at[idx, hvcol]
        err = (HVap_prd - HVap_act) if (pd.notna(HVap_prd) and pd.notna(HVap_act)) else np.nan

        rows.append({
            id_col: ID,
            "temp_index": j,
            "temp_col": tcol,
            "T": T,
            "HVap_actual": HVap_act,
            "HVap_pred": HVap_prd,
            "error = pred - actual": err,
            "T_ref": Tb_pred_all[idx],
            "HVap_ref": HVap_Tb_all[idx]
        })

long_compare = pd.DataFrame(rows).sort_values([id_col, "temp_index"])

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    long_compare.to_excel(writer, sheet_name="compare_long", index=False)

print(f"\n✅ 结果已保存到: {out_path}")