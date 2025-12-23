
#ds
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# ==== 常数与路径 ====
HV0, HVB, Tb0 = 9612.7, 15419.9, 222.543
T_ref = 298.15

# ==== 读取数据 ====
df_main = pd.read_excel("heat of vaporization 204.xlsx", sheet_name="Sheet1")
Nk_all = df_main.iloc[:, 13:32].apply(pd.to_numeric, errors='coerce')  # 19基团
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)

# ==== 读取需要的列 ====
temp_cols = list(df_main.columns[32:42])  # 10个温度列
hvap_cols = list(df_main.columns[42:52])  # 10个汽化焓列

# 数值化处理
for col in temp_cols + hvap_cols:
    df_main[col] = pd.to_numeric(df_main[col], errors='coerce')

# ==== Tb 模型 (参考温度预测模型) ====
Tb_raw = df_main.iloc[:, 5].values  # 读取参考温度列
mask_tb_ref = ~np.isnan(Tb_raw)  # 筛选参考温度有效数据
model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly[mask_tb_ref], np.exp(Tb_raw[mask_tb_ref] / Tb0))
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-6, None))  # 所有物质的参考温度预测

# ==== HVPb 模型 (汽化焓预测模型) ====
df_Tb = pd.read_excel("selected_25_descriptors_data_boiling_point.xlsx")
X_Tb = df_Tb.drop(columns=["Heat of vaporization at boiling temperature"])
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, df_Tb["Heat of vaporization at boiling temperature"])
HVap_Tb_all = rf_Tb.predict(X_Tb)  # 预测参考汽化焓

# ==== A_k 系数训练 ====
G = Nk_all.values # (n, 19) 基团数据
X_rows, y_rows = [], []
temp_eval = []  # 保存温度点信息用于评估

# 修正：正确构建训练集，处理NaN值
for i in range(len(df_main)):  # 遍历所有物质
    for j, (tcol, hvcol) in enumerate(zip(temp_cols, hvap_cols)):
        Tj = df_main.at[i, tcol]  # 温度值
        Hvapj = df_main.at[i, hvcol]  # 汽化焓值

        # 跳过NaN值
        if np.isnan(Tj) or np.isnan(Hvapj):
            continue

        Tb_i = Tb_pred_all[i]  # 物质i的参考温度
        HVap_Tb_i = HVap_Tb_all[i]  # 物质i的参考汽化焓

        # 特征：(T - T_ref) × G
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
A_vec = A_solver.coef_  # 长度19，对应基团列顺序

# ==== 生成汽化焓预测 ====
HVap_pred = pd.DataFrame(index=df_main.index, columns=hvap_cols, dtype=float)

# 修正：正确遍历所有物质和温度点
for i in range(len(df_main)):  # 遍历所有物质
    Tb_i = Tb_pred_all[i]  # 物质i的参考温度
    HVap_Tb_i = HVap_Tb_all[i]  # 物质i的参考汽化焓

    for j, (tcol, hvcol) in enumerate(zip(temp_cols, hvap_cols)):
        Tj = df_main.at[i, tcol]  # 温度值

        if np.isnan(Tj):
            HVap_pred.at[i, hvcol] = np.nan
            continue

        # 特征：(T - T_ref) × G
        Xj = (Tj - Tb_i) * G[i]

        # 预测：Hvap_ref + A_k × (T - T_ref) × G
        HVap_pred_j = HVap_Tb_i + Xj @ A_vec
        HVap_pred.at[i, hvcol] = HVap_pred_j

# ==== 评估模型性能 ====
y_true_all, y_pred_all = [], []
for hvcol in hvap_cols:
    m = (~df_main[hvcol].isna()) & (~HVap_pred[hvcol].isna())
    if m.any():
        y_true_all.append(df_main.loc[m, hvcol].to_numpy())
        y_pred_all.append(HVap_pred.loc[m, hvcol].to_numpy())

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

mse_hvap = mean_squared_error(y_true_all, y_pred_all)
r2_hvap = r2_score(y_true_all, y_pred_all)
print(f"MSE (on HVap): {mse_hvap:.6f}")
print(f"R2  (on HVap): {r2_hvap:.6f}")

# ==== 分温度点评估 ====
print("\n分温度点评估:")
for tcol, hvcol in zip(temp_cols, hvap_cols):
    m = (~df_main[tcol].isna()) & (~df_main[hvcol].isna()) & (~HVap_pred[hvcol].isna())
    if m.any():
        hvap_true = df_main.loc[m, hvcol].to_numpy()
        hvap_pred = HVap_pred.loc[m, hvcol].to_numpy()
        mse_temp = mean_squared_error(hvap_true, hvap_pred)
        r2_temp = r2_score(hvap_true, hvap_pred)
        print(f"  {tcol}: MSE = {mse_temp:.6f}, R2 = {r2_temp:.6f}")

# ==== 保存最终的结果 ====
id_col = df_main.columns[0]  # 物质ID/名称所在列
out_path = "hvap_actual_vs_pred_long.xlsx"

rows = []
for idx, _ in df_main.iterrows():
    ID = df_main.at[idx, id_col]
    for j, (tcol, hvcol) in enumerate(zip(temp_cols, hvap_cols), start=1):
        T = df_main.at[idx, tcol]
        HVap_act = df_main.at[idx, hvcol]
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