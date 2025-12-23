# import pandas as pd
# import numpy as np
# from sklearn.linear_model import HuberRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
# # from sklearn.linear_model import Ridge  # 可选：用岭回归替代 Huber，增强稳定性
#
# # ========= 1. 读取数据 =========
# file_path = "heat capacity 207.xlsx"
# sheet = "Sheet1"
# df = pd.read_excel(file_path, sheet_name=sheet).copy()
#
# df = df.dropna(subset=[df.columns[0]])
# df[df.columns[0]] = df[df.columns[0]].astype(int)
#
# # ========= 2. 列定义 =========
# group_cols = list(df.columns[11:30])   # 19个基团列
# temp_cols  = list(df.columns[30:40])   # 10个温度点
# cp_cols    = list(df.columns[40:50])   # 10个 Cp 值
# target_column_T1 = 'ASPEN Half Critical T'
#
# # 做一个温度列 -> 对应Cp列 的映射（索引一一对应）
# temp_to_cp = {t: c for t, c in zip(temp_cols, cp_cols)}
#
# # 数值化
# for cols in [group_cols, temp_cols, cp_cols]:
#     for c in cols:
#         df[c] = pd.to_numeric(df[c], errors='coerce')
#
# # ========= 3. 子模型训练：T_ref(=T1) 与 C_pref(=Cp1) =========
# X_groups = df[group_cols].fillna(0)
#
# valid_mask = ~df[target_column_T1].isna()
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly_train = poly.fit_transform(X_groups[valid_mask])
#
# y_T1 = df.loc[valid_mask, target_column_T1].to_numpy()
# T1_model = GradientBoostingRegressor(
#     n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
# ).fit(X_poly_train, y_T1)
#
# # 对所有样本预测 T_ref
# X_poly_all = poly.transform(X_groups)
# T_ref_pred = T1_model.predict(X_poly_all)  # (n,)
#
# # Cp1模型（示例里用第9列作为目标）
# y_cp1_target = df.iloc[:, 9].to_numpy()
# Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, y_cp1_target)
#
# # 对所有样本预测 C_pref
# C_pref_pred = Cp1_model.predict(X_groups)  # (n,)
#
# # ========= 4. 构造 A_k 的训练集（物质×温度点展开）=========
# G = X_groups.to_numpy()  # (n, 19)
# X_rows, y_rows = [], []
# temp_eval = []  # 保存 (tcol, cpcol, msk) 以便分温度评估
#
# for tcol, cpcol in zip(temp_cols, cp_cols):
#     Tj  = df[tcol].to_numpy()        # (n,)
#     CPj = df[cpcol].to_numpy()       # (n,)
#     msk = (~np.isnan(Tj)) & (~np.isnan(CPj))
#     if msk.sum() == 0:
#         continue
#
#     # 特征：(T - T_ref)[:, None] * G  → (n_j, 19)
#     Xj = ((Tj - T_ref_pred)[:, None] * G)[msk]
#     # 目标：Cp - C_pref
#     yj = (CPj - C_pref_pred)[msk]
#
#     X_rows.append(Xj)
#     y_rows.append(yj)
#     temp_eval.append((tcol, cpcol, msk))
#
# X_A = np.vstack(X_rows)          # (sum_j n_j, 19)
# y_A = np.concatenate(y_rows)     # (sum_j n_j,)
#
# # ========= 5. 拟合 A_k（无截距；截距由 C_pref 承担）=========
# A_solver = HuberRegressor(fit_intercept=False, max_iter=5000)
# # A_solver = Ridge(fit_intercept=False, alpha=1.0)  # 可切换
# A_solver.fit(X_A, y_A)
# A_vec = A_solver.coef_           # 长度19，对应 group_cols 顺序
#
# # ========= 6. 评价（整体 + 分温度点）=========
# y_hat_all = A_solver.predict(X_A)
# print(f"[ALL] MSE = {mean_squared_error(y_A, y_hat_all):.6f}, R2 = {r2_score(y_A, y_hat_all):.6f}")
#
# print("\nPer-temperature metrics:")
# for tcol, cpcol, msk in temp_eval:
#     Tj  = df[tcol].to_numpy()
#     CPj = df[cpcol].to_numpy()
#     Xj  = ((Tj - T_ref_pred)[:, None] * G)[msk]
#     yj  = (CPj - C_pref_pred)[msk]
#     yj_hat = Xj @ A_vec
#     print(f"  {tcol}: MSE = {mean_squared_error(yj, yj_hat):.6f}, R2 = {r2_score(yj, yj_hat):.6f}")
#
#
#
# # ========= 7. 输出 A_k 系数表 =========
# A_df = pd.Series(A_vec, index=group_cols, name="A_k").to_frame()
# print("\nEstimated group coefficients A_k:")
# print(A_df)
#
# # ========= 8. 生成全表的 Cp 预测，便于对比 =========
# Cp_pred = pd.DataFrame(index=df.index, columns=cp_cols, dtype=float)
# for tcol, cpcol in zip(temp_cols, cp_cols):
#     Tj = df[tcol].to_numpy()
#     Xj = (Tj - T_ref_pred)[:, None] * G
#     Cp_pred_j = C_pref_pred + Xj @ A_vec
#     Cp_pred[cpcol] = np.where(np.isnan(Tj), np.nan, Cp_pred_j)
#
# print("\nPreview of predicted Cp (first 5 rows):")
# print(Cp_pred.head())
#
# # ========= 9. （可选）保存 =========
# # A_df.to_excel("A_k_coefficients.xlsx")
# # Cp_pred.to_excel("Cp_pred_matrix.xlsx")
# # pd.DataFrame({"T_ref_pred": T_ref_pred, "C_pref_pred": C_pref_pred}).to_excel("reference_point_predictions.xlsx", index=False)
# # ===== 将“每物质10个温度点各占一行”的实际 vs 预测保存为 Excel =====
# id_col = df.columns[0]  # 物质ID/名称所在列
# out_path = "cp_actual_vs_pred_long.xlsx"
#
# rows = []
# for idx, _ in df.iterrows():
#     ID = df.at[idx, id_col]
#     for j, (tcol, cpcol) in enumerate(zip(temp_cols, cp_cols), start=1):
#         T = df.at[idx, tcol]
#         Cp_act = df.at[idx, cpcol]
#         Cp_prd = Cp_pred.at[idx, cpcol]
#         err = (Cp_prd - Cp_act) if (pd.notna(Cp_prd) and pd.notna(Cp_act)) else np.nan
#         rows.append({
#             id_col: ID,
#             "temp_index": j,          # 1~10，保证每个物质恰好10行
#             "temp_col": tcol,         # 温度列名
#             "T": T,                   # 该行温度值
#             "Cp_actual": Cp_act,      # 实际Cp
#             "Cp_pred": Cp_prd,        # 预测Cp
#             "error = pred - actual": err
#             # 如需同时放入参考点，解除下两行注释
#             # "T_ref": T_ref_pred[idx],
#             # "C_pref": C_pref_pred[idx],
#         })
#
# long_compare = pd.DataFrame(rows).sort_values([id_col, "temp_index"])
#
# with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
#     long_compare.to_excel(writer, sheet_name="compare_long", index=False)
# ===== 完成，文件在当前工作目录：cp_actual_vs_pred_long.xlsx =====
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.linear_model import Ridge  # 可选：用岭回归替代 Huber，增强稳定性

# ========= 1. 读取数据 =========
file_path = "heat capacity 207.xlsx"
sheet = "Sheet1"
df = pd.read_excel(file_path, sheet_name=sheet).copy()

df = df.dropna(subset=[df.columns[0]])
df[df.columns[0]] = df[df.columns[0]].astype(int)

# ========= 2. 列定义 =========
group_cols = list(df.columns[11:30])   # 19个基团列
temp_cols  = list(df.columns[30:40])   # 10个温度点
cp_cols    = list(df.columns[40:50])   # 10个 Cp 值
target_column_T1 = 'ASPEN Half Critical T'

# 做一个温度列 -> 对应Cp列 的映射（索引一一对应）
temp_to_cp = {t: c for t, c in zip(temp_cols, cp_cols)}

# 数值化
for cols in [group_cols, temp_cols, cp_cols]:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ========= 3. 子模型训练：T_ref(=T1) 与 C_pref(=Cp1) =========
X_groups = df[group_cols].fillna(0)

valid_mask = ~df[target_column_T1].isna()
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_groups[valid_mask])

y_T1 = df.loc[valid_mask, target_column_T1].to_numpy()
T1_model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0
).fit(X_poly_train, y_T1)

# 对所有样本预测 T_ref
X_poly_all = poly.transform(X_groups)
T_ref_pred = T1_model.predict(X_poly_all)  # (n,)

# Cp1模型（示例里用第9列作为目标）
y_cp1_target = df.iloc[:, 9].to_numpy()
Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, y_cp1_target)

# 对所有样本预测 C_pref
C_pref_pred = Cp1_model.predict(X_groups)  # (n,)

# ========= 4. 构造 A_k 的训练集（物质×温度点展开）=========
G = X_groups.to_numpy()  # (n, 19)
X_rows, y_rows = [], []
temp_eval = []  # 保存 (tcol, cpcol, msk) 以便分温度评估

for tcol, cpcol in zip(temp_cols, cp_cols):
    Tj  = df[tcol].to_numpy()        # (n,)
    CPj = df[cpcol].to_numpy()       # (n,)
    msk = (~np.isnan(Tj)) & (~np.isnan(CPj))
    if msk.sum() == 0:
        continue

    # 特征：(T - T_ref)[:, None] * G  → (n_j, 19)
    Xj = ((Tj - T_ref_pred)[:, None] * G)[msk]
    # 目标：Cp - C_pref（用于训练A）
    yj = (CPj - C_pref_pred)[msk]

    X_rows.append(Xj)
    y_rows.append(yj)
    temp_eval.append((tcol, cpcol, msk))

X_A = np.vstack(X_rows)          # (sum_j n_j, 19)
y_A = np.concatenate(y_rows)     # (sum_j n_j,)

# ========= 5. 拟合 A_k（无截距；截距由 C_pref 承担）=========
A_solver = HuberRegressor(fit_intercept=False, max_iter=5000)
# A_solver = Ridge(fit_intercept=False, alpha=1.0)  # 可切换
A_solver.fit(X_A, y_A)
A_vec = A_solver.coef_           # 长度19，对应 group_cols 顺序

# ========= 6. 生成全表的 Cp 预测 =========
Cp_pred = pd.DataFrame(index=df.index, columns=cp_cols, dtype=float)
for tcol, cpcol in zip(temp_cols, cp_cols):
    Tj = df[tcol].to_numpy()
    Xj = (Tj - T_ref_pred)[:, None] * G
    Cp_pred_j = C_pref_pred + Xj @ A_vec
    Cp_pred[cpcol] = np.where(np.isnan(Tj), np.nan, Cp_pred_j)

# ========= 7. 评估（基于 Cp 本身）=========
# 7.1 整体（把所有温度点拼起来）
y_true_all, y_pred_all = [], []
for cpcol in cp_cols:
    m = (~df[cpcol].isna()) & (~Cp_pred[cpcol].isna())
    if m.any():
        y_true_all.append(df.loc[m, cpcol].to_numpy())
        y_pred_all.append(Cp_pred.loc[m, cpcol].to_numpy())

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

mse_cp = mean_squared_error(y_true_all, y_pred_all)
r2_cp  = r2_score(y_true_all, y_pred_all)
print(f"MSE (on Cp): {mse_cp:.6f}")
print(f"R2  (on Cp): {r2_cp:.6f}")

# 7.2 （可选）按温度列分别评估（同样在 Cp 上）
print("\nPer-temperature metrics (on Cp):")
for tcol, cpcol, msk in temp_eval:
    cp_true = df[cpcol].to_numpy()[msk]
    cp_pred = Cp_pred[cpcol].to_numpy()[msk]
    print(f"  {tcol}: MSE = {mean_squared_error(cp_true, cp_pred):.6f}, "
          f"R2 = {r2_score(cp_true, cp_pred):.6f}")

# ========= 8. 生成“每物质×10温度点一行”的实际 vs 预测（Excel）
id_col = df.columns[0]  # 物质ID/名称所在列
out_path = "cp_actual_vs_pred_long.xlsx"

rows = []
for idx, _ in df.iterrows():
    ID = df.at[idx, id_col]
    for j, (tcol, cpcol) in enumerate(zip(temp_cols, cp_cols), start=1):
        T = df.at[idx, tcol]
        Cp_act = df.at[idx, cpcol]
        Cp_prd = Cp_pred.at[idx, cpcol]
        err = (Cp_prd - Cp_act) if (pd.notna(Cp_prd) and pd.notna(Cp_act)) else np.nan
        rows.append({
            id_col: ID,
            "temp_index": j,
            "temp_col": tcol,
            "T": T,
            "Cp_actual": Cp_act,
            "Cp_pred": Cp_prd,
            "error = pred - actual": err
        })

long_compare = pd.DataFrame(rows).sort_values([id_col, "temp_index"])

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    long_compare.to_excel(writer, sheet_name="compare_long", index=False)
