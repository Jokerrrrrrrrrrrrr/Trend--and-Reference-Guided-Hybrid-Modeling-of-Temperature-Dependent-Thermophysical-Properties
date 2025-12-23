import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares

# ==== 常数与路径 ====
HV0, HVB, Tb0 = 9612.7, 15419.9, 222.543
T_ref = 298.15

# ==== 读取主表 ====
df_main = pd.read_excel("heat of vaporization 204.xlsx", sheet_name="Sheet1")
Nk_all = df_main.iloc[:, 13:32].apply(pd.to_numeric, errors='coerce')  # 19基团
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all)

# ==== Tb 模型（exp 变换）====
Tb_raw = df_main.iloc[:, 5].values
mask_tb = ~np.isnan(Tb_raw)
Nk_valid_tb = Nk_all[mask_tb]
Nk_poly_valid_tb = poly.transform(Nk_valid_tb)
model_Tb = HuberRegressor(max_iter=10000).fit(Nk_poly_valid_tb, np.exp(Tb_raw[mask_tb] / Tb0))
Tb_pred_mask = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly_valid_tb), 1e-6, None))  # 仅 mask_tb 行

# ==== 读取 298K / Tb 特征数据并训练 RF（in-sample 预测即≈真实）====
df_298 = pd.read_excel("selected_25_descriptors_data_298.xlsx")
X_298 = df_298.drop(columns=["Heat of vaporization at normal temperature"])
y_298_true_all = df_298["Heat of vaporization at normal temperature"].values
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, y_298_true_all)
y_298_pred_all = rf_298.predict(X_298)

df_Tb = pd.read_excel("selected_25_descriptors_data_boiling_point.xlsx")
X_Tb = df_Tb.drop(columns=["Heat of vaporization at boiling temperature"])
y_Tb_true_all = df_Tb["Heat of vaporization at boiling temperature"].values
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, y_Tb_true_all)
y_Tb_pred_all = rf_Tb.predict(X_Tb)

# ==== 取多温度点、其它变量（仅 mask_tb 行）====
T = df_main.iloc[:, 32:42].values[mask_tb]
Hvap = df_main.iloc[:, 42:52].values[mask_tb]
MW = df_main.iloc[:, 4].values[mask_tb].reshape(-1, 1)
Nc = df_main.iloc[:, 10].values[mask_tb].reshape(-1, 1)

# ==== 行有效性（建议把温度也判空）====
valid_row_mask = np.isfinite(Hvap).all(axis=1) & np.isfinite(T).all(axis=1)
Nk_valid = Nk_valid_tb[valid_row_mask].values
MW = MW[valid_row_mask]
Nc = Nc[valid_row_mask]
T = T[valid_row_mask]
Hvap = Hvap[valid_row_mask]
Tb_pred = Tb_pred_mask[valid_row_mask]               # 预测 Tb（行向量）
Tb_true = Tb_raw[mask_tb][valid_row_mask]            # 真实 Tb（行向量）

# ==== 对齐 298/Tb 的真实与预测 Hvap（需与 df_main 行顺序一致）====
def align_vector(vec, name):
    """把外部向量 vec 对齐到 mask_tb 的行数；若 vec 长度等于 df_main 总行，就切 mask_tb；
       若已经等于 mask_tb.sum() 则直接用；否则报错提示检查行对齐。"""
    if len(vec) == mask_tb.sum():
        base = vec
    elif len(vec) == len(df_main):
        base = vec[mask_tb]
    else:
        raise ValueError(f"{name} 的长度为 {len(vec)}，无法与 mask_tb({mask_tb.sum()}) 或 df_main({len(df_main)}) 对齐，请检查数据顺序。")
    return base[valid_row_mask]

HVap_298_true = align_vector(y_298_true_all, "HVap_298_true")
HVap_Tb_true  = align_vector(y_Tb_true_all,  "HVap_Tb_true")
HVap_298_pred = align_vector(y_298_pred_all, "HVap_298_pred")
HVap_Tb_pred  = align_vector(y_Tb_pred_all,  "HVap_Tb_pred")

# ==== 四种 slope（每种是 shape=(n_rows,1) 的列向量）====
def safe_div(num, den):
    den = np.asarray(den, dtype=float)
    num = np.asarray(num, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(np.isfinite(den) & (den != 0), num / den, np.nan)
    return out.reshape(-1, 1)

den_pred = Tb_pred - T_ref
den_real = Tb_true - T_ref
num_real = HVap_Tb_true - HVap_298_true
num_pred = HVap_Tb_pred - HVap_298_pred

slope_variants = {
    "A_真实ΔHvap_预测ΔT": safe_div(num_real, den_pred),
    "B_预测ΔHvap_真实ΔT": safe_div(num_pred, den_real),
    "C_预测ΔHvap_预测ΔT": safe_div(num_pred, den_pred),
    "D_真实ΔHvap_真实ΔT": safe_div(num_real, den_real),
}

# ==== 构造（每种 slope）训练数据 ====
def build_Xy_for_slope(slope_col):
    X = np.hstack([
        Nk_valid.repeat(10, axis=0),          # 19
        MW.repeat(10, axis=0),               # +1 -> 20
        Nc.repeat(10, axis=0),               # +1 -> 21
        T.flatten().reshape(-1, 1),          # +1 -> 22
        slope_col.repeat(10, axis=0)         # +1 -> 23（最后一列为 slope）
    ])
    y = Hvap.flatten()
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return X[mask], y[mask], T.flatten()[mask], mask

# ==== 残差函数（与你原版一致）====
def residuals(params, X, y):
    Nk = X[:, :19]
    MW = X[:, 19].reshape(-1, 1)
    Nc = X[:, 20].reshape(-1, 1)
    T  = np.clip(X[:, 21].reshape(-1, 1), 1e-6, None)
    slope = X[:, 22].reshape(-1, 1)

    B1k = params[0:19]
    B2k = params[19:38]
    C1k = params[38:57]
    C2k = params[57:76]
    D1k = params[76:95]
    D2k = params[95:114]
    β, γ, δ = params[114:117]
    f0, f1 = params[117:119]
    γ_slope = params[119]
    intercept = params[120]

    R = 8.3144
    Bi = np.sum(Nk * (B1k + MW * B2k), axis=1, keepdims=True) + β * (f0 + Nc * f1)
    Ci = np.sum(Nk * (C1k + MW * C2k), axis=1, keepdims=True) + γ * (f0 + Nc * f1)
    Di = np.sum(Nk * (D1k + MW * D2k), axis=1, keepdims=True) + δ * (f0 + Nc * f1)

    y_pred = -R * ((1.5 * Bi) / np.sqrt(T) + Ci * T + Di * T**2) + γ_slope * slope * T + intercept
    return y_pred.flatten() - y

# ==== 拟合 & 评估 & 导出（循环四个变体）====
param_names = (
    [f"B1_{i}" for i in range(19)] + [f"B2_{i}" for i in range(19)] +
    [f"C1_{i}" for i in range(19)] + [f"C2_{i}" for i in range(19)] +
    [f"D1_{i}" for i in range(19)] + [f"D2_{i}" for i in range(19)] +
    ["beta", "gamma", "delta", "f0", "f1", "gamma_slope", "intercept"]
)

compound_ids_rows = df_main.iloc[mask_tb, 0].values[valid_row_mask]  # 每行一个物质ID

for tag, slope_col in slope_variants.items():
    # 1) 组装 X,y
    X, y, T_valid_flat, mask_points = build_Xy_for_slope(slope_col)

    # 2) 最小二乘拟合
    params_init = np.zeros(121)
    result = least_squares(residuals, x0=params_init, args=(X, y), max_nfev=10000)

    # 3) 评估
    y_pred = y - residuals(result.x, X, y)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    ard = np.mean(np.abs((y_pred - y) / y)) * 100
    rel = np.abs((y_pred - y) / y) * 100
    within_1pct  = int((rel <= 1).sum())
    within_5pct  = int((rel <= 5).sum())
    within_10pct = int((rel <= 10).sum())

    print(f"\n📈 主模型评估（{tag}，含 slope×T 和截距项）:")
    print(f"R²  = {r2:.6f}")
    print(f"MSE = {mse:.2f}")
    print(f"ARD = {ard:.2f}%")
    print(f"✅ 相对误差 ≤ 1% : {within_1pct}")
    print(f"✅ 相对误差 ≤ 5% : {within_5pct}")
    print(f"✅ 相对误差 ≤ 10%: {within_10pct}")

    # 4) 导出预测明细
    #    先把“行级”ID 展开为“10个温度点”的 ID，再用 mask_points 过滤
    compound_ids_rep = np.repeat(compound_ids_rows, 10)[mask_points]
    df_result = pd.DataFrame({
        "Compound_ID": compound_ids_rep,
        "Temperature (K)": T_valid_flat,
        "Hvap_true (J/mol)": y,
        "Hvap_pred (J/mol)": y_pred,
        "Absolute Error": np.abs(y - y_pred),
        "Relative Error (%)": 100 * np.abs((y - y_pred) / y)
    })
    out_pred_name = f"Hvap_prediction_{tag}.xlsx"
    df_result.to_excel(out_pred_name, index=False)
    print(f"✅ 预测结果已保存为 {out_pred_name}")

    # 5) 导出参数
    df_params = pd.DataFrame({"Parameter": param_names, "Value": result.x})
    out_param_name = f"Hvap_params_{tag}.xlsx"
    df_params.to_excel(out_param_name, index=False)
    print(f"🔧 参数表已保存为 {out_param_name}")
