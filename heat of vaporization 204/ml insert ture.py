import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

# ==== 1. 读取数据 ====
df = pd.read_excel("heat of vaporization 204.xlsx", sheet_name="Sheet1")

# ==== 2. 定义列 ====
group_cols = df.columns[13:32]   # 19 个基团（修正为 13:32）
temp_cols  = df.columns[32:42]   # 10 个温度点
hvap_cols  = df.columns[42:52]   # 10 个 Hvap

# ==== 3. 准备 slope 所需模型输入（298K 与 Tb 的真实与预测）====
# 3.1 298K
df_298 = pd.read_excel("selected_25_descriptors_data_298.xlsx")
X_298 = df_298.drop(columns=["Heat of vaporization at normal temperature"])
y_298_true_all = df_298["Heat of vaporization at normal temperature"].values
rf_298 = RandomForestRegressor(random_state=42).fit(X_298, y_298_true_all)
y_298_pred_all = rf_298.predict(X_298)

# 3.2 沸点 Tb
df_Tb = pd.read_excel("selected_25_descriptors_data_boiling_point.xlsx")
X_Tb = df_Tb.drop(columns=["Heat of vaporization at boiling temperature"])
y_Tb_true_all = df_Tb["Heat of vaporization at boiling temperature"].values
rf_Tb = RandomForestRegressor(random_state=42).fit(X_Tb, y_Tb_true_all)
y_Tb_pred_all = rf_Tb.predict(X_Tb)

# ==== 4. Tb 模型预测（按主表基团构建）====
Tb0 = 222.543
Nk_all = df.iloc[:, 13:32].apply(pd.to_numeric, errors='coerce')  # 19 基团（与 group_cols 一致）
Tb_raw = df.iloc[:, 5].values                                    # 真实 Tb（主表第 6 列）
poly = PolynomialFeatures(degree=2, include_bias=False)
Nk_poly = poly.fit_transform(Nk_all.fillna(0))

mask_tb = ~np.isnan(Tb_raw)
model_Tb = HuberRegressor(max_iter=5000).fit(
    Nk_poly[mask_tb], np.exp(Tb_raw[mask_tb] / Tb0)
)
Tb_pred_all = Tb0 * np.log(np.clip(model_Tb.predict(Nk_poly), 1e-9, None))  # 全行预测 Tb

# ==== 5. 计算四种 slope ====
T_ref = 298.15

def align_vector(vec, name):
    """将外部向量与 df 的行对齐。
       - 若 len(vec)==len(df): 直接返回
       - 否则抛错，提醒检查数据来源顺序"""
    if len(vec) == len(df):
        return np.asarray(vec, dtype=float)
    raise ValueError(f"{name} 的长度为 {len(vec)}，与主表行数 {len(df)} 不一致，请检查文件行顺序。")

# 若 298/Tb 两个特征文件与主表一一对齐，这里直接用；否则请先在外部对齐
HVap_298_true = align_vector(y_298_true_all, "HVap_298_true")
HVap_Tb_true  = align_vector(y_Tb_true_all,  "HVap_Tb_true")
HVap_298_pred = align_vector(y_298_pred_all, "HVap_298_pred")
HVap_Tb_pred  = align_vector(y_Tb_pred_all,  "HVap_Tb_pred")

num_real = HVap_Tb_true - HVap_298_true
num_pred = HVap_Tb_pred - HVap_298_pred
den_real = Tb_raw - T_ref
den_pred = Tb_pred_all - T_ref

def safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(np.isfinite(den) & (den != 0), num / den, np.nan)
    return out

slope_variants = {
    "A_真实ΔHvap_预测ΔT": safe_div(num_real, den_pred),
    "B_预测ΔHvap_真实ΔT": safe_div(num_pred, den_real),
    "C_预测ΔHvap_预测ΔT": safe_div(num_pred, den_pred),  # 原脚本定义
    "D_真实ΔHvap_真实ΔT": safe_div(num_real, den_real),
}

# ==== 6. 为每个变体构建训练数据、训练与评估 ====
def build_dataset(slope_vec):
    X_total, y_total, material_ids, temperatures = [], [], [], []
    slope_arr = np.asarray(slope_vec, dtype=float)

    for i, row in df.iterrows():
        try:
            material_id = row.iloc[0]
            Nk = row[group_cols].astype(float).values
            temps = row[temp_cols].astype(float).values
            hvaps = row[hvap_cols].astype(float).values
            slope = float(slope_arr[i])

            if np.isnan(slope) or np.isnan(Nk).any():
                continue

            mask_pts = np.isfinite(temps) & np.isfinite(hvaps)
            if not mask_pts.any():
                continue

            for T, Hv in zip(temps[mask_pts], hvaps[mask_pts]):
                # 你的原始特征：Nk + T + slope  （注意：这里没有 slope×T）
                features = np.concatenate([Nk, [T], [slope]])
                X_total.append(features)
                y_total.append(Hv)
                material_ids.append(material_id)
                temperatures.append(T)
        except Exception as e:
            # 某些行异常就跳过
            continue

    X_total = np.array(X_total, dtype=float)
    y_total = np.array(y_total, dtype=float)
    return X_total, y_total, material_ids, temperatures

def train_eval_export(tag, slope_vec):
    X_total, y_total, material_ids, temperatures = build_dataset(slope_vec)
    n = X_total.shape[0]
    print(f"\n🧪 {tag} 样本数: {n}")
    if n == 0:
        print(f"[WARN] {tag} 无可用样本，跳过。")
        return

    # 训练
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_total, y_total)

    # 评估
    y_pred = model.predict(X_total)
    r2  = r2_score(y_total, y_pred)
    mse = mean_squared_error(y_total, y_pred)
    ard = np.mean(np.abs((y_pred - y_total) / y_total)) * 100

    print(f"\n📊 模型评估（{tag}；特征=基团 + 温度 + slope）：")
    print(f"R²  = {r2:.4f}")
    print(f"MSE = {mse:.2f}")
    print(f"ARD = {ard:.2f}%")

    rel = np.abs((y_pred - y_total) / y_total) * 100
    print(f"相对误差 ≤ 1% 的点数: {(rel <= 1).sum()}")
    print(f"相对误差 ≤ 5% 的点数: {(rel <= 5).sum()}")
    print(f"相对误差 ≤ 10% 的点数: {(rel <= 10).sum()}")

    # 保存结果
    results = pd.DataFrame({
        "Material_ID": material_ids,
        "Temperature (K)": temperatures,
        "Hvap_measured": y_total,
        "Hvap_predicted": y_pred,
        "Absolute Error": np.abs(y_total - y_pred),
        "Relative Error (%)": 100 * np.abs((y_total - y_pred) / y_total)
    })
    out_name = f"Hvap预测结果_RF_{tag}.xlsx"
    results.to_excel(out_name, index=False)
    print(f"✅ 已保存预测结果为: {out_name}")

# 逐个变体运行
for tag, slope_vec in slope_variants.items():
    train_eval_export(tag, slope_vec)
