import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. 读取数据
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.dropna(subset=[df.columns[0]])
df[df.columns[0]] = df[df.columns[0]].astype(int)

# 2. 列定义
group_cols = df.columns[11:30]   # 基团列
temp_cols  = df.columns[30:40]   # 10个温度点
cp_cols    = df.columns[40:50]   # 10个 Cp 值
target_column_T1 = 'ASPEN Half Critical T'
Tc0 = 138

# 真实四列：Cp1_true=第10列, Cp2_true=第51列, T1_true=target_column_T1, T2_true=1.5*T1_true
CP1_TRUE_IDX = 9
CP2_TRUE_IDX = 50
T1_TRUE_COL  = target_column_T1

# 2.1 强制数值化（避免 isnan 在 object 上报错）
df[group_cols] = df[group_cols].apply(pd.to_numeric, errors="coerce")
df[temp_cols]  = df[temp_cols].apply(pd.to_numeric, errors="coerce")
df[cp_cols]    = df[cp_cols].apply(pd.to_numeric, errors="coerce")
df.iloc[:, CP1_TRUE_IDX] = pd.to_numeric(df.iloc[:, CP1_TRUE_IDX], errors="coerce")
df.iloc[:, CP2_TRUE_IDX] = pd.to_numeric(df.iloc[:, CP2_TRUE_IDX], errors="coerce")
df[T1_TRUE_COL]          = pd.to_numeric(df[T1_TRUE_COL], errors="coerce")

# 3. 子模型训练：用于估算 T1, Cp1, Cp2 → 计算 slope
X_groups = df[group_cols]
valid_mask = X_groups.notna().all(1) & df[target_column_T1].notna()
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_groups[valid_mask])
y_exp_T1 = np.exp(df.loc[valid_mask, target_column_T1] / Tc0)

T1_model  = HuberRegressor(max_iter=9000).fit(X_poly, y_exp_T1)
Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups.fillna(0), df.iloc[:, CP1_TRUE_IDX].fillna(0))
Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups.fillna(0), df.iloc[:, CP2_TRUE_IDX].fillna(0))

# 3.1 子模型（in-sample）评估
y_pred_T1_exp = T1_model.predict(X_poly)
print("\n📌 子模型评估结果：")
print(f"T1_model ->     R²: {r2_score(y_exp_T1, y_pred_T1_exp):.4f} | MSE: {mean_squared_error(y_exp_T1, y_pred_T1_exp):.4f}")

mask_cp_eval = X_groups.notna().all(1) & df.iloc[:, CP1_TRUE_IDX].notna()
y_Cp1_true = df.iloc[:, CP1_TRUE_IDX][mask_cp_eval]
y_Cp1_pred = Cp1_model.predict(X_groups[mask_cp_eval].fillna(0))
print(f"Cp1_model ->    R²: {r2_score(y_Cp1_true, y_Cp1_pred):.4f} | MSE: {mean_squared_error(y_Cp1_true, y_Cp1_pred):.4f}")

mask_cp2_eval = X_groups.notna().all(1) & df.iloc[:, CP2_TRUE_IDX].notna()
y_Cp2_true = df.iloc[:, CP2_TRUE_IDX][mask_cp2_eval]
y_Cp2_pred = Cp2_model.predict(X_groups[mask_cp2_eval].fillna(0))
print(f"Cp2_model ->    R²: {r2_score(y_Cp2_true, y_Cp2_pred):.4f} | MSE: {mean_squared_error(y_Cp2_true, y_Cp2_pred):.4f}")

# 4. 构建训练数据 —— 四种 slope 变体
variants = {
    "A_真实ΔCp_预测ΔT": ("realCp", "predT"),
    "B_预测ΔCp_真实ΔT": ("predCp", "realT"),
    "C_预测ΔCp_预测ΔT": ("predCp", "predT"),
    "D_真实ΔCp_真实ΔT": ("realCp", "realT"),
}
datasets = {k: {"X": [], "y": [], "id": [], "T": []} for k in variants.keys()}

X_poly_all = poly.transform(X_groups.fillna(0))  # 用于逐行预测 T1(exp)

for i, row in df.iterrows():
    try:
        material_id = row.iloc[0]

        # —— 基团向量（确保数值）
        Nk_series = row[group_cols].astype(float)
        if pd.isna(Nk_series).any():
            continue
        Nk = Nk_series.values

        # —— 预测侧：T1_pred, T2_pred, Cp1_pred, Cp2_pred
        T1_exp_pred = float(T1_model.predict(X_poly_all[i:i+1])[0])
        if not np.isfinite(T1_exp_pred) or T1_exp_pred <= 0:
            continue
        T1_pred = Tc0 * np.log(T1_exp_pred)
        T2_pred = 1.5 * T1_pred

        Nk_df = pd.DataFrame([Nk], columns=group_cols).fillna(0)
        Cp1_pred = float(Cp1_model.predict(Nk_df)[0])
        Cp2_pred = float(Cp2_model.predict(Nk_df)[0])
        if not (np.isfinite(Cp1_pred) and np.isfinite(Cp2_pred)):
            continue

        # —— 真实侧：Cp1_true, Cp2_true, T1_true, T2_true
        Cp1_true = row.iloc[CP1_TRUE_IDX]
        Cp2_true = row.iloc[CP2_TRUE_IDX]
        T1_true  = row[T1_TRUE_COL]
        if not (np.isfinite(Cp1_true) and np.isfinite(Cp2_true) and np.isfinite(T1_true)):
            continue
        T2_true  = 1.5 * T1_true

        # —— 计算四种 slope
        num_den = {
            "A_真实ΔCp_预测ΔT": (Cp2_true - Cp1_true,  T2_pred - T1_pred),
            "B_预测ΔCp_真实ΔT": (Cp2_pred - Cp1_pred,  T2_true - T1_true),
            "C_预测ΔCp_预测ΔT": (Cp2_pred - Cp1_pred,  T2_pred - T1_pred),
            "D_真实ΔCp_真实ΔT": (Cp2_true - Cp1_true,  T2_true - T1_true),
        }
        slopes = {k: (np.nan if den == 0 else num/den) for k, (num, den) in num_den.items()}

        # —— 逐温度点展开
        temps = row[temp_cols].astype(float).values
        cps   = row[cp_cols].astype(float).values
        mask_pts = np.isfinite(temps) & np.isfinite(cps)
        if not mask_pts.any():
            continue

        for key in variants.keys():
            s = slopes[key]
            if not np.isfinite(s):
                continue
            for T, Cp in zip(temps[mask_pts], cps[mask_pts]):
                # 你的原始特征：Nk + T + slope*T
                feats = np.concatenate([Nk, [T], [s * T]])
                datasets[key]["X"].append(feats)
                datasets[key]["y"].append(Cp)
                datasets[key]["id"].append(material_id)
                datasets[key]["T"].append(T)
    except Exception as e:
        print(f"[WARN] row {i} skipped: {e}")
        continue

# 转数组 & 检查
for key in datasets:
    datasets[key]["X"] = np.asarray(datasets[key]["X"])
    datasets[key]["y"] = np.asarray(datasets[key]["y"])
    n = datasets[key]["X"].shape[0]
    print(f"🧪 {key} 样本数: {n}")
    if n == 0:
        raise RuntimeError(f"{key} 没有可用样本，请检查列类型与缺失。")

# 5. 拟合机器学习模型（随机森林）—— 四套
def eval_and_print(tag, model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    ard = np.mean(np.abs((y - y_pred) / y)) * 100
    rel = np.abs((y - y_pred) / y) * 100
    within_1  = int((rel <= 1).sum())
    within_5  = int((rel <= 5).sum())
    within_10 = int((rel <= 10).sum())

    print(f"\n📊 模型评估（{tag}，含 slope×T 特征）：")
    print(f"R²  = {r2:.4f}")
    print(f"MSE = {mse:.2f}")
    print(f"ARD = {ard:.2f}%")
    print(f"✅ 误差 ≤ 1% 的点数: {within_1}")
    print(f"✅ 误差 ≤ 5% 的点数: {within_5}")
    print(f"✅ 误差 ≤ 10% 的点数: {within_10}")
    return y_pred

models = {}
preds  = {}

for key in variants:
    X = datasets[key]["X"]; y = datasets[key]["y"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    models[key] = model
    preds[key]  = eval_and_print(key, model, X, y)

# 6. 保存预测结果（四份）
for key in variants:
    out = pd.DataFrame({
        "Material_ID": datasets[key]["id"],
        "Temperature (K)": datasets[key]["T"],
        "Cp_measured": datasets[key]["y"],
        "Cp_predicted": preds[key]
    })
    fname = f"Cp预测结果_RF_{key}.xlsx"
    out.to_excel(fname, index=False)
    print(f"✅ 已保存预测结果为: {fname}")
